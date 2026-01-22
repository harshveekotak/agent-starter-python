import logging
import os
import requests

from bookings import check_slots, create_cal_booking
from utils import local_to_utc
from livekit.agents import function_tool, RunContext
from dotenv import load_dotenv
from livekit import rtc

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

booking_state = {}

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                    You are a warm, natural, human‑like voice assistant.
                    Speaking style rules:
                    - Speak like a real person, not a robot
                    - Use short sentences
                    - Add natural pauses when speaking
                    - Sometimes say things like "hmm", "okay", "let me think"
                    - Do not rush your answers
                    - If the user sounds confused, slow down
                    - Be friendly and reassuring

                    Human behavior rules:
                    - Avoid long explanations unless asked
                    - React naturally to emotions (happy, confused, frustrated)
                    - Be calm and reassuring if the user sounds unsure
                    - If the user pauses, wait patiently and do not interrupt

                    Language behavior:
                    - Detect the user's language automatically
                    - Reply in the same language as the user
                    - If the user switches languages, switch smoothly
                    - If unsure, politely ask which language they prefer

                    General rules:
                    - Be friendly and approachable
                    - Keep answers concise but helpful
                    - Ask one question at a time
                    - If you don't know something, admit it honestly
                    - Never make up information
                    - Always prioritize user safety and comfort
                    - If the user seems distressed, offer help calmly
                    - If the user asks for personal info, decline politely
                    - Be helpful and apologetic when needed

                    Weather rules:
                    - If the user asks about weather, temperature, rain, heat, or climate,
                    always use the weather tool.
                    - Do not guess weather information.
                    - If the location is unclear, ask the user to clarify.

                    Scheduling rules:
                    - If the user asks about meetings, appointments, salon, or booking you may check available event types.
                    - Use the calendar tool to list event types.
                    - Ask for preferred date and time.
                    - If the time is unavailable, suggest alternatives
                    - If the user sounds urgent, suggest the earliest slot.
                    - Do not schedule or book events directly.  
                    - Always confirm with the user before taking any action.
                    - Ask for name, email, and phone politely.
                    - Never say a booking is confirmed unless it is actually booked.
                    """,
        )


@function_tool
async def get_weather(context: RunContext, location: str):
    """
    Get current weather for a given city, state, or region.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather service is not configured."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        return f"Sorry, I couldn't find weather information for {location}."

    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]

    return (
        f"The weather in {location} is {weather}. "
        f"The temperature is {temp} degrees Celsius, "
        f"and it feels like {feels_like} degrees."
    )



@function_tool
async def list_cal_events(context: RunContext):
    """
    List available Cal.com event types.
    Use this tool to check what events can be booked.
    """

    api_key = os.getenv("CAL_API_KEY")
    if not api_key:
        return "Cal.com API key is not configured."

    url = "https://api.cal.com/v1/event-types"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    if response.status_code != 200:
        return "Unable to fetch events from Cal.com."

    events = [event["title"] for event in data.get("event_types", [])]

    if not events:
        return "No event types found."

    return "Available events are: " + ", ".join(events)



@function_tool
async def start_booking(
    context: RunContext,
    name: str,
    email: str,
    phone: str,
    date: str,
    time: str,
    timezone: str = "Asia/Kolkata",
    urgency: str = "normal"
):
    # 1. Check slot availability
    available, slots = check_slots(time)

    if not available:
        if urgency == "urgent":
            return (
                f"I don’t have that time available. "
                f"The earliest available slot is {slots[0]}. "
                f"Would you like me to book that?"
            )

        return (
            f"That time isn’t available. "
            f"I can offer {', '.join(slots)}. "
            f"Which one works for you?"
        )

    # 2. Convert to UTC
    start_utc = local_to_utc(date, time, timezone)

    # 3. Create booking via Cal.com
    result = create_cal_booking(
        start_utc,
        name,
        email,
        phone,
        timezone
    )

    if "error" in result:
        return (
            "Sorry, I couldn’t complete the booking right now. "
            "Would you like to try a different time?"
        )

    return (
        f"Your appointment is confirmed for {date} at {time}. "
        f"You’ll receive a confirmation email shortly."
    )



server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        tools=[get_weather, start_booking],
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
