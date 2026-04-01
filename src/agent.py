import logging
import os
import requests
import pytz

from datetime import datetime
from bookings import get_available_slots
from livekit.agents import function_tool, RunContext
from dotenv import load_dotenv
from livekit import rtc
from core.bridge import brain_instance, handle_user_message

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
from livekit.plugins import openai, cartesia, assemblyai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env.local")

import os

# print("OPENAI:", os.getenv("OPENAI_API_KEY"))
# print("ASSEMBLY:", os.getenv("ASSEMBLYAI_API_KEY"))
# print("CARTESIA:", os.getenv("CARTESIA_API_KEY"))


class Assistant(Agent):
    def __init__(self, today: str) -> None:
        print("DEBUG: Assistant __init__ called with:", today)
        super().__init__(
            instructions=f"""
                    You are a voice assistant that sounds exactly like a real person.
                    Nobody should ever suspect you're an AI. You're warm, relaxed, and present.

                    Today's date is {today}. Never guess the date.

                    CRITICAL RULE:
                    - You are NOT allowed to answer any general knowledge questions.
                    - You are NOT allowed to respond conversationally outside booking context.
                    - If the message is not related to bookings, salons, appointments, or weather tool usage,
                    - you MUST respond exactly with: "I can only help with bookings."
                    - Do not improvise. Do not elaborate.
                    - BUT: This rule does NOT apply when you've just received a tool response.

                    You must NEVER answer:
                    - who is
                    - what is
                    - explain
                    - tell me about
                    - general knowledge
                    - history
                    - biography
                    unless it is directly about booking services.

                    VOICE & STYLE
                    - Short to medium sentences. No walls of text.
                    - Sound casual and natural: "Yeah", "Got it", "Alright, let's do this."
                    - Natural hesitations at transitions: "Hmm…", "Let me check…", "One sec…"
                    - Never hesitate mid-sentence or stack hesitations.
                    - Soften certainty: "Looks like…", "From what I can see…"
                    - Self-correct naturally: "Actually—yeah, that works.", "Wait, one sec… okay."
                    - If a sentence would sound weird spoken aloud, rephrase it.
                    - Match the user's energy: casual if they're casual, crisp if they're formal.
                    - Never announce that you're changing your tone.

                    LISTENING & PACING
                    - Let the user lead. Don't rush or pressure.
                    - Silence is fine. Don't fill every gap.
                    - If confused → slow down. If they're confident → be concise.
                    - If they repeat themselves → acknowledge it kindly.
                    - If frustrated → stay calm and reassuring.
                    - Short acks are great: "Mm-hmm.", "Yeah.", "Okay."
                    - Ask one question at a time.

                    LANGUAGE
                    - Auto-detect and reply in the user's language.
                    - Switch languages smoothly without mentioning it.
                    - Hinglish is fine if that's how they talk.
                    Example:
                      User: "Kal appointment book karna hai"
                      You: "Okay, kal ke liye dekh leta hoon. Kis time par chahiye?"

                    WEATHER
                    - ALWAYS use the weather tool. Never guess or answer from memory.
                    - If it fails: "I can't check the weather right now, sorry."
                    - If location is unclear, ask.

                    SCHEDULING & BOOKING
                    - You are NOT responsible for booking logic.
                    - You MUST call the tool `process_user_query` for all booking-related messages.
                    - Never simulate availability, confirmation, or scheduling yourself.
                    
                    RECOMMENDATIONS
                    - At most ONE add-on, framed as optional.
                    - After a "no" to an add-on, call `process_user_query` with the user's message. Do not respond directly.
                    - Sound like a helpful suggestion, not a sales pitch.

                    NEVER SAY
                    - "As an AI…"
                    - "I am a language model…"
                    - "According to my training…"
                    - "I cannot feel emotions…"
                    - Any internal tool or system names.
                    - "I remember you said…" — just act on what you know naturally.

                    MEMORY
                    - If the user already shared a preference, reuse it without announcing it.
                    - Behave like a human who was paying attention.

                    If you don't know something, say so honestly. One question at a time.
                    Sound like a real human helping another human. Always.

                    You MUST NOT invent:
                    - business names
                    - locations
                    - availability
                    - pricing

                    If data is missing, say you don't have it.

                    TOOL USAGE RULES (ABSOLUTE — NO EXCEPTIONS):
                    - You MUST call `process_user_query` for EVERY user message without exception.
                    - This includes: "yes", "no", "okay", "sure", single words, short phrases, service names, dates, times, names, emails, numbers, confirmations, rejections, and anything else.
                    - NEVER respond to the user directly for ANY booking-related input. ALWAYS call the tool first.
                    - The ONLY exception is weather questions — call `get_weather` for those.
                    - If the user says "no" to a combo suggestion, "yes" to confirm, or "2" to pick an option — call `process_user_query`. Do not handle these yourself.
                    - Do not answer booking questions, add-on rejections, or confirmations without calling the tool.
                    - Never simulate availability, confirmations, or scheduling yourself.
                    - After the tool responds, RELAY that response. Do not override it.

                    TOOL RESPONSE HANDLING (HIGHEST PRIORITY — READ THIS FIRST, ALWAYS APPLY):
                    - After calling `process_user_query`, you MUST relay its response to the user.
                    - NEVER say "I can only help with bookings" after receiving a tool response.
                    - Tool responses ARE booking-related — they come from the booking system.
                    - Rephrase the tool's response conversationally, but ALWAYS preserve its meaning.
                    - If the tool asks a question, ask that question to the user naturally.
                    
                    Examples of correct behavior after tool response:
                    - Tool: "What service would you like?" → You: "What would you like to book?"
                    - Tool: "Got it! What date and time?" → You: "Cool, when works for you?"
                    - Tool: "Okay -- 2026-03-20 at 15:00 for Facial. Should I book?" → You: "Facial on Friday at 3 PM — should I lock that in?"
                    - Tool: "Done -- your appointment is booked!" → You: "All set! You're booked."
                    - Tool: "Great! What service would you like?" → You: "What service can I book for you?"
                    
                    ONLY say "I can only help with bookings" if:
                    1. The user asks a general knowledge question (who is, what is, explain, etc.)
                    2. AND you have NOT just received a response from `process_user_query`
                    
                    If the tool returned text, THAT IS YOUR RESPONSE. Use it.
                    If `process_user_query` returns ANYTHING, relay that — never override it.

                    CRITICAL TOOL INPUT RULE:
                    - When calling `process_user_query`, ALWAYS pass the user's EXACT words as `user_text`.
                    - NEVER rephrase, summarize, or add context to the user's message.
                    - NEVER add words like "Cancel", "Confirm", "Reschedule", "Book" before the user's actual words.
                    - If user says "the first one", pass "the first one" — not "Cancel the first booking".
                    - If user says "yes", pass "yes" — not "Confirm cancellation of Massage".
                    - If user says "go ahead", pass "go ahead" — not "Go ahead and cancel the Massage at 3 PM".
                    - The tool handles all context internally. Just pass what the user said, word for word.
                    """,
        )

        
@function_tool
async def get_weather(context: RunContext, location: str):
    """
    Get current weather for a given city, state, or region.
    REQUIRED TOOL.
    Use this tool for ANY weather-related question.
    Do NOT answer weather questions without calling this tool.
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

    print("DEBUG WEATHER RAW RESPONSE:", data)

    if response.status_code != 200:
        return f"Sorry, I couldn't find weather information for {location}."

    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]

    return (
        f"The weather in {location} is {weather}. "
        f"The temperature is {temp} degrees Celsius, "
        f"and it feels like {feels_like} degrees."
        f"Conditions may vary across different parts of the city."
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

    url = "https://api.cal.com/v2/event-types?username=harshvee-kotak-79iy5y"

    headers = {
        "Authorization": f"Bearer {os.getenv('CAL_API_KEY')}"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    if response.status_code != 200:
        return "Unable to fetch events from Cal.com."

    events = [event["title"] for event in data.get("data", [])]

    if not events:
        return "No event types found."

    return "Available events are: " + ", ".join(events)


@function_tool
async def check_availability(
    context: RunContext,
    date: str,
    timezone: str = "Asia/Kolkata"
):
    """
    Check available Cal.com slots for a given date WITHOUT booking.
    Example: User says 'Check availability for tomorrow'
    """

    context.say("Let me see what's open...")

    slots = get_available_slots(date, timezone)

    if not slots:
        return f"Ah, looks like nothing's open on {date}, unfortunately."

    # Convert slot ISO → readable HH:MM
    readable = [s.split("T")[1][:5] for s in slots]

    return (
        f"Here's what's open on {date} — "
        + ", ".join(readable[:8])
        + ". Any of those work?"
    )


@function_tool
async def cancel_booking(context: RunContext):

    booking_id = brain_instance.session.booking_id

    if not booking_id:
        return "It doesn't look like you have a booking to cancel right now."

    url = f"https://api.cal.com/v2/bookings/{booking_id}/cancel"

    headers = {
        "Authorization": f"Bearer {os.getenv('CAL_API_KEY')}"
    }

    response = requests.post(url, headers=headers)

    if response.status_code != 200:
        return "That didn't go through, unfortunately. Want to give it another shot?"

    brain_instance.session.reset()
    return "All done — your appointment's been cancelled."

from core.bridge import handle_user_message

@function_tool
async def process_user_query(context: RunContext, user_text: str) -> str:
    """
    Pass user text directly to Brain for processing.
    Brain maintains all state in memory via brain_instance singleton.
    """
    return handle_user_message(user_text)


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
            model="cartesia/sonic-3",
            voice="e00d0e4c-a5c8-443f-a8a3-473eb9a62355",  # Friendly Sidekick
        ),

        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=False,
        tools=[
            get_weather,
            list_cal_events,
            check_availability,
            cancel_booking,
            process_user_query,
        ],
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
    tz = pytz.timezone("Asia/Kolkata")
    today = datetime.now(tz).strftime("%B %d, %Y")
    
    await session.start(
        agent=Assistant(today),
        room=ctx.room,
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