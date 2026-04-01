from core.brain import Brain

def run_cli():

    brain = Brain()

    print("\nAgent running in CMD mode (type 'exit' to quit)\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() in ("exit", "quit"):
            print("Agent: Bye! Take care...")
            break

        try:
            response = brain.handle_text(user_input)
            print("Agent:", response)
        except Exception as e:
            print("Agent: Sorry, I ran into an internal error.")
            print("DEBUG:", e)
            
if __name__ == "__main__":
    run_cli()