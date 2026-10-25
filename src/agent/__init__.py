from .main import run_agent


def main() -> None:
    print("Hello from your friendly AI sales agent!")

    # The agent struggles to find the product name.

    # Test 1: Pure SQL Tool
    run_agent("How many units of the Aura Pro wireless headphones have we sold in total, and what was the revenue?")
    print("-"*40)

    # Test 2: Pure RAG Tool
    run_agent("Does the warranty for the Aura Pro Wireless headphones cover accidental water damage?")
    print("-"*40)

    # Test 3: Compound Query (Calls both SQL and RAG)
    run_agent("Which product generated the most revenue, and what is its warranty period?")
