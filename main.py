
import argparse

from src.ask import ask, print_response, start_chat_cli
from src.config import EMBEDDING_MODEL, LLM_MODEL, MODES


# uv run python -m main -- question "..." --mode dense
def main() -> None:

    parser = argparse.ArgumentParser(description="Ask questions about six Italian building regulations")
    parser.add_argument("--question", help="Ask once and exit. Omit for the interactive chat.")
    parser.add_argument("--mode", default="dense", choices=MODES)
    args = parser.parse_args()

    if args.question:
        answer, sources = ask(args.question, args.mode, EMBEDDING_MODEL, LLM_MODEL, history_aware=False)
        print_response(answer, sources)
    else:
        print("Starting the cli...\n")
        start_chat_cli(args.mode)


if __name__ == "__main__":
    main()
