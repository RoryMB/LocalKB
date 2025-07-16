import sys

from .cli import main as cli_main
from .server import main as server_main


def main():
    """Main entry point - runs MCP server by default."""
    # Check if we should run CLI mode based on arguments
    if len(sys.argv) > 1:
        cli_main()
    else:
        server_main()


# This allows the module to be executed directly with python -m or uvx
if __name__ == "__main__":
    main()
