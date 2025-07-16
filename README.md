# LocalKB

An MCP (Model Context Protocol) server for local knowledge base operations. LocalKB allows agents to create, manage, and search personal knowledge bases using semantic search and document chunking.

## Available Tools

### Knowledge Base Management

**`list_kbs`** - List all available knowledge bases
- No arguments

**`create_kb`** - Create a new knowledge base
- `kb_name` (string): Name for the knowledge base
- `description` (string): Description of the knowledge base

### Content Management

**`add_text_to_kb`** - Add text content directly to a knowledge base
- `kb_name` (string): Name of the knowledge base to add content to
- `source` (string): Unique identifier for the text content (e.g., filename, document title)
- `text` (string): The text content to add to the knowledge base
- `metadata` (object, optional): Optional metadata dict to attach to all content

**`search_kb`** - Search for relevant content using semantic search
- `kb_name` (string): Name of the knowledge base to search
- `query` (string): Search query string
- `filters` (object, optional): Optional ChromaDB metadata filters to narrow search


## Add to Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "LocalKB": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/RoryMB/LocalKB@main", "localkb"]
    }
  }
}
```

## Manual Use

### Command Line

Run this command to open an interactive menu with options to use the various tools:

```bash
uvx --from git+https://github.com/RoryMB/LocalKB@main localkb --cli
```

### Python Import

```python
from localkb.tools import (
    create_kb,
    add_text_to_kb,
    search_kb,
    list_kbs
)

def main():
    # Create a knowledge base
    result = create_kb("my-docs", "Project documentation")
    print(result)
    
    # Add content
    result = add_text_to_kb(
        "my-docs",
        "installation-guide",
        "To install this project, run: pip install -e ."
    )
    print(result)
    
    # Search for content
    results = search_kb("my-docs", "how to install")
    for result in results:
        print(f"Content: {result['content']}")
        print(f"Source: {result['metadata']['source']}")

# Run the function
main()
```
