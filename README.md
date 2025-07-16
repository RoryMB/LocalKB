# LocalKB

An MCP (Model Context Protocol) server for local knowledge base operations. LocalKB allows agents to create, manage, and search personal knowledge bases using semantic search and document chunking.

## Available Tools

### Knowledge Base Management

**`list_knowledge_bases`** - List all available knowledge bases
- No arguments

**`create_knowledge_base`** - Create a new knowledge base
- `kb_name` (string): Name for the knowledge base
- `description` (string): Description of the knowledge base

**`get_knowledge_base_info`** - Get detailed information about a knowledge base
- `kb_name` (string): Name of the knowledge base

### Content Management

**`text_to_knowledge_base`** - Add text content directly to a knowledge base
- `kb_name` (string): Name of the knowledge base to add content to
- `source` (string): Unique identifier for the text content (e.g., filename, document title)
- `text` (string): The text content to add to the knowledge base
- `metadata` (object, optional): Optional metadata dict to attach to all content

**`search_knowledge_base`** - Search for relevant content using semantic search
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
import asyncio
from localkb import (
    create_knowledge_base,
    add_text_to_knowledge_base,
    search_knowledge_base,
    list_knowledge_bases
)

async def main():
    # Create a knowledge base
    result = create_knowledge_base("my-docs", "Project documentation")
    print(result['message'])
    
    # Add content
    result = await add_text_to_knowledge_base(
        "my-docs",
        "installation-guide",
        "To install this project, run: pip install -e ."
    )
    
    # Search for content
    results = await search_knowledge_base("my-docs", "how to install")
    for result in results:
        print(f"Score: {result['score']}")
        print(f"Content: {result['content']}")
        print(f"Source: {result['metadata']['source']}")

# Run the async function
asyncio.run(main())
```
