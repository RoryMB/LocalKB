# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chromadb",
#     "mcp",
#     "numpy",
#     "semantic-text-splitter",
#     "sentence-transformers",
#     "torch",
#     "transformers",
# ]
# ///

import asyncio
from pathlib import Path

from .tools import (
    _kb_exists,
    _kb_path,
    add_text_to_kb,
    create_kb,
    delete_chunks_by_source,
    list_kb_sources,
    list_kbs,
    search_kb,
)


def main():
    while True:
        print("\nAvailable commands:")
        print("1. Create knowledge base")
        print("2. List knowledge bases")
        print("3. Search knowledge base")
        print("4. List all sources in knowledge base")
        print("5. Add file to knowledge base")
        print("6. Add multiple files from list")
        print("7. Delete chunks by source pattern")
        print("8. Exit")

        choice = input("\nEnter your choice (1-8): ").strip()

        if choice == "1":
            kb_name = input("Enter knowledge base name: ").strip()
            description = input("Enter description: ").strip()

            try:
                result = create_kb(kb_name, description)
            except Exception as e:
                print(f"Error: {e}")
                continue

            print(f"{result['message']}")

        elif choice == "2":
            try:
                kbs = list_kbs()
            except Exception as e:
                print(f"Error: {e}")
                continue

            if not kbs:
                print("No knowledge bases found")
                continue

            print("\nKnowledge bases:")
            for kb in kbs:
                print(f"  {kb['name']}")
                print(f"    Config: {kb['config']}")

        elif choice == "3":
            kb_name = input("Enter knowledge base name: ").strip()
            query = input("Enter search query: ").strip()

            try:
                results = asyncio.run(search_kb(kb_name, query))
            except Exception as e:
                print(f"Error: {e}")
                continue

            if not results:
                print("No results found")
                continue

            print("\nSearch results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']}")
                if result['metadata'] and 'source' in result['metadata']:
                    print(f"   Source: {result['metadata']['source']}")
                content = result['content']
                print(f"   Content: {content[:200]}{'...' if len(content) > 200 else ''}")

        elif choice == "4":
            kb_name = input("Enter knowledge base name: ").strip()

            try:
                kb_dir = _kb_path(kb_name)
                sources = list_kb_sources(kb_dir)
            except Exception as e:
                print(f"Error: {e}")
                continue

            if not sources:
                print("No sources found in knowledge base")
                continue

            print(f"\nSources in knowledge base '{kb_name}':")
            for source_info in sources:
                source = source_info['source']
                count = source_info['chunk_count']
                print(f"  {count:3d} chunks | {source}")
            print(f"\nTotal: {len(sources)} sources, {sum(s['chunk_count'] for s in sources)} chunks")

        elif choice == "5":
            kb_name = input("Enter knowledge base name: ").strip()
            file_path = input("Enter path to file: ").strip()
            source_name = input("Enter source name (optional, defaults to filename): ").strip()

            # Use filename as source if not provided
            if not source_name:
                source_name = Path(file_path).name

            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    print("Error: File is empty")
                    continue

                # Add to knowledge base
                result = asyncio.run(add_text_to_kb(kb_name, source_name, content))
            except Exception as e:
                print(f"Error: {e}")
                continue

            print(result)

        elif choice == "6":
            kb_name = input("Enter knowledge base name: ").strip()
            list_file_path = input("Enter path to file containing file paths: ").strip()

            try:
                # Read file paths from list file
                with open(list_file_path, 'r') as f:
                    file_paths = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error: {e}")
                continue

            if not file_paths:
                print("Error: No file paths found in list file")
                continue

            print(f"\nProcessing {len(file_paths)} files:")

            # Process each file
            processed_count = 0
            for i, file_path in enumerate(file_paths, 1):
                print(f"[{i}/{len(file_paths)}] Processing: {file_path}")

                # Use filename as source
                source_name = Path(file_path).name

                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception:
                    print(f"  Error reading file: {file_path}")
                    continue

                if not content.strip():
                    print(f"  Skipping empty file: {file_path}")
                    continue

                # Add to knowledge base
                result = asyncio.run(add_text_to_kb(kb_name, source_name, content))
                print(f"  {result}")
                processed_count += 1

            print(f"\nCompleted processing {processed_count} files")

        elif choice == "7":
            kb_name = input("Enter knowledge base name: ").strip()
            pattern = input("Enter source pattern to match: ").strip()

            confirm = input(f"This will delete all chunks matching '{pattern}'. Continue? (y/N): ").strip().lower()

            if confirm != "y":
                print("Operation cancelled")
                continue

            try:
                kb_dir = _kb_path(kb_name)
                result = delete_chunks_by_source(kb_dir, pattern)
            except Exception as e:
                print(f"Error: {e}")
                continue

            print(result)

        elif choice == "8":
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("Exiting")
