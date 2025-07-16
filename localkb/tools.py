import json
from pathlib import Path

from .utils import (
    ChromaReranker,
    chunk_text,
    process_chunks_with_deduplication,
    sanitize_filename,
)

# Global ChromaReranker instance
_reranker = None

# Global knowledge base path
KB_BASE_PATH = Path("~/.mcp/knowledge_base").expanduser()
KB_BASE_PATH.mkdir(parents=True, exist_ok=True)


def _get_reranker():
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = ChromaReranker(batch_size=4)
    return _reranker


def _kb_path(kb_name: str) -> Path:
    """Get the path to a knowledge base directory."""
    return KB_BASE_PATH / kb_name


def _kb_exists(kb_dir: Path) -> bool:
    """Check if a knowledge base exists."""
    return kb_dir.exists() and (kb_dir / "config.json").exists()


def _check_source_in_kb(kb_dir: Path, source: str) -> bool:
    """Check if a source already exists in the knowledge base."""
    if not kb_dir.exists():
        return False

    try:
        reranker = _get_reranker()
        _, collection = reranker._get_collection(kb_dir / "chroma")
        results = collection.get(where={"source": source}, limit=1)
        return len(results.get('ids', [])) > 0
    except Exception:
        return False


def _save_raw_content(kb_dir: Path, source: str, text: str) -> None:
    """Save raw content to the knowledge base raw directory."""
    raw_file = kb_dir / "raw" / f"{sanitize_filename(source)}.md"
    raw_file.parent.mkdir(exist_ok=True)
    with open(raw_file, "w") as f:
        f.write(text)


def list_kb_sources(kb_dir: Path):
    """List all sources in a knowledge base with document counts."""
    # Validate KB exists
    if not _kb_exists(kb_dir):
        raise RuntimeError(f"Knowledge base '{kb_dir.name}' does not exist")

    try:
        reranker = _get_reranker()
        _, collection = reranker._get_collection(kb_dir / "chroma")

        # Get all documents
        all_docs = collection.get()

        if not all_docs.get('ids'):
            return []

        # Count chunks by source
        source_counts = {}
        for metadata in all_docs.get('metadatas', []):
            if metadata and 'source' in metadata:
                source = metadata['source']
                source_counts[source] = source_counts.get(source, 0) + 1

        if not source_counts:
            return []

        # Sort by chunk count (descending) and convert to list
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)

        return [{"source": source, "chunk_count": count} for source, count in sorted_sources]

    except Exception as e:
        raise RuntimeError(f"Failed to list sources: {e}")


def delete_chunks_by_source(kb_dir: Path, source_pattern: str):
    """Delete all chunks matching a source pattern."""
    # Validate KB exists
    if not _kb_exists(kb_dir):
        raise RuntimeError(f"Knowledge base '{kb_dir.name}' does not exist")

    try:
        reranker = _get_reranker()
        _, collection = reranker._get_collection(kb_dir / "chroma")

        # Get all documents to check sources
        all_docs = collection.get()

        if not all_docs.get('ids'):
            return {"success": True, "message": "No documents found in knowledge base", "deleted_count": 0}

        # Find matching IDs and collect their sources
        matching_ids = []
        deleted_sources = set()
        for i, metadata in enumerate(all_docs.get('metadatas', [])):
            if metadata and 'source' in metadata:
                source = metadata['source']
                if source_pattern in source:
                    matching_ids.append(all_docs['ids'][i])
                    deleted_sources.add(source)

        if not matching_ids:
            return {"success": True, "message": f"No chunks found matching source pattern: {source_pattern}", "deleted_count": 0}

        # Delete matching chunks
        collection.delete(ids=matching_ids)

        # Check which sources are completely removed and delete their raw files
        raw_dir = kb_dir / "raw"
        if raw_dir.exists() and deleted_sources:
            # Get remaining documents to check if any deleted sources still exist
            remaining_docs = collection.get()
            remaining_sources = set()
            if remaining_docs.get('metadatas'):
                for metadata in remaining_docs['metadatas']:
                    if metadata and 'source' in metadata:
                        remaining_sources.add(metadata['source'])

            # Only delete raw files for sources that are completely gone
            completely_removed_sources = deleted_sources - remaining_sources

            for source in completely_removed_sources:
                raw_filename = f"{sanitize_filename(source)}.md"
                raw_file = raw_dir / raw_filename
                if raw_file.exists():
                    raw_file.unlink()

        return {
            "success": True,
            "message": f"Deleted {len(matching_ids)} chunks matching pattern: {source_pattern}",
            "deleted_count": len(matching_ids),
            "deleted_sources": list(deleted_sources)
        }

    except Exception as e:
        raise RuntimeError(f"Failed to delete chunks: {e}")


# ===== TOOLS =====

def list_kbs():
    """List all available knowledge bases with detailed information."""
    kbs = []
    for kb_dir in KB_BASE_PATH.iterdir():
        if kb_dir.is_dir():
            try:
                with open(kb_dir / "config.json") as f:
                    config = json.load(f)
            except Exception:
                config = {}
            kbs.append({
                "name": kb_dir.name,
                "config": config,
            })
    return kbs


def create_kb(kb_name: str, description: str):
    """Create a new knowledge base."""
    if not kb_name or not kb_name.replace("_", "").replace("-", "").isalnum():
        raise RuntimeError("Invalid kb_name. Use only alphanumeric characters, hyphens, and underscores.")

    kb_dir = _kb_path(kb_name)
    if kb_dir.exists():
        raise RuntimeError("Knowledge base already exists")

    kb_dir.mkdir(parents=True)
    (kb_dir / "chroma").mkdir()
    (kb_dir / "raw").mkdir()

    # Create config.json
    default_config = {
        "description": description,
    }
    with open(kb_dir / "config.json", "w") as f:
        json.dump(default_config, f, indent=2)

    return f"Successfully created knowledge base '{kb_name}'"


def add_text_to_kb(kb_name: str, source: str, text: str, metadata: dict = None):
    """Add text content directly to the knowledge base."""
    # Validate KB exists
    if not _kb_exists(_kb_path(kb_name)):
        raise RuntimeError(f"Knowledge base '{kb_name}' does not exist")

    if not text.strip():
        raise RuntimeError("No text content provided")

    if _check_source_in_kb(_kb_path(kb_name), source):
        raise RuntimeError(f"Text with name '{source}' already exists in knowledge base")

    # Process text
    chunks = chunk_text(text)

    # Save raw content
    kb_dir = _kb_path(kb_name)
    _save_raw_content(kb_dir, source, text)

    # Deduplicate chunks and generate hash-based IDs
    unique_chunks, unique_ids, unique_metadata = process_chunks_with_deduplication(source, chunks, metadata)

    # Add to ChromaDB
    if unique_chunks:
        reranker = _get_reranker()
        reranker.add_documents(kb_dir / "chroma", unique_chunks, unique_ids, unique_metadata)

    return f"Added {len(unique_chunks)} unique chunks"


def search_kb(kb_name: str, query: str, filters: dict = None, max_results: int = 5, relevance_threshold: float = 0.9):
    """Search for snippets in the knowledge base that might be relevant to the query."""
    # Validate KB exists
    if not _kb_exists(_kb_path(kb_name)):
        raise RuntimeError(f"Knowledge base '{kb_name}' does not exist")

    if not query.strip():
        raise RuntimeError("No search query provided")

    reranker = _get_reranker()
    kb_dir = _kb_path(kb_name)

    try:
        results = reranker.reranked_query(
            kb_path=kb_dir / "chroma",
            query=query,
            rerank_top_k=max_results,
            rerank_threshold=relevance_threshold,
            where=filters,
        )
    except Exception as e:
        raise RuntimeError(f"Search failed for knowledge base '{kb_name}': {e}")

    # Format results
    formatted_results = []
    for content, score, _, metadata in results:
        formatted_results.append({
            # "score": round(score, 2),
            "metadata": metadata,
            "content": content,
        })

    return formatted_results
