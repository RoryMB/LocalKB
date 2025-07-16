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

"""LocalKB - An MCP server for local knowledge base operations """

import asyncio
import logging
import queue
import threading
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .tools import (
    _check_source_in_kb,
    _kb_exists,
    _kb_path,
    add_text_to_kb as add_text_to_kb_impl,
    create_kb as create_kb_impl,
    list_kbs as list_kbs_impl,
    search_kb as search_kb_impl,
)


class RerankerWorker:
    """
    Manages all interactions with the ChromaReranker in a separate, synchronous thread
    to ensure sequential access and prevent blocking the main async event loop.
    It uses a high-priority queue for searches and a low-priority queue for text.
    """
    def __init__(self):
        self._text_queue = queue.Queue()
        self._search_queue = queue.Queue()

        # Start worker thread
        threading.Thread(target=self._worker, daemon=True).start()

    def submit_text_job(self, kb_name: str, source: str, text: str, metadata: dict = None):
        job_data = (kb_name, source, text, metadata or {})
        self._text_queue.put(job_data)

    def submit_search_job(self, kb_name: str, query: str, filters: dict, max_results: int, relevance_threshold: float, result_future):
        job_data = (kb_name, query, filters, max_results, relevance_threshold, result_future)
        self._search_queue.put(job_data)

    def _worker(self):
        logger.info("RerankerWorker thread started")
        while True:
            try:
                search_job = None
                text_job = None

                # Check search queue first (high priority)
                try:
                    search_job = self._search_queue.get(block=False)
                    kb_name, query, filters, max_results, relevance_threshold, result_future = search_job
                    self._process_search(kb_name, query, filters, max_results, relevance_threshold, result_future)
                    continue
                except queue.Empty:
                    pass

                # No search jobs, check text queue (low priority)
                try:
                    text_job = self._text_queue.get(timeout=1.0)
                    kb_name, source, text, metadata = text_job
                    self._process_text(kb_name, source, text, metadata)
                    continue
                except queue.Empty:
                    pass

            except Exception as e:
                logger.error(f"RerankerWorker error: {e}", exc_info=True)

                # If a job with a future fails, we must notify it.
                if search_job is not None:
                    result_future = search_job[-1]
                    if not result_future.done():
                        result_future.get_loop().call_soon_threadsafe(result_future.set_exception, e)

    def _process_text(self, kb_name: str, source: str, text: str, metadata: dict):
        logger.info(f"Starting text processing: {source} for kb '{kb_name}'")
        try:
            result = add_text_to_kb_impl(kb_name, source, text, metadata)
            logger.info(f"Completed text processing: {source} for kb '{kb_name}' - {result}")
        except Exception as e:
            logger.error(f"Text processing failed for {source} in kb '{kb_name}': {e}", exc_info=True)

    def _process_search(self, kb_name: str, query: str, filters: dict, max_results: int, relevance_threshold: float, result_future):
        logger.info(f"Starting search: '{query}' for kb '{kb_name}'")
        try:
            results = search_kb_impl(kb_name, query, filters, max_results, relevance_threshold)
            result_future.get_loop().call_soon_threadsafe(result_future.set_result, results)
            logger.info(f"Completed search: '{query}' for kb '{kb_name}' ({len(results)} results)")
        except Exception as e:
            logger.error(f"Search failed for '{query}' in kb '{kb_name}': {e}", exc_info=True)
            if not result_future.done():
                result_future.get_loop().call_soon_threadsafe(result_future.set_exception, e)


# ===== INITIALIZATION =====

# Initialize FastMCP
mcp = FastMCP("LocalKB")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize global RerankerWorker
reranker_worker = RerankerWorker()
logger.info("LocalKB initialized successfully")


# ===== TOOLS =====

@mcp.tool()
async def list_kbs():
    """List information on all available knowledge bases."""
    return list_kbs_impl()


@mcp.tool()
async def create_kb(kb_name: str, description: str):
    """Create a new knowledge base.

    Args:
        kb_name: Name of the knowledge base
        description: Description of the knowledge base
    """
    return create_kb_impl(kb_name, description)


@mcp.tool()
async def add_text_to_kb(kb_name: str, source: str, text: str, metadata: Optional[dict] = None):
    """Add text content to the knowledge base.

    Args:
        kb_name: Name of the knowledge base
        source: Unique identifier for the text content (e.g., URL, filename, document title) automatically added to metadata as "source"
        text: The text content to add to the knowledge base
        metadata: Optional metadata dict to attach to all content
    """
    # Validate KB exists
    if not _kb_exists(_kb_path(kb_name)):
        raise RuntimeError(f"Knowledge base '{kb_name}' does not exist")

    # Validate text not empty
    if not text.strip():
        raise RuntimeError("No text content provided")

    if _check_source_in_kb(_kb_path(kb_name), source):
        raise RuntimeError(f"Text with name '{source}' already exists in knowledge base. Pick a new source name or validate that the text is new.")

    # Submit job to reranker worker (fire-and-forget)
    reranker_worker.submit_text_job(kb_name, source, text, metadata)

    return "Submitted"


@mcp.tool()
async def search_kb(kb_name: str, query: str, filters: Optional[dict] = None):
    """Search for snippets in the knowledge base that answer the query.

    Args:
        kb_name: Name of the knowledge base to search
        query: Search query string
        filters: Optional ChromaDB metadata filters to narrow search (e.g., {"source": "http://www.example.com"}, {"source": {"$contains": "example.com"}}]})
    """
    # Old arguments. Might put them back later.
    # question: Question with a single focus
    # keywords: Key words or terms of interest
    # max_results: Maximum number of results to return (default: 5)
    # relevance_threshold: Minimum relevance score threshold (0.0-1.0, default: 0.9)

    # # Combine question and keywords into a comprehensive search query
    # query = question
    # # Add keywords to enhance the search, avoiding duplicates
    # keywords = [kw for kw in keywords if kw.lower() not in question.lower()]
    # if keywords:
    #     keywords = " ".join(keywords[:5])
    #     query += f"Relevant keywords include: {keywords}"

    # Validate KB exists
    if not _kb_exists(_kb_path(kb_name)):
        raise RuntimeError(f"Knowledge base '{kb_name}' does not exist")

    # Create a future to receive search results
    loop = asyncio.get_running_loop()
    result_future = loop.create_future()

    # Submit search job to reranker worker
    reranker_worker.submit_search_job(kb_name, query, filters, max_results=5, relevance_threshold=0.9, result_future=result_future)

    # Wait for the results
    try:
        result = await result_future
    except Exception as e:
        raise RuntimeError(f"Search failed for knowledge base '{kb_name}': {e}") from e

    return result


def main():
    mcp.run()


if __name__ == '__main__':
    main()
