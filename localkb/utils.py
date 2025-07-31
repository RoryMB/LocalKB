import hashlib
import os
import platform
import re
from pathlib import Path

import torch

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'


def is_apple_silicon() -> bool:
    """
    Detect if running on Apple Silicon (M1/M2/M3 etc.) Mac.
    
    Returns:
        True if running on Apple Silicon, False otherwise
    """
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'


def is_mlx_available() -> bool:
    """
    Check if MLX library is available for import.
    
    Returns:
        True if mlx-lm can be imported, False otherwise
    """
    try:
        import mlx_lm
        return True
    except ImportError:
        return False


class ChromaReranker:
    """
    A class for embedding and reranking documents with ChromaDB integration.
    """

    def __init__(
        self,
        max_length: int = 8192,
        batch_size: int = 4,
    ):
        """
        Initialize the ChromaReranker with specified parameters.

        Args:
            max_length: Maximum sequence length for the reranker
            batch_size: Batch size for model operations
        """
        self.max_length = max_length
        self.batch_size = batch_size

        # Determine model type once at initialization
        # Temporarily disabled until proper MLX implementation is complete
        self.use_mlx = False  # is_apple_silicon() and is_mlx_available()
        
        # Store model references
        self._embedding_model = None
        self._tokenizer = None
        self._reranker_model = None
        self._token_false_id = None
        self._token_true_id = None
        self._prefix_tokens = None
        self._suffix_tokens = None

        # self.default_instruction = "Find documents that answer the user's question." # Original
        self.default_instruction = "Find documents with relevant information about the query"
        prefix = "Judge whether the Document meets the requirements based on the Query and the Instruct provided."

        # Define templates for reranking
        self.prefix = f"<|im_start|>system\n{prefix} Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.embed_prompt_pattern = "<Instruct>: {instruction}\n<Query>:"
        self.rerank_prompt_pattern = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    @property
    def embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            if self.use_mlx:
                try:
                    from mlx_lm import load
                    self._embedding_model, _ = load("kerncore/Qwen3-Embedding-0.6B-MXL-4bit")
                except Exception as e:
                    print(f"Warning: Failed to load MLX embedding model: {e}, falling back to standard model")
                    self.use_mlx = False
            
            if not self.use_mlx:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        return self._embedding_model

    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            if self.use_mlx:
                # Load MLX reranker model to get tokenizer
                self.reranker_model  # This will load the model and tokenizer
            else:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
        return self._tokenizer

    @property
    def reranker_model(self):
        """Lazy load the reranker model."""
        if self._reranker_model is None:
            if self.use_mlx:
                try:
                    from mlx_lm import load
                    self._reranker_model, self._tokenizer = load("kerncore/Qwen3-Reranker-0.6B-MLX-4bit")
                except Exception as e:
                    print(f"Warning: Failed to load MLX reranker model: {e}, falling back to standard model")
                    self.use_mlx = False
            
            if not self.use_mlx:
                from transformers import AutoModelForCausalLM
                self._reranker_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
        return self._reranker_model

    @property
    def token_false_id(self):
        """Get token ID for 'no' classification."""
        if self._token_false_id is None:
            self._token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        return self._token_false_id

    @property
    def token_true_id(self):
        """Get token ID for 'yes' classification."""
        if self._token_true_id is None:
            self._token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        return self._token_true_id

    @property
    def prefix_tokens(self):
        """Get encoded prefix tokens."""
        if self._prefix_tokens is None:
            self._prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        return self._prefix_tokens

    @property
    def suffix_tokens(self):
        """Get encoded suffix tokens."""
        if self._suffix_tokens is None:
            self._suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        return self._suffix_tokens

    def _format_embed_prompt(self, instruction: str) -> str:
        """Format an instruction for embedding."""
        # return f"<Instruct>: {instruction}\n<Query>:"
        return self.embed_prompt_pattern.format(instruction=instruction)

    def _format_rerank_prompt(self, instruction: str, query: str, document: str) -> str:
        """Format input for reranking."""
        # return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        return self.rerank_prompt_pattern.format(instruction=instruction, query=query, document=document)

    def _process_rerank_inputs(self, pairs: list[str]) -> dict:
        """Process input pairs for reranking."""
        if self.use_mlx:
            # MLX tokenization needs to be implemented properly
            raise NotImplementedError("MLX tokenization needs proper implementation")
        
        # Standard transformers tokenizer
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        # Add prefix and suffix tokens
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens

        maxlen = max(len(x) for x in inputs['input_ids'])

        # Pad and convert to tensors
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=maxlen)

        # Move to model device
        for key in inputs:
            inputs[key] = inputs[key].to(self.reranker_model.device)

        return inputs

    def _compute_rerank_scores(self, inputs: dict) -> list[float]:
        """Compute reranking scores for processed inputs."""
        if self.use_mlx:
            # For MLX models, we need to implement logit extraction properly
            # This is a placeholder - MLX models should return proper logits
            # The current implementation is incorrect and needs to be fixed
            # based on the actual MLX model API
            raise NotImplementedError("MLX reranking needs proper logit extraction implementation")
        
        # Standard PyTorch model
        with torch.no_grad():
            batch_scores = self.reranker_model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores

    def _batch_process(self, items: list, batch_size: int, process_func, *args, **kwargs):
        """Generic function to process items in batches."""
        import numpy as np

        all_results = []

        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            batch_results = process_func(batch_items, *args, **kwargs)

            if isinstance(batch_results, np.ndarray):
                all_results.append(batch_results)
            elif isinstance(batch_results, list):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)

        # Concatenate numpy arrays or return list
        if all_results and isinstance(all_results[0], np.ndarray):
            return np.concatenate(all_results, axis=0)
        return all_results

    def _embed_batch(self, texts: list[str], prompt: str = None):
        """Embed a batch of texts."""
        if self.use_mlx:
            # MLX embedding needs proper implementation
            raise NotImplementedError("MLX embedding needs proper implementation")
        
        # Standard SentenceTransformer model
        return self.embedding_model.encode(texts, prompt=prompt)

    def _rerank_batch(self, pairs: list[str]) -> list[float]:
        """Rerank a batch of text pairs."""
        inputs = self._process_rerank_inputs(pairs)
        return self._compute_rerank_scores(inputs)

    def embed_queries(self, queries: list[str], instruction: str = ""):
        """Embed queries with the given instruction."""
        prompt = self._format_embed_prompt(instruction) if instruction else None

        if len(queries) <= self.batch_size:
            return self.embedding_model.encode(queries, prompt=prompt)

        return self._batch_process(queries, self.batch_size, self._embed_batch, prompt)

    def embed_documents(self, documents: list[str]):
        """Embed documents."""
        if len(documents) <= self.batch_size:
            return self.embedding_model.encode(documents)

        return self._batch_process(documents, self.batch_size, self._embed_batch)

    def compute_similarity(self, query_embeddings, document_embeddings) -> torch.Tensor:
        """Compute cosine similarity matrix between query and document embeddings."""
        if self.use_mlx:
            # MLX similarity computation needs proper implementation
            raise NotImplementedError("MLX similarity computation needs proper implementation")
        
        # Standard SentenceTransformer model
        return self.embedding_model.similarity(query_embeddings, document_embeddings)

    def rerank(self, instruction: str, query: str, documents: list[str]) -> list[tuple[str, float]]:
        """
        Rerank documents based on relevance to a query.

        Args:
            instruction: Task (Given a _, find documents that _)
            query: Query to guide reranking
            documents: List of documents to rerank

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        # Format pairs for reranking
        pairs = [self._format_rerank_prompt(instruction, query, doc) for doc in documents]

        # Process and compute scores
        inputs = self._process_rerank_inputs(pairs)
        scores = self._compute_rerank_scores(inputs)

        # Combine documents with scores and sort by relevance
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores

    def _get_collection(self, kb_path: Path, collection_name: str = "documents"):
        """Get or create a ChromaDB collection at the specified path."""
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=str(kb_path), settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return client, collection

    def add_documents(
        self,
        kb_path: Path,
        documents: list[str],
        ids: list[str] = None,
        metadata: list[dict[str, any]] = None,
    ) -> None:
        """Automatically embeds all documents and adds to the vector DB. You should always have `source` in the metadata, to point to the original url or source document."""
        client, collection = self._get_collection(kb_path)

        # Generate IDs if not provided
        if ids is None:
            document_count = collection.count()
            ids = [f"doc_{document_count + i}" for i in range(len(documents))]

        # Validate inputs
        if len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        if metadata is not None and len(metadata) != len(documents):
            raise ValueError("Number of metadata entries must match number of documents")

        # Embed documents
        document_embeddings = self.embed_documents(documents)

        # Add to ChromaDB
        collection.add(
            embeddings=document_embeddings.tolist(),
            documents=documents,
            ids=ids,
            metadatas=metadata,
        )

    def reranked_query(
        self,
        kb_path: Path,
        query: str,
        instruction: str = None,
        query_top_k: int = 10,
        rerank_top_k: int = None,
        rerank_threshold: float = None,
        where: dict[str, any] = None,
        where_document: dict[str, any] = None,
    ) -> list[tuple[str, float, str, dict[str, any]]]:
        """
        Query ChromaDB and rerank the results using the reranking model.

        Args:
            kb_path: Path to the knowledge base
            query: Query string
            instruction: Instruction for both embedding and reranking
            query_top_k: Number of initial results to retrieve from ChromaDB
            rerank_top_k: Number of top results to return after reranking
            rerank_threshold: Minimum score threshold for results
            where: Optional metadata filter
            where_document: Optional document content filter

        Returns:
            List of tuples: (document, rerank_score, document_id, metadata)
            Sorted by reranking score in descending order
        """
        if instruction is None:
            instruction = self.default_instruction

        client, collection = self._get_collection(kb_path)

        # Embed query
        query_embedding = self.embed_queries([query], instruction)[0]

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=query_top_k,
            where=where,
            where_document=where_document,
        )

        documents = results['documents'][0] if results['documents'] else []
        ids = results['ids'][0] if results['ids'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []

        if not documents:
            return []

        # Rerank the documents
        reranked_results = self.rerank(instruction, query, documents)

        # Combine reranked results with metadata and IDs
        final_results = []
        for doc, score in reranked_results:
            # Apply threshold filter if specified
            if rerank_threshold is not None and score < rerank_threshold:
                continue

            # Find the original index of this document
            original_idx = documents.index(doc)
            doc_id = ids[original_idx] if original_idx < len(ids) else None
            metadata = metadatas[original_idx] if original_idx < len(metadatas) else None

            final_results.append((doc, score, doc_id, metadata))

        # Apply top_k limit if specified
        if rerank_top_k is not None:
            final_results = final_results[:rerank_top_k]

        return final_results

def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be used as a filename."""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    # Ensure it's not empty
    if not filename or filename.isspace():
        filename = "untitled"
    return filename.strip()

def process_chunks_with_deduplication(source: str, chunks: list[str], metadata: dict = None) -> tuple[list[str], list[str], list[dict]]:
    """
    Process chunks with deduplication and generate hash-based IDs.

    Args:
        source: Source identifier for the chunks
        chunks: List of text chunks to process
        metadata: Optional base metadata to attach to all chunks

    Returns:
        Tuple of (unique_chunks, unique_ids, unique_metadata)
    """
    unique_chunks = []
    unique_ids = []
    unique_metadata = []
    seen_hashes = set()

    base_metadata = metadata or {}

    for chunk in chunks:
        chunk_hash = hashlib.sha256(f"{source}:{chunk}".encode()).hexdigest()[:16]
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            unique_chunks.append(chunk)
            unique_ids.append(chunk_hash)
            unique_metadata.append({**base_metadata, "source": source})

    return unique_chunks, unique_ids, unique_metadata

def chunk_text(text: str, chunk_size: int = 3000, overlap: bool = True) -> list[str]:
    """
    Split text into chunks using semantic-text-splitter with a pairing strategy.

    Args:
        text: The input text to chunk
        text_type: Type of text ('markdown', 'code', or 'text')
        chunk_size: Maximum size of each chunk in characters

    Returns:
        List of text chunks paired as (0+1), (1+2), (2+3), etc.
    """
    from semantic_text_splitter import TextSplitter

    if overlap:
        chunk_size = chunk_size // 2

    base_chunks = TextSplitter(chunk_size, trim=False).chunks(text)

    if not overlap:
        return base_chunks

    # If only one chunk, return it as-is
    if len(base_chunks) == 1:
        return base_chunks

    # Pair up chunks: (0+1), (1+2), (2+3), etc. for overlap
    paired_chunks = []
    for i in range(len(base_chunks) - 1):
        # Combine current chunk with next chunk
        paired_chunks.append(base_chunks[i] + base_chunks[i + 1])

    return paired_chunks
