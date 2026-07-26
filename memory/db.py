import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import uuid
import json
from datetime import datetime, timezone

class ExperimentDB:
    """
    RAG-style memory of past candidates, backed by a local ChromaDB
    collection. Every experiment (success, failure, held, or conflict) is
    embedded and stored so generation/prompt_builder.py can retrieve
    similar past successes/failures into the next prompt, and
    evolution/duplicate_checker.py can check a new diff against
    near-duplicates already tried - both by nearest-neighbor search over
    the same collection, not separate stores.
    """
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        # using default sentence-transformers model embedded in chromadb
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        # cosine distance is bounded and near 0 for near-identical text, which
        # is what duplicate_checker.py's small default threshold (0.1) assumes -
        # chromadb's collection default (L2) doesn't range near 0 for this kind
        # of text, so that threshold would never trigger on real near-duplicates.
        self.collection = self.client.get_or_create_collection(
            name="experiments",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def store_experiment(
        self,
        hypothesis: str,
        diff: str,
        rationale: str,
        metrics: Dict[str, float],
        outcome: str,
        failure_reason: Optional[str] = None,
        traceback: Optional[str] = None,
    ) -> str:
        """
        Stores an experiment run in the vector DB. traceback is the raw
        diagnostic text (a real git-apply error, or the sandboxed
        candidate's actual stderr) when one exists - kept as its own
        structured field, distinct from failure_reason, since
        failure_reason is often just a human-readable category label (e.g.
        "Malformed diff rejected by git apply.") with no real diagnostic
        content of its own. generation/prompt_builder.py reads this field
        specifically to feed the real signal into the next prompt.
        """
        record_id = uuid.uuid4().hex

        # Combine text for embedding so it's retrievable by similar hypotheses,
        # failure contexts, OR similar diffs - duplicate_checker.py queries by
        # diff text specifically, so the diff itself must be part of what's
        # embedded, not just stored as unsearched metadata.
        document = f"Hypothesis: {hypothesis}\nRationale: {rationale}\nOutcome: {outcome}\nDiff:\n{diff}"
        if failure_reason:
            document += f"\nFailure Reason: {failure_reason}"
        if traceback:
            document += f"\nTraceback:\n{traceback}"

        metadata = {
            "hypothesis": hypothesis,
            "diff": diff,
            "rationale": rationale,
            "metrics": json.dumps(metrics),
            "outcome": outcome,
            "failure_reason": failure_reason or "",
            # ChromaDB metadata values must be str/int/float/bool, never
            # None - same reason failure_reason above falls back to "".
            "traceback": traceback or "",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[record_id]
        )
        return record_id

    def has_exact_diff(self, diff: str) -> bool:
        """
        Cheap exact-match check via a metadata filter - collection.get()
        does not invoke the embedding function, unlike collection.query().
        Lets duplicate_checker.py short-circuit on an exact repeat (e.g. a
        deterministic client like MockLLMClient re-proposing the same
        diff) without paying for an embedding + nearest-neighbor search
        just to then string-compare the result.
        """
        results = self.collection.get(where={"diff": diff}, limit=1)
        return bool(results and results.get('ids'))

    def retrieve_experiments(self, query: str, k: int = 3, filter_outcome: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves top-k most similar experiments based on query text."""
        where_clause = {}
        if filter_outcome:
            where_clause = {"outcome": filter_outcome}

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_clause if where_clause else None
        )

        retrieved = []
        if results and results['metadatas'] and len(results['metadatas']) > 0:
            distances = results['distances'][0] if results.get('distances') else []
            for i, metadata in enumerate(results['metadatas'][0]):
                meta = dict(metadata)
                meta['metrics'] = json.loads(meta['metrics'])
                if i < len(distances):
                    meta['distance'] = distances[i]
                retrieved.append(meta)

        return retrieved

    def list_all_experiments(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Returns every stored experiment (no similarity search), for
        aggregate KPI computation - retrieve_experiments is nearest-neighbor
        search and isn't suited to "all of them."
        """
        results = self.collection.get(limit=limit)
        retrieved = []
        if results and results.get('metadatas'):
            for metadata in results['metadatas']:
                meta = dict(metadata)
                meta['metrics'] = json.loads(meta['metrics'])
                retrieved.append(meta)
        retrieved.sort(key=lambda m: m.get('created_at', ''))
        return retrieved
