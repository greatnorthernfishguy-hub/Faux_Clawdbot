import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import math

# Robustness: Use numpy if available, fallback to pure python if not
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class XetVectorStore:
    def __init__(self, repo_path: str = "xet_data"):
        self.root = Path(repo_path)
        self.vectors_path = self.root / "vectors"
        self.vectors_path.mkdir(parents=True, exist_ok=True)

    def _content_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def store_vector(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> str:
        payload = {
            "id": id,
            "vector": vector,
            "metadata": metadata,
            "timestamp": time.time()
        }
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        key = self._content_hash(data)
        
        # Sharded storage: aa/bb/aabb...
        shard_dir = self.vectors_path / key[:2] / key[2:4]
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        target = shard_dir / key
        target.write_bytes(data)
        return key

    def similarity_search(self, query_vector: List[float], n: int = 5) -> List[Dict]:
        results = []
        if HAS_NUMPY:
            q_vec = np.array(query_vector)
            q_norm = np.linalg.norm(q_vec)
        else:
            q_norm = math.sqrt(sum(x*x for x in query_vector))

        if q_norm == 0: return []

        for f in self.vectors_path.glob("*/*/*"):
            if not f.is_file(): continue
            try:
                blob = json.loads(f.read_text(encoding='utf-8'))
                d_vec = blob['vector']
                
                if HAS_NUMPY:
                    d_np = np.array(d_vec)
                    d_norm = np.linalg.norm(d_np)
                    if d_norm == 0: continue
                    sim = np.dot(q_vec, d_np) / (q_norm * d_norm)
                else:
                    dot_product = sum(a*b for a, b in zip(query_vector, d_vec))
                    d_norm = math.sqrt(sum(x*x for x in d_vec))
                    if d_norm == 0: continue
                    sim = dot_product / (q_norm * d_norm)
                
                results.append({
                    "similarity": float(sim),
                    "metadata": blob.get("metadata", {}),
                    "id": blob.get("id")
                })
            except Exception: continue
            
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:n]
