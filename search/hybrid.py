import re
from collections import defaultdict
from typing import Iterable, List, Tuple


def normalize_fts_query(text: str) -> str:
    if not text:
        return ""
    tokens = re.findall(r"[\w]+", text.lower())
    return " OR ".join(tokens)


def rrf_fuse(
    dense_ids: Iterable[str],
    sparse_ids: Iterable[str],
    rrf_k: int = 60,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] += 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] += 1.0 / (rrf_k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
