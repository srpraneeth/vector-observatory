"""Nearest-neighbor inspection example."""

from vector_observatory import EmbeddingDataset
from vector_observatory.retrieval.knn import KNNIndex

ds = EmbeddingDataset.from_parquet(
    "path/to/embeddings.parquet",
    id_col="id",
    embedding_col="vector",
    metadata_cols=["title", "category"],
)

index = KNNIndex(metric="cosine")
index.build(ds)

result = index.search_by_id("your_query_id", ds, k=10)

print(f"Query: {result.query_id}")
print(f"\nTop {len(result.neighbor_ids)} neighbors:")
for i, (nid, dist) in enumerate(zip(result.neighbor_ids, result.distances, strict=False)):
    row = result.metadata.iloc[i]
    print(f"  {i + 1}. [{dist:.4f}] {nid} — {row.to_dict()}")
