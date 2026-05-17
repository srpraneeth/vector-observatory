"""Drift comparison example — compare two embedding datasets."""

from vector_observatory import DriftComparison, EmbeddingDataset, UMAPReducer

ds_v1 = EmbeddingDataset.from_parquet(
    "path/to/model_v1_embeddings.parquet",
    id_col="id",
    embedding_col="vector",
    metadata_cols=["title"],
    name="model_v1",
)

ds_v2 = EmbeddingDataset.from_parquet(
    "path/to/model_v2_embeddings.parquet",
    id_col="id",
    embedding_col="vector",
    metadata_cols=["title"],
    name="model_v2",
)

comparison = DriftComparison(
    dataset_a=ds_v1,
    dataset_b=ds_v2,
    reducer=UMAPReducer(n_components=2, n_neighbors=15),
)

result = comparison.run()

print(f"MMD score: {result.mmd_score:.4f}")
print("(0 = identical distributions, higher = more drift)")
print("\nCluster overlap:")
for cid, counts in result.cluster_overlap.items():
    label = "Noise" if cid == -1 else f"Cluster {cid}"
    print(f"  {label}: v1={counts['a_count']}, v2={counts['b_count']}")
