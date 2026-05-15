"""Basic analysis example — demonstrates headless library usage without Streamlit."""

from vector_observatory import EmbeddingPipeline, UMAPReducer, HDBSCANClusterer
from vector_observatory.storage.experiment import Experiment

result = (
    EmbeddingPipeline()
    .from_parquet(
        "path/to/your_embeddings.parquet",
        id_col="id",
        embedding_col="vector",
        metadata_cols=["title", "category"],
    )
    .reduce(UMAPReducer(n_components=2, n_neighbors=15, metric="cosine"))
    .cluster(HDBSCANClusterer(min_cluster_size=10))
    .compute_metrics()
    .store(project="my-project")
    .run()
)

ds = result.dataset
print(f"Loaded: {ds.n_samples:,} points, {ds.dim} dims")
print(f"Clusters found: {ds.n_clusters}")
print(f"Noise fraction: {ds.noise_fraction:.1%}")

geo = result.geometry_metrics
print(f"\nEmbedding health:")
print(f"  Anisotropy:      {geo.anisotropy:.3f}  (>0.5 = collapse risk)")
print(f"  Isotropy score:  {geo.isotropy_score:.3f}")
print(f"  Intrinsic dim:   {geo.intrinsic_dim:.1f}  (of {ds.dim} total)")
print(f"\nRun ID: {result.run_id}")
print("Saved to project 'my-project'. Open the app to explore.")
