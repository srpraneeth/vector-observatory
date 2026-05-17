"""Generate a movies demo dataset and bootstrap a ready-to-use experiment.

Produces 1000 synthetic movie records across 8 genres with 384-dim embeddings
that behave like sentence-transformer output: well-separated genre clusters
with realistic within-cluster variance.

Outputs:
  data/movies_demo.parquet          — raw source file
  data/experiments/movies-demo.duckdb  — pre-analysed experiment (UMAP + HDBSCAN)

Run from the repo root:
  uv run python scripts/generate_demo.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_PER_GENRE = 125  # 8 × 125 = 1000 total
DIM = 384
SEED = 42

REPO_ROOT = Path(__file__).parent.parent
PARQUET_OUT = REPO_ROOT / "data" / "movies_demo.parquet"
DB_OUT = REPO_ROOT / "data" / "projects" / "movies-demo.duckdb"
DEMO_NAME = "movies-demo"

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Thriller", "Animation"]

TITLES: dict[str, list[str]] = {
    "Action": [
        "Edge of Steel",
        "Thunder Protocol",
        "Last Stand at Dawn",
        "Iron Veil",
        "Black Horizon",
        "Storm the Fortress",
        "Fire and Shadow",
        "Rogue Operative",
        "Critical Velocity",
        "Shattered Bastion",
        "Desert Strike",
        "Apex Predator",
        "Zero Meridian",
        "High Altitude",
        "Breach and Clear",
    ],
    "Comedy": [
        "Mostly Harmless",
        "My Brother the Disaster",
        "Accidental Honeymoon",
        "The Wrong Suitcase",
        "Office Romance 101",
        "Three Weddings, No Plan",
        "Dad Jokes: The Movie",
        "Roommates Forever",
        "Totally Normal Tuesday",
        "Fake It Till You Bake It",
        "The Worst Best Man",
        "Speed Dating Catastrophe",
        "Six Months in the Suburbs",
        "Caution: Falling Pies",
        "Double Booked",
    ],
    "Drama": [
        "The Quiet Between Us",
        "Letters Never Sent",
        "Where the River Bends",
        "A House Divided",
        "The Last Summer",
        "What Remains",
        "Borrowed Light",
        "Paper Walls",
        "The Distance Home",
        "One More Season",
        "All We Carry",
        "Shadows of the Valley",
        "The Weight of Choices",
        "Still Waters",
        "The Long Goodbye",
    ],
    "Sci-Fi": [
        "Echoes of the Void",
        "The Fermi Paradox",
        "Synthetic Dreams",
        "Orbital Decay",
        "The Second Signal",
        "Dark Matter Protocol",
        "Exodus Fleet",
        "Parallel Machine",
        "The Turing Threshold",
        "Aftermath Colony",
        "Sub-Light Corridor",
        "The Consensus Algorithm",
        "Last Transmission",
        "Gravity Well",
        "Recursive Horizon",
    ],
    "Horror": [
        "What Lives Below",
        "The Hollow Season",
        "Pale Visitor",
        "Beneath the Floorboards",
        "Night Erosion",
        "The Unopened Room",
        "Something in the Static",
        "Rot Season",
        "The Borrowed Face",
        "Cold Ritual",
        "Sightless",
        "Wicker and Bone",
        "The Last Parish",
        "Threshold Creature",
        "Skin Deep",
    ],
    "Romance": [
        "Late Arrivals",
        "The Second Chance Café",
        "Unexpected Layover",
        "You Again",
        "One Last Paris",
        "The Bookshop Agreement",
        "Wrong Number, Right Person",
        "Slow Burn Summer",
        "Autumn in Between",
        "A Quiet Kind of Love",
        "First Draft",
        "The Apartment Upstairs",
        "Someone Like Sunday",
        "Three Summers Later",
        "Almost Strangers",
    ],
    "Thriller": [
        "The Alibi Collapse",
        "Safe Harbour",
        "Blind Spot",
        "The Witness Protocol",
        "Without a Trace",
        "The Last Asset",
        "Dark Arrangement",
        "Red Channel",
        "The Confessor",
        "Lockdown Protocol",
        "False Flag",
        "The Quiet Operative",
        "Signal Lost",
        "Controlled Burn",
        "Endgame Clause",
    ],
    "Animation": [
        "The Star Keeper",
        "Bumble and the Big Storm",
        "Tiny Kingdom",
        "A Fox Called Blue",
        "The Last Dragon of Ember Isle",
        "Wing and Wind",
        "Pocket Universe",
        "The Clockwork Garden",
        "Moonwhisper",
        "Pebble and the Mountain",
        "Where Fireflies Go",
        "The Map of Everything",
        "Little Ghost",
        "Sunward",
        "The Great Mushroom Mystery",
    ],
}

DIRECTORS = [
    "Sofia Alvarez",
    "James Okafor",
    "Lena Brandt",
    "Raj Nair",
    "Yuki Tanaka",
    "Marcus Webb",
    "Elena Vasquez",
    "Tom Ashford",
    "Nina Kowalski",
    "David Chen",
]

RATINGS_PARAMS: dict[str, tuple[float, float]] = {
    "Action": (7.0, 0.8),
    "Comedy": (6.8, 0.9),
    "Drama": (7.5, 0.7),
    "Sci-Fi": (7.2, 0.8),
    "Horror": (6.5, 1.0),
    "Romance": (6.9, 0.8),
    "Thriller": (7.3, 0.7),
    "Animation": (7.6, 0.6),
}


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def _make_genre_embeddings(
    n: int,
    center: np.ndarray,
    within_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.standard_normal((n, DIM)) * within_var
    n_outliers = max(1, n // 12)
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    noise[outlier_idx] *= 3.5
    raw = center + noise
    norms = np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-8)
    return (raw / norms).astype(np.float32)


def generate_movies_df(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    centers = rng.standard_normal((len(GENRES), DIM))
    Q, _ = np.linalg.qr(centers.T)
    centers = Q[:, : len(GENRES)].T
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    rows = []
    for i, genre in enumerate(GENRES):
        embeddings = _make_genre_embeddings(N_PER_GENRE, centers[i], 0.32, rng)
        title_pool = TITLES[genre]
        mu, sigma = RATINGS_PARAMS[genre]

        for j in range(N_PER_GENRE):
            base_title = title_pool[j % len(title_pool)]
            title = (
                base_title if j < len(title_pool) else f"{base_title} {j // len(title_pool) + 2}"
            )
            rating = float(np.clip(rng.normal(mu, sigma), 1.0, 10.0))
            rows.append(
                {
                    "movie_id": f"{genre.lower().replace('-', '')}_{j:04d}",
                    "title": title,
                    "genre": genre,
                    "year": int(rng.integers(1990, 2024)),
                    "rating": round(rating, 1),
                    "director": rng.choice(DIRECTORS),
                    "embedding": embeddings[j].tolist(),
                }
            )

    rng.shuffle(rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap experiment
# ---------------------------------------------------------------------------


def bootstrap_experiment(df: pd.DataFrame) -> None:
    from vector_observatory.clustering import KMeansClusterer
    from vector_observatory.ingestion.validators import _build_dataset
    from vector_observatory.metrics.cluster import compute_cluster_metrics
    from vector_observatory.metrics.geometry import compute_geometry_metrics
    from vector_observatory.reducers import UMAPReducer
    from vector_observatory.storage.experiment import Experiment

    print("Building dataset…")
    ds = _build_dataset(
        df,
        id_col="movie_id",
        embedding_col="embedding",
        metadata_cols=["title", "genre", "year", "rating", "director"],
        name=DEMO_NAME,
    )

    print("Running UMAP…")
    reducer = UMAPReducer(n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(ds.embeddings)
    ds = ds.with_reduction(coords)

    print("Running K-Means (k=8, one cluster per genre)…")
    clusterer = KMeansClusterer(n_clusters=8)
    labels = clusterer.fit_predict(ds.embeddings)
    ds = ds.with_clusters(labels)

    print("Computing metrics…")
    geo = compute_geometry_metrics(ds.embeddings)
    clust = compute_cluster_metrics(ds.cluster_labels)

    if DB_OUT.exists():
        DB_OUT.unlink()
        print(f"Removed existing {DB_OUT.name}")

    run_id = str(uuid.uuid4())[:8]
    config = {
        "run_id": run_id,
        "reducer": reducer.config,
        "clusterer": clusterer.config,
    }

    project = Experiment.load_or_create(DEMO_NAME)
    project.store.save_run(ds, config, geo, clust)
    print(f"Saved experiment → {DB_OUT}  (run_id: {run_id})")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    print(f"  clusters: {n_clusters}  noise: {noise_pct:.1f}%  anisotropy: {geo.anisotropy:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {N_PER_GENRE * len(GENRES)} movie records…")
    df = generate_movies_df()
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"Saved parquet → {PARQUET_OUT}  ({PARQUET_OUT.stat().st_size // 1024} KB)")
    print(f"Genres: {df['genre'].value_counts().to_dict()}")

    print()
    bootstrap_experiment(df)

    print()
    print("Demo experiment ready. Start the app and you'll see 'movies-demo' in the sidebar.")
    print("You can delete it from the app if you don't need it.")
