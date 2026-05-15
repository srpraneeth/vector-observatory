from __future__ import annotations

from pathlib import Path

from .store import DuckDBStore

_DEFAULT_EXPERIMENTS_DIR = Path.cwd() / "data" / "experiments"


class Experiment:
    """A named workspace backed by a single DuckDB file.

    Each experiment is self-contained: embeddings, reductions, clusters,
    runs, and drift comparisons all live in one file.
    """

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.store = DuckDBStore(path)

    @classmethod
    def create(cls, name: str, experiments_dir: Path | None = None) -> Experiment:
        """Create a new experiment. Raises if it already exists."""
        dir_ = experiments_dir or _DEFAULT_EXPERIMENTS_DIR
        dir_.mkdir(parents=True, exist_ok=True)
        path = dir_ / f"{name}.duckdb"
        if path.exists():
            raise FileExistsError(f"Experiment {name!r} already exists at {path}.")
        return cls(name=name, path=path)

    @classmethod
    def load(cls, name: str, experiments_dir: Path | None = None) -> Experiment:
        """Load an existing experiment. Raises if not found."""
        dir_ = experiments_dir or _DEFAULT_EXPERIMENTS_DIR
        path = dir_ / f"{name}.duckdb"
        if not path.exists():
            raise FileNotFoundError(f"Experiment {name!r} not found at {path}.")
        return cls(name=name, path=path)

    @classmethod
    def load_or_create(cls, name: str, experiments_dir: Path | None = None) -> Experiment:
        """Load experiment if it exists, create it otherwise."""
        dir_ = experiments_dir or _DEFAULT_EXPERIMENTS_DIR
        path = dir_ / f"{name}.duckdb"
        dir_.mkdir(parents=True, exist_ok=True)
        return cls(name=name, path=path)

    @staticmethod
    def list_all(experiments_dir: Path | None = None) -> list[str]:
        """Return names of all experiments in the experiments directory."""
        dir_ = experiments_dir or _DEFAULT_EXPERIMENTS_DIR
        if not dir_.exists():
            return []
        return sorted(p.stem for p in dir_.glob("*.duckdb"))

    @staticmethod
    def delete(name: str, experiments_dir: Path | None = None) -> None:
        """Permanently delete an experiment file."""
        dir_ = experiments_dir or _DEFAULT_EXPERIMENTS_DIR
        path = dir_ / f"{name}.duckdb"
        if path.exists():
            path.unlink()

    def __repr__(self) -> str:
        return f"Experiment(name={self.name!r}, path={self.path})"
