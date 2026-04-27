"""Face database for identity management.

Stores face embeddings with names (dimension auto-detected from model output).
Persistence via JSON metadata + npy embedding files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Simple face embedding database with file persistence."""

    def __init__(self, data_dir: str):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / "faces.json"
        self._faces: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        """Load face database from disk."""
        if not self._meta_path.exists():
            return

        try:
            with open(self._meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load face database: {e}")
            return

        for name in meta.get("names", []):
            npy_path = self._dir / f"{name}.npy"
            if npy_path.exists():
                try:
                    self._faces[name] = np.load(npy_path)
                except Exception as e:
                    logger.warning(f"Failed to load embedding for '{name}': {e}")

        logger.info(f"Loaded {len(self._faces)} faces from database")

    def _save_meta(self) -> None:
        """Save metadata (name list) to disk."""
        with open(self._meta_path, "w") as f:
            json.dump({"names": list(self._faces.keys())}, f)

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Register a face embedding."""
        embedding = embedding.astype(np.float32).flatten()
        # Accept any reasonable embedding dimension (128 or 512 depending on model)
        if embedding.shape[0] not in (128, 512):
            raise ValueError(f"Expected 128 or 512-dim embedding, got {embedding.shape[0]}")

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self._faces[name] = embedding
        np.save(self._dir / f"{name}.npy", embedding)
        self._save_meta()
        logger.info(f"Enrolled face: {name}")

    def delete(self, name: str) -> bool:
        """Remove a face from the database."""
        if name not in self._faces:
            return False

        del self._faces[name]
        npy_path = self._dir / f"{name}.npy"
        if npy_path.exists():
            os.remove(npy_path)
        self._save_meta()
        logger.info(f"Deleted face: {name}")
        return True

    def list_faces(self) -> list[str]:
        """Return all registered face names."""
        return list(self._faces.keys())

    def identify(
        self, embedding: np.ndarray, threshold: float = 0.4
    ) -> tuple[Optional[str], float]:
        """Find the closest matching face.

        Args:
            embedding: Face embedding vector (L2 normalized).
            threshold: Maximum cosine distance for a match.

        Returns:
            (name, distance) or (None, inf) if no match.
        """
        if not self._faces:
            return None, float("inf")

        embedding = embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        best_name = None
        best_dist = float("inf")

        for name, stored in self._faces.items():
            # Cosine distance = 1 - cosine_similarity
            dist = 1.0 - float(np.dot(embedding, stored))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist <= threshold:
            return best_name, best_dist

        return None, best_dist
