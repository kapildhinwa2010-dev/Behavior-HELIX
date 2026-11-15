# Behavior-HELIX - Behavior fingerprints stored and searched by wave similarity
# Part of the Kapildhinwa2010-dev Toolkit.

"""BehaviorStore - fingerprint memory for behavior vectors.

Lineage: DNA + HELIX. Instead of storing weight matrices, this store
keeps behavior fingerprints (fixed-length feature vectors, unit
normalized) and finds them by cosine similarity. nearest() ranks every
stored fingerprint against a probe; similar_to() finds neighbours of a
stored fingerprint.

Cosine similarity on declared features is shallow: two behaviors that
look alike in the chosen features are "the same" here, whatever they are
in reality. See README limitations.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BehaviorStore"]


class BehaviorStore:
    """Store behavior fingerprints; search them by cosine similarity."""

    def __init__(self, dim: int) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.dim = int(dim)
        self._names = []
        self._vectors = []

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.size != self.dim:
            raise ValueError("fingerprint must have exactly {} entries".format(self.dim))
        norm = float(np.linalg.norm(v))
        if norm <= 0.0:
            raise ValueError("fingerprint must be non-zero")
        return v / norm

    def add(self, name: str, vector: np.ndarray) -> None:
        """Store a fingerprint under a unique name."""
        if not name:
            raise ValueError("name must be non-empty")
        if name in self._names:
            raise ValueError("fingerprint {} already stored".format(name))
        self._names.append(name)
        self._vectors.append(self._normalize(vector))

    def nearest(self, vector: np.ndarray, top_k: int = 1):
        """Rank all fingerprints against a probe; returns [(name, similarity)]."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self._names:
            return []
        probe = self._normalize(vector)
        sims = np.stack(self._vectors) @ probe
        order = np.argsort(-sims)[:top_k]
        return [(self._names[int(i)], float(sims[int(i)])) for i in order]

    def similar_to(self, name: str, top_k: int = 2):
        """Nearest fingerprints to a stored one, excluding itself."""
        if name not in self._names:
            raise ValueError("unknown fingerprint {}".format(name))
        idx = self._names.index(name)
        probe = self._vectors[idx]
        sims = np.stack(self._vectors) @ probe
        sims[idx] = -np.inf
        order = np.argsort(-sims)[:top_k]
        return [(self._names[int(i)], float(sims[int(i)]))
                for i in order if np.isfinite(sims[int(i)])]

    def has(self, name: str) -> bool:
        return name in self._names

    def list_models(self) -> list:
        return sorted(self._names)

    def __len__(self) -> int:
        return len(self._names)

class RecallMiss(KeyError):
    """Raised when a requested model is not in the store."""
    pass


class HybridStore:
    """Compressed behavior-fingerprint store using truncated SVD + quantization.

    ``num_coeffs`` controls the rank (number of singular vectors kept).
    ``bits`` controls quantization precision of U, S, Vt components.
    """

    _VALID_KWARGS = frozenset({"num_coeffs", "bits"})

    def __init__(self, num_coeffs: int = 8, bits: int = 12, **kwargs):
        if kwargs:
            bad = set(kwargs.keys()) - self._VALID_KWARGS
            raise ValueError("unexpected keyword arguments: {}".format(sorted(bad)))
        if num_coeffs < 1:
            raise ValueError("num_coeffs must be >= 1")
        if not 1 <= bits <= 32:
            raise ValueError("bits must be in [1, 32]")
        self.num_coeffs = int(num_coeffs)
        self.bits = int(bits)
        self._store: dict = {}

    @staticmethod
    def _quantize(arr, bits):
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        levels = (1 << bits) - 1
        scale = (hi - lo) / max(levels, 1)
        codes = np.round((arr - lo) / max(scale, 1e-15)).astype(np.int32)
        return codes, lo, scale

    @staticmethod
    def _dequantize(codes, lo, scale):
        return codes.astype(np.float64) * scale + lo

    def put(self, key: str, matrix) -> float:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2-dimensional")
        k = min(self.num_coeffs, min(matrix.shape))
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        U_k, s_k, Vt_k = U[:, :k], s[:k], Vt[:k, :]
        Uc, Ulo, Us = self._quantize(U_k, self.bits)
        Sc, Slo, Ss = self._quantize(s_k, self.bits)
        Vc, Vlo, Vs = self._quantize(Vt_k, self.bits)
        self._store[key] = {
            "shape": matrix.shape, "k": k,
            "Uc": Uc, "Ulo": Ulo, "Us": Us,
            "Sc": Sc, "Slo": Slo, "Ss": Ss,
            "Vc": Vc, "Vlo": Vlo, "Vs": Vs,
        }
        original_bytes = matrix.size * 8
        stored_bytes = (Uc.size + Sc.size + Vc.size) * 4 + 80
        return original_bytes / max(stored_bytes, 1)

    def get(self, key: str) -> np.ndarray:
        if key not in self._store:
            raise RecallMiss(key)
        e = self._store[key]
        U = self._dequantize(e["Uc"], e["Ulo"], e["Us"])
        s = self._dequantize(e["Sc"], e["Slo"], e["Ss"])
        Vt = self._dequantize(e["Vc"], e["Vlo"], e["Vs"])
        return (U * s) @ Vt

    def get_noisy(self, key: str, noise: float = 0.0, seed: int = 0) -> np.ndarray:
        if not 0.0 <= noise < 1.0:
            raise ValueError("noise must be in [0, 1)")
        rebuilt = self.get(key)
        rng = np.random.default_rng(seed)
        return rebuilt + noise * rng.standard_normal(rebuilt.shape) * max(float(np.std(rebuilt)), 1e-9)

    def list_models(self) -> list:
        return sorted(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)
