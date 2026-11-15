"""Tests for BehaviorStore: fingerprint memory with similarity search."""

import numpy as np
import pytest

from behavior_helix import BehaviorStore


def test_add_and_len():
    store = BehaviorStore(dim=4)
    store.add("walk", [1.0, 0.0, 0.0, 0.0])
    store.add("run", [0.9, 0.1, 0.0, 0.0])
    assert len(store) == 2
    assert store.has("walk")
    assert store.list_models() == ["run", "walk"]


def test_add_rejects_bad_input():
    store = BehaviorStore(dim=4)
    with pytest.raises(ValueError):
        store.add("", [1, 0, 0, 0])
    with pytest.raises(ValueError):
        store.add("bad-dim", [1.0, 0.0])
    with pytest.raises(ValueError):
        store.add("zero", [0.0, 0.0, 0.0, 0.0])
    store.add("walk", [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        store.add("walk", [0.0, 1.0, 0.0, 0.0])


def test_nearest_ranks_by_similarity():
    store = BehaviorStore(dim=4)
    store.add("walk", [1.0, 0.0, 0.0, 0.0])
    store.add("run", [0.9, 0.435, 0.0, 0.0])
    store.add("idle", [0.0, 0.0, 1.0, 0.0])
    hits = store.nearest([1.0, 0.2, 0.0, 0.0], top_k=2)
    assert hits[0][0] == "walk"
    assert hits[1][0] == "run"
    assert hits[0][1] > hits[1][1]


def test_nearest_empty_store_returns_empty():
    assert BehaviorStore(dim=4).nearest([1, 0, 0, 0]) == []


def test_similar_to_excludes_self():
    store = BehaviorStore(dim=4)
    store.add("walk", [1.0, 0.0, 0.0, 0.0])
    store.add("run", [0.9, 0.435, 0.0, 0.0])
    hits = store.similar_to("walk", top_k=2)
    assert all(name != "walk" for name, _ in hits)
    assert hits[0][0] == "run"


def test_similar_to_unknown_name_raises():
    store = BehaviorStore(dim=4)
    with pytest.raises(ValueError):
        store.similar_to("ghost")


def test_vectors_are_normalized():
    store = BehaviorStore(dim=2)
    store.add("a", [3.0, 4.0])
    name, sim = store.nearest([3.0, 4.0])[0]
    assert name == "a"
    assert sim == pytest.approx(1.0)
