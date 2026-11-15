# Behavior-HELIX

[![CI](https://github.com/kapildhinwa2010-dev/Behavior-HELIX/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Behavior-HELIX/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Behavior-HELIX/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Behavior-HELIX/actions/workflows/codeql.yml)

Store behavior fingerprints and search them by cosine similarity.

## What it does

BehaviorStore keeps fixed-length feature vectors (behavior fingerprints,
unit normalized) and ranks them against a probe by cosine similarity.
Use it when you have behaviors described as feature vectors and want
"which stored behavior is this most like" queries.

    from behavior_helix import BehaviorStore

    store = BehaviorStore(dim=4)
    store.add("walk", features_a)
    store.add("run", features_b)
    hits = store.nearest(probe_features, top_k=2)

## API

- BehaviorStore(dim) - fingerprint memory for dim-length vectors
- add(name, vector) - store a fingerprint (unique names, non-zero vectors)
- nearest(vector, top_k) - rank all fingerprints against a probe
- similar_to(name, top_k) - neighbours of a stored fingerprint, minus itself
- has / list_models / len

## Honest results and limitations

- Cosine similarity on declared features is shallow: behaviors that look
  alike in the chosen features are "the same" here, whatever they are in
  reality. Feature choice is the whole game and this library does not help
  you choose features.
- Similarity search is a linear scan; cost grows with the number of stored
  fingerprints.
- No learning, no updates: a fingerprint is fixed once added (remove and
  re-add to change it).

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
