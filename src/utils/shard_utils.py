#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The Google Research Authors and The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for constructing PyTrees of PartitionSpecs."""

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py

import re
import jax.numpy as jnp
import jax.random as random
from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict


# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace

def set_partitions(in_dict, rules):
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    print('unmatches keys:', {k for k, v in result.items() if v is _unmatched})
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))

# Source: https://github.com/google-research/t5x/blob/e5f61889114b2cb5bbfa916eb1ec35e6767427a0/t5x/partitioning.py#L537
# NB: This needs to be top-level for the jax compilation cache.
def _id_fn(x, ix):
  """Identity function for copying parameters to the devices, sharded."""
  # A pure identity such as `lambda x, *: x` can get optimized away, so we
  # include a random.split as a cheap function that cannot be optimized away.
  return x, random.split(jnp.array([ix, ix], dtype=jnp.uint32))
