from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class TextBatch:
    input_ids: np.ndarray        # (B, T)
    attention_mask: np.ndarray   # (B, T)


def load_splits(
    dataset_name: str,
    dataset_config: str,
    split_cal: str,
    split_test: str,
    cache_dir: str,
) -> tuple[Dataset, Dataset]:
    cal = load_dataset(dataset_name, dataset_config, split=split_cal, cache_dir=cache_dir)
    test = load_dataset(dataset_name, dataset_config, split=split_test, cache_dir=cache_dir)
    return cal, test


def iter_token_blocks(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_field: str,
    max_samples: int,
    max_seq_len: int,
    seed: int,
    batch_size: int = 1,
) -> Iterator[TextBatch]:
    assert batch_size >= 1
    n = min(len(ds), max_samples)

    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    idxs = idxs[:n]

    buf: List[int] = []
    cur_ids: List[List[int]] = []
    cur_mask: List[List[int]] = []

    for i in idxs:
        txt = ds[i][text_field]
        if not isinstance(txt, str) or not txt.strip():
            continue
        ids = tokenizer.encode(txt, add_special_tokens=False)
        buf.extend(ids)

        while len(buf) >= max_seq_len:
            block = buf[:max_seq_len]
            buf = buf[max_seq_len:]

            cur_ids.append(block)
            cur_mask.append([1] * len(block))

            if len(cur_ids) == batch_size:
                yield TextBatch(
                    input_ids=np.asarray(cur_ids, dtype=np.int64),
                    attention_mask=np.asarray(cur_mask, dtype=np.int64),
                )
                cur_ids, cur_mask = [], []