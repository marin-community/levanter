import numpy as np
import pytest

from levanter.data.style_prefix import STYLE_PREFIX_TOKEN, STYLE_SUFFIX_TOKEN
from levanter.data.text import StylePrefixConfig, StylePrefixProcessor


class MockTokenizer:
    def __init__(self):
        # Reserve ids for unk and pad to mimic HF behaviour.
        self.vocab = {"<pad>": 0, "<unk>": 1, "wiki": 2, "books": 3}
        self.additional_special_tokens: list[str] = []
        self.unk_token = "<unk>"
        self.unk_token_id = 1
        self.name_or_path = "mock-tokenizer"

    def __len__(self) -> int:
        return len(self.vocab)

    def add_special_tokens(self, special_tokens_dict):
        added = 0
        for token in special_tokens_dict.get("additional_special_tokens", []):
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    added += 1
        return added

    def convert_tokens_to_ids(self, token: str):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, text: str, add_special_tokens: bool = False):
        if text == "":
            return {"input_ids": []}
        tokens = text.split()
        return {"input_ids": [self.convert_tokens_to_ids(tok) for tok in tokens]}


class DummyChatProcessor:
    def __call__(self, batch):
        return [
            {
                "input_ids": np.asarray(example["tokens"], dtype=np.int32),
                "assistant_masks": np.asarray(example["mask"], dtype=np.int32),
            }
            for example in batch
        ]

    @property
    def output_exemplar(self):
        return {
            "input_ids": np.zeros((0,), dtype=np.int32),
            "assistant_masks": np.zeros((0,), dtype=np.int32),
        }

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self):
        return {"base": "dummy"}


def _build_processor(tokenizer, **config_kwargs):
    config = StylePrefixConfig(**config_kwargs)
    base = DummyChatProcessor()
    return StylePrefixProcessor(tokenizer, base, config=config)


def test_style_prefix_inserts_tokens_and_masks():
    tokenizer = MockTokenizer()
    processor = _build_processor(tokenizer, style_field="style")

    prefix_id = tokenizer.convert_tokens_to_ids(STYLE_PREFIX_TOKEN)
    suffix_id = tokenizer.convert_tokens_to_ids(STYLE_SUFFIX_TOKEN)
    wiki_id = tokenizer.convert_tokens_to_ids("wiki")

    batch = [
        {"style": "wiki", "tokens": [10, 11, 12], "mask": [0, 1, 1]},
    ]

    outputs = processor(batch)
    assert len(outputs) == 1
    out = outputs[0]

    np.testing.assert_array_equal(
        out["input_ids"],
        np.array([prefix_id, wiki_id, suffix_id, 10, 11, 12], dtype=np.int32),
    )

    np.testing.assert_array_equal(
        out["assistant_masks"],
        np.array([0, 0, 0, 0, 1, 1], dtype=np.int32),
    )

    metadata = processor.metadata
    assert metadata["style_prefix"]["prefix_token"] == STYLE_PREFIX_TOKEN
    assert metadata["style_prefix"]["suffix_token"] == STYLE_SUFFIX_TOKEN


def test_style_prefix_uses_default_style_when_field_missing():
    tokenizer = MockTokenizer()
    processor = _build_processor(tokenizer, default_style="books")

    prefix_id = tokenizer.convert_tokens_to_ids(STYLE_PREFIX_TOKEN)
    suffix_id = tokenizer.convert_tokens_to_ids(STYLE_SUFFIX_TOKEN)
    books_id = tokenizer.convert_tokens_to_ids("books")

    outputs = processor([
        {"tokens": [7, 8], "mask": [1, 1]},
    ])

    out = outputs[0]
    np.testing.assert_array_equal(
        out["input_ids"],
        np.array([prefix_id, books_id, suffix_id, 7, 8], dtype=np.int32),
    )

    np.testing.assert_array_equal(
        out["assistant_masks"],
        np.array([0, 0, 0, 1, 1], dtype=np.int32),
    )


def test_style_prefix_requires_style_when_no_default():
    tokenizer = MockTokenizer()
    processor = _build_processor(tokenizer, style_field="style")

    with pytest.raises(ValueError):
        processor([
            {"tokens": [1, 2], "mask": [1, 1]},
        ])

