from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.generation.utils import LogitsProcessorList

from phone_similarity.g2p.charisu import LANGUAGE_CODES_CHARSIU, load_dictionary


class CharisuGraphemeToPhonemeGenerator:
    """
    CharsiuGraphemeToPhonemeGenerator

    A class that uses the Charsiu models defined in [1].

    References:
        [1] J. Zhu, C. Zhang, and D. Jurgens, “Byt5 model for massively
            multilingual grapheme-to-phoneme conversion,” 2022.
            [Online]. Available: https://arxiv.org/abs/2204.03067
    """

    def __init__(self, language: str):
        assert (
            language in LANGUAGE_CODES_CHARSIU
        ), "Language not in Charsiu language codes"

        self._language = language
        self._pdict: Dict[str, str] = load_dictionary.load_dictionary_tsv(language)
        self._model = T5ForConditionalGeneration.from_pretrained(
            "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
        )
        self._tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    @property
    def pdict(self) -> Optional[Dict[str, str]]:
        if self._pdict is None:
            raise ValueError("pdict was not defined")
        return self._pdict

    @lru_cache(maxsize=2048)
    def generate(
        self, words: Tuple[str], **generation_kwargs
    ) -> Union[List[str], List[Tuple[str]]]:
        """Generate from a list of words"""
        _words = []
        for word in words:
            _words.append(
                f"<{self._language}>: {word}"
                if not word.startswith(f"<{self._language}>: ")
                else word
            )
        out = self._tokenizer(
            _words, padding=True, add_special_tokens=False, return_tensors="pt"
        )

        num_return_sequences_active = bool("num_return_sequences" in generation_kwargs)

        preds = self._model.generate(  # We do not find beam search helpful. Greedy decoding is enough.
            **out, logits_processor=LogitsProcessorList(), **generation_kwargs
        )
        if not isinstance(preds, dict):
            sequences_preds = preds
            sequences_probs = [1]
        else:
            sequences_preds = preds["sequences"]  # type: ignore
            sequences_probs = [torch.exp(sc) for sc in preds["sequences_scores"]]  # type: ignore

        assert isinstance(sequences_preds, torch.LongTensor)

        if num_return_sequences_active:
            sequences_preds = sequences_preds.reshape(
                len(_words), generation_kwargs["num_return_sequences"], -1
            )
            phones = [
                tuple(
                    self._tokenizer.batch_decode(
                        pred.tolist(), skip_special_tokens=True
                    )
                )
                for pred in sequences_preds
            ]
            return phones, sequences_probs  # type: ignore

        else:
            phones = self._tokenizer.batch_decode(
                sequences_preds.tolist(), skip_special_tokens=True
            )
            return list(phones), sequences_probs[0]  # type: ignore

    @lru_cache(maxsize=2048)
    def get_phonemes_for_word(
        self, word, language: str, generation_kwargs: Optional[Dict[str, str]]
    ):
        """
        Get Phonemes for a single word, first trying the dictionary
        """
        phonemes = None

        if self._pdict is not None and generation_kwargs is None:
            phonemes = self._pdict.get(word, None)

        if phonemes is None:
            tokens = self._tokenizer(
                [f"<{language}>: " + word],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            assert generation_kwargs is not None
            phone_indexes = self._model.generate(*tokens, **generation_kwargs)
            assert isinstance(phone_indexes, torch.LongTensor)

            phonemes = self._tokenizer.batch_decode(
                phone_indexes.tolist(), skip_special_tokens=True
            )[0]
        else:
            phonemes = phonemes.split(",")
        return phonemes
