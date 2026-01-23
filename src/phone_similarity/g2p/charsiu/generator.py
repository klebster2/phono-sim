import logging
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.generation.utils import LogitsProcessorList

from phone_similarity.g2p.charsiu import LANGUAGE_CODES_CHARSIU, load_dictionary


class GraphemeToPhonemeResourceType(Enum):
    DICT = 1
    G2P_GENERATOR = 2


class CharsiuGraphemeToPhonemeGenerator:
    """CharsiuGraphemeToPhonemeGenerator.

    A class that uses the Charsiu models defined in [1] to perform
    grapheme-to-phoneme conversion.

    Parameters
    ----------
    language : str
        The language for which to generate phonemes. Must be a valid
        language code from `phone_similarity.g2p.charsiu.LANGUAGE_CODES_CHARSIU`.

    Attributes
    ----------
    pdict : Dict[str, str]
        A dictionary mapping words to their phonemic representations for the
        specified language. This is used as a cache to speed up phoneme
        lookup.

    Methods
    -------
    generate(words: Tuple[str], **generation_kwargs)
        Generate phonemes for a list of words.
    get_phonemes_for_word(word, language, generation_kwargs)
        Get phonemes for a single word, first trying the dictionary.

    References
    ----------
    [1] J. Zhu, C. Zhang, and D. Jurgens,
        “Byt5 model for massively multilingual grapheme-to-phoneme conversion,” 2022.
        [Online]. Available: https://arxiv.org/abs/2204.03067

    """

    DEFAULT_TOKENIZER_MODEL_NAME: str = "google/byt5-small"
    DEFAULT_T5_G2P_MODEL_NAME: str = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"

    def __init__(self, language: str):
        """Initializes the CharsiuGraphemeToPhonemeGenerator.

        Parameters
        ----------
        language : str
            The language for which to generate phonemes. Must be a valid
            language code from `phone_similarity.g2p.charsiu.LANGUAGE_CODES_CHARSIU`.
        """
        assert (
            language in LANGUAGE_CODES_CHARSIU
        ), "Language not in Charsiu language codes"

        self._language = language
        self._pdict: Dict[str, str] = load_dictionary.load_dictionary_tsv(language)

        self._model = T5ForConditionalGeneration.from_pretrained(
            self.DEFAULT_T5_G2P_MODEL_NAME
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.DEFAULT_TOKENIZER_MODEL_NAME
        )
        self._available_g2p_resources: Set[GraphemeToPhonemeResourceType] = set(
            [
                GraphemeToPhonemeResourceType.G2P_GENERATOR,
                GraphemeToPhonemeResourceType.DICT,
            ]
        )

    @property
    def pdict(self) -> Dict[str, str]:
        """A dictionary mapping words to their phonemic representations.

        Returns
        -------
        Dict[str, str]
            The phoneme dictionary mapping words to comma-separated pronunciations

        Raises
        ------
        ValueError
            If the phoneme dictionary was not defined during initialization.
        """
        return self._pdict

    @lru_cache(maxsize=2048)
    def generate(
        self, words: Tuple[str], **generation_kwargs
    ) -> Union[List[str], List[Tuple[str]]]:
        """Generate phonemes for a list of words.

        This method uses the Charsiu model to generate phonemic representations
        for a given list of words.

        Please note that while the original authors state
        "We do not find beam search helpful. Greedy decoding is enough."
        We expose beam search here to expand the number of possibilities

        Parameters
        ----------
        words : Tuple[str]
            A tuple of words to be converted to phonemes.
        **generation_kwargs : dict
            Additional keyword arguments to be passed to the underlying
            model's `generate` method.

        Returns
        -------
        Union[List[str], List[Tuple[str]]]
            A list of phonemic representations for each word. If
            `num_return_sequences` is greater than 1, a list of tuples
            is returned, where each tuple contains multiple phonemic
            representations for a single word.
        """
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

        preds = self._model.generate(
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

        phones = self._tokenizer.batch_decode(
            sequences_preds.tolist(), skip_special_tokens=True
        )
        return phones, sequences_probs  # type: ignore

    def get_phones_from_dict(self, word: str) -> str:
        """get phonemes from a pronunciations dictionary"""
        maybe_phonemes = self._pdict.get(word, None)
        if maybe_phonemes is not None:
            return maybe_phonemes
        error_message = f"{word} was not found in the dictionary"
        logging.error(error_message)
        raise ValueError(error_message)

    @lru_cache(maxsize=2048)
    def get_phones_for_word(
        self,
        word: str,
        language: str,
        limit_resource: Optional[GraphemeToPhonemeResourceType] = None,
        **generation_kwargs: Optional[Dict[str, str]],
    ) -> Union[str, Tuple[str]]:
        """Get phones for a single word.

        This method first attempts to look up the word in the phone
        dictionary. If the word is not found, it uses the Charsiu model available
        to generate the phone representation.

        Parameters
        ----------
        word : str
            The word to be converted to phones.
        language : str
            The language of the word.
        limit_resource : Optional[GraphemeToPhonemeResourceType]
            limit the resource to use only either the G2P 'Generator', or the 'Dict'.
            If left as None, both will be used.
        **generation_kwargs : Optional[Dict[str, str]]
            Additional keyword arguments to be passed to the underlying
            model's `generate` method.

        Returns
        -------
        Union[str, Tuple[str]]
            The phone representation of the word. If found in the
            dictionary, a list of phones is returned. Otherwise, a
            single string of phones is returned.
        """

        if limit_resource == GraphemeToPhonemeResourceType.DICT:
            phoneme_strings: List[str] = self.get_phones_from_dict(word).split()
            return tuple(phoneme_strings)  # type: ignore

        elif limit_resource == GraphemeToPhonemeResourceType.G2P_GENERATOR:
            phonemes = self.generate(tuple([word]), **generation_kwargs)
            return tuple(phonemes) if isinstance(phonemes, list) else phonemes

        tokens = self._tokenizer(
            [f"<{language}>: " + word],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        assert generation_kwargs is not None
        phone_indexes = self._model.generate(*tokens, **generation_kwargs)
        assert isinstance(phone_indexes, torch.LongTensor)

        phonemes: List[str] = self._tokenizer.batch_decode(
            phone_indexes.tolist(), skip_special_tokens=True
        )
        phonemes.append(self.get_phones_from_dict(word).split())
        import pdb

        pdb.set_trace()
        return tuple(phonemes) if isinstance(phonemes, list) else phonemes

        return tuple(phonemes.split(","))  # type: ignore
