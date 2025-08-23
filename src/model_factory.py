# pip install transformers bitarray langcodes numpy
import abc
import csv
import json
import os
from collections import defaultdict
from functools import lrucache
from itertools import combinations_with_replacement
from typing import Any, Dict, List

import langcodes
import numpy as np
import requests
from bitarray import bitarray
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing_extensions import TypedDict


class BitArrayGenerator(abc.ABC):
    """
    Map from character based phonological units into bitarray representation
    """
    def __init__(
        self,
        vowels: List[str],
        consonants: List[str],
        phoneme_features: List[str],
        onsets: List[str],
        codas: List[str],
        max_syllables_per_text: int = 6,
        three_char_token: Set[str] = {}, # expand if needed
        two_char_tokens: Set[str] = {"iː", "uː", "ɜː", "ɔː", "ɑː", "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "eə", "ɪə", "ʊə", "tʃ", "dʒ"},
    ):
        self._vowels = vowels
        self._consonants = consonants
        self._phoneme_features = phoneme_features

        self._consonant_features = {
            k: v
            for k, v in filter(lambda kv: kv[0] in consonants, phoneme_features.items())
        }
        self._vowel_features = {
            k: v
            for k, v in filter(lambda kv: kv[0] in vowels, phoneme_features.items())
        }

        self._onsets = onsets
        self._codas = codas
        self._max_syllables_per_text_chunk = max_syllables_per_text_chunk

        self._two_char_tokens = two_char_token
        self._three_char_tokens = three_char_token

    def generate(self, text) -> bitarray:
        raise NotImplementedError(
            "Bitarray generate method needs to first be implemented"
        )

    def get_phoneme_features(self, phoneme: str) -> Dict[str, bool]:
        if phoneme in self._phoneme_features:
            return self._phoneme_features[phoneme]
        else:
            raise Exception("Unknown Phoneme input '%s'", phoneme)


    @lrucache
    def features_to_bitarray(self, feature_dict: dict, columns: list) -> bitarray:
        """
        Convert a phoneme's feature dictionary into a consistent list of 0 or 1 bit based on FEATURE_COLUMNS.
        """
        bitlist = []

        for idx, col in enumerate(columns, start=0):
            bit = feature_dict.get(col, False)
            bitlist.append(1 if bit else 0)

        return bitarray(bitlist)


    def simple_ipa_string_tokenizer(self, ipa_str: str):
        """
        Basic parser: tries to match 3-char tokens, then 2-char, else single char.
        Returns a list of tokens for the string.

        'strɪŋz' will be mapped into ['s','t','r','ɪ','ŋ','z']
        '
        """
        tokens = []
        i = 0
        while i < len(ipa_str):
            if ipa_str[i : i + 3] in self._three_char_tokens:
                tokens.append(ipa_str[i : i + 3])
                i += 3
                continue
            elif ipa_str[i : i + 2] in self._two_char_tokens:
                tokens.append(ipa_str[i : i + 2])
                i += 2
                continue
            elif ipa_str[i] in self._phoneme_features.keys():
                tokens.append(ipa_str[i])
            else:
                logger.warning("Skipping unknown char %s", ipa_str[i]
            i += 1
        return tokens


    def ipa_to_syllable(self, ipa_str: str):
        """
        Naive approach:
          - For each vowel encountered,
              * gather all preceding consonants (onset)
              * gather the vowel(s) (nucleus)
              * gather subsequent consonants (coda) until the next vowel or end
            Then produce a dictionary with three vectors:
               { "onset": <sum onset consonant bits>,  "nucleus": <vowel bits>, "coda": <sum of coda consonant bits>}

        This lumps consonants together as big consonant-clusters.

        If there are multiple vowels in the word, multiple syllables will be produced.
        """
        tokens = parse_ipa_string(ipa_str)
        i: int = 0
        results: List[dict] = []

        while i < len(tokens):
            # 1) Gather onset (initial consonants until a vowel)
            onset: bitarray = self.make_empty_vector(consonant=True)
            while i < len(tokens) and tokens[i] not in self._vowels:
                onset: bitarray = self._elementwise_addition(
                    vector_a=onset_vec,
                    vector_b=self.features_to_bitarray(
                        feature_dict=self.get_phoneme_features(
                            phoneme=tokens[i]
                        ),
                        columns=self._consonant_columns
                    )
                )
                i += 1

            # 2.
            # Check if current token is final. If it is, merge onset with previous coda.
            # Otherwise merge previous coda with current onset and delete prev coda
            if i >= len(tokens):
                results[-1] = (
                    {
                        "onset": results[-1]["onset"],
                        "nucleus": results[-1]["nucleus"],
                        "coda": self._elementwise_addition(
                            vector_a=results[-1]["coda"],
                            vector_b=onset_vec
                        ),
                    }
                )
            elif len(results) > 0:
                # Compress previous coda into current onset
                onset = self._elementwise_addition(
                    vector_a=results[-1]["coda"],
                    vector_b=onset_vec
                )
                if tokens[i] in self._vowels:
                    del results[-1]["coda"]


            # 3.
            # Now tokens[i] is vowel / nucleus.
            nucleus_vec: bitarray = self.make_empty_vector(consonant=False)
            _vowel_start: int = i
            assert tokens[i] in self._vowels
            while i < len(tokens) and tokens[i] in self._vowels:
                if i != _vowel_start:
                    logger.warning(
                        (
                            "Found multiple vowels in the nucleus (tokens: %s)."
                            " Perhaps better parsing can be applied or the vowel"
                            " set can be expanded into dipthongs."
                        ),
                        ' '.join(tokens[vowel_start:i])
                    )

                nucleus: bitarray = self._elementwise_addition(
                    vector_a=onset_vec,
                    vector_b=self.features_to_bitarray(
                        feature_dict=self.get_phoneme_features(
                            phoneme=tokens[i]
                        ),
                        columns=self._consonant_columns
                    )
                )
                i += 1  # Consume token

            # 4) Gather coda (all consonants after the vowel until next vowel or end)
            coda_vec: bitarray = self.make_empty_vector(consonant=True)
            assert tokens[i] in self._consonants
            while i < len(tokens) and tokens[i] in self._consonants.keys():
                coda_vec: bitarray = self._elementwise_addition(
                    vector_a=onset_vec,
                    vector_b=self.features_to_bitarray(
                        feature_dict=self.get_phoneme_features(
                            phoneme=tokens[i]
                        ),
                        columns=self._consonant_columns
                    )
                )
                i += 1
                # Ensure that no new onset consontant clusters are detected.
                # If they are, then duplicate depending on the position.
                # For the word 'nasty' e.g. 'nasti:' consider that we can split
                # in several places,
                # e.g. ['na', 'sti'], ['nas', 'ti'], ['nast', 'i']
                # This step will fork and generate the onset as a fresh syllable
                   # nasty
                   #  [ [o:'n',n:'a'],[o:'st',n:'i'] ]

                # Better to consider using:
                   # nasties
                   #  [ [o:'n',n:'a'],[o:'st',n:'i','c':'z'] ]

                # This will work for sentences:
                    # Youth in asia
                    # [ [[ o: "j", n: "u:", c:"th"]], [[n: "i", c: "n"]], [[n: "ei", c: "S"], [n:"e"]]
                    ### ->
                    # [[[ o: "j", n: "u:", c:"th"], [n: "i", c: "n"], [n: "ei", c: "S"], [n:"e"]]]
                    # Euthanasia
                    # [ [[ n: "u:", c:"th"], [n: "e", c: "n"], [n: "ei", c: "z"], [n: "e"]]]

            # Combine onset + coda
            # 5) Store the result for this syllable
            results.append(
                {
                    "onset": onset_vec,
                    "nucleus": nucleus_vec,
                    "coda": coda_vec,
                }
            )

        return results


    def make_empty_vector(self, consonant: bool):
        """Return a zeroed bit-vector (integers) for the length of FEATURE_COLUMNS."""
        if consonant:
            return [0] * len(self._consonant_columns)
        return [0] * len(self._vowel_columns)


    def _elementwise_addition(self, vector_a, vector_b):
        """Elementwise addition of two vectors (same length)."""
        return [int(bool(a + b)) for (a, b) in zip(vector_a, vector_b)]


def print_feature_vector(columns):
    print()
    new_columns = [feat for feat in columns]
    for char_idx in range(max([len(n) for n in new_columns])):
        print("               ", end="")
        for word in new_columns:
            if char_idx < len(word):
                print(word[char_idx], end="  ")
            else:
                print(" ", end="  ")
        print()
    print()


def print_syl_verbose(
    syl, cons_columns, vowel_columns, print_zeros: bool = True
) -> None:
    print(" " * 10, "=== ONSET ===")

    for i in range(len(syl["onset"])):
        if print_zeros:
            print("            ", syl["onset"][i], cons_columns[i])
        elif syl["onset"][i] != 0:
            print("            ", syl["onset"][i], cons_columns[i])

    print(" " * 10, "=== NUCLEUS ===")
    for i in range(len(syl["nucleus"])):
        if print_zeros:
            print("            ", syl["nucleus"][i], vowel_columns[i])
        elif syl["nucleus"][i] != 0:
            print("            ", syl["nucleus"][i], vowel_columns[i])

    print(" " * 10, "=== CODA ===")
    for i in range(len(syl["coda"])):
        if print_zeros:
            print("            ", syl["coda"][i], cons_columns[i])
        elif syl["coda"][i] != 0:
            print("            ", syl["coda"][i], cons_columns[i])
