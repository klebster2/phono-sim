# pip install transformers bitarray langcodes numpy
import abc
import logging
from functools import lru_cache
from typing import Dict, List, Set

from bitarray import bitarray

from phone_similarity.language.en_gb import CONSONANT_COLUMNS, VOWEL_COLUMNS

_CONSONANT_COLUMNS = tuple(CONSONANT_COLUMNS)
_VOWEL_COLUMNS = tuple(VOWEL_COLUMNS)

del CONSONANT_COLUMNS
del VOWEL_COLUMNS


class BitArrayGenerator(abc.ABC):  # pylint: disable=too-many-instance-attributes
    """
    Map from character based phonological units into bitarray representation
    """

    def __init__(
        self,
        vowels: Set[str],
        consonants: Set[str],
        phoneme_features: Dict[str, Dict[str, bool]],
        max_syllables_per_text: int = 6,
    ):
        self._vowels = vowels
        self._consonants = consonants
        self._phoneme_features = phoneme_features

        self._consonant_features = dict(
            filter(lambda kv: kv[0] in consonants, phoneme_features.items())
        )
        self._vowel_features = dict(
            filter(lambda kv: kv[0] in vowels, phoneme_features.items())
        )
        self._max_syllables_per_text_chunk = max_syllables_per_text
        self._phones_sorted_by_length = tuple(
            sorted(
                phoneme_features.keys(),
                key=lambda x: len(x),  # pylint: disable=unnecessary-lambda
            )
        )
        self._max_phoneme_size = max(len(p) for p in self._phones_sorted_by_length)

    def generate(self, text) -> bitarray:  # pylint: disable=missing-function-docstring
        raise NotImplementedError(
            "Bitarray generate method needs to first be implemented"
        )

    def get_phoneme_features(  # pylint: disable=missing-function-docstring
        self, phoneme: str
    ) -> tuple[tuple[str | bool, ...], ...]:
        if phoneme in self._phoneme_features:
            return tuple(
                tuple([k, v]) for k, v in self._phoneme_features[phoneme].items()
            )
        raise ValueError(f"Unknown Phoneme input '{phoneme}'")

    @lru_cache
    def features_to_bitarray(self, feature_dict: dict, columns: tuple) -> bitarray:
        """
        Convert a phoneme's feature dictionary into a consistent list of 0 or 1 bit based on FEATURE_COLUMNS.
        """
        if isinstance(feature_dict, tuple):
            feature_dict = dict(feature_dict)
        bits: List[int] = []

        for _, col in enumerate(columns, start=0):
            # find feature in column name ('voiced': binary, or values 'place', 'manner')
            if "=" in col:
                # e.g. "place=alveolar", "height=low" ...
                attr, val = col.split("=")
                bit = bool(feature_dict.get(attr) == val)
            else:
                bit = int(feature_dict.get(col, False) or col in feature_dict.values())
            bits.append(bit)

        return bitarray(bits)

    @lru_cache(maxsize=256)
    def search_length_sorted_phonemes(self, ipa_tok_str: str):
        """
        Searches longest phonemes matches first e.g. ai vs. a
        """
        for phone in self._phones_sorted_by_length:
            if ipa_tok_str == phone:
                return phone
        return None

    def simple_ipa_string_tokenizer(self, ipa_str: str):
        """
        Basic Parser: Tries to match longer tokens (descending order)
        Returns a list of tokens for the string.

        'strɪŋz' will be mapped into ['s','t','r','ɪ','ŋ','z']
        """
        tokens = []
        current_index = 1
        start = 0
        while current_index <= len(ipa_str):
            phoneme = self.search_length_sorted_phonemes(ipa_str[start:current_index])
            if phoneme is not None:
                tokens.append(phoneme)
                start = current_index
                current_index += 1
                continue

            if current_index - start == self._max_phoneme_size:
                # hypothetical
                raise ValueError(
                    (
                        f"IPA string contains phonemes outside usual range {ipa_str} "
                        f"(max phoneme length = {self._max_phoneme_size}) "
                        f"Searched: {ipa_str[start:current_index]}"
                    )
                )
            current_index += 1
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
        tokens = self.simple_ipa_string_tokenizer(ipa_str)
        n: int = len(tokens)
        i: int = 0
        results: List[dict] = []

        while i < n:
            # 1.
            # Gather onset (initial consonants until a vowel)
            onset: bitarray = self.make_empty_vector(consonant=True)
            # print(i)
            while i < n and tokens[i] not in self._vowels:
                onset_phoneme = tokens[i]
                onset_features = self.get_phoneme_features(phoneme=onset_phoneme)
                onset_bitarray = self.features_to_bitarray(
                    feature_dict=onset_features,
                    columns=_CONSONANT_COLUMNS,
                )
                onset: bitarray = self._elementwise_addition(
                    vector_a=onset,
                    vector_b=onset_bitarray,
                )
                # print("onset", "idx", i, onset_phoneme, "current array", onset_features)
                i += 1

            # 2.
            # Check if current token is final. If it is, merge onset with previous coda.
            # Otherwise merge previous coda with current onset and delete prev coda
            if i == n:
                results[-1] = {
                    "onset": results[-1]["onset"],
                    "nucleus": results[-1]["nucleus"],
                    "coda": self._elementwise_addition(
                        vector_a=results[-1]["coda"], vector_b=onset
                    ),
                }
                # print("BREAKING WHILE LOOP")
                break

            elif len(results) > 0:
                # Compress previous coda into current onset
                onset = self._elementwise_addition(
                    vector_a=results[-1]["coda"], vector_b=onset
                )
                if tokens[i] in self._vowels:
                    del results[-1]["coda"]

            # 3.
            # Now tokens[i] is vowel / nucleus.
            nucleus: bitarray = self.make_empty_vector(consonant=False)
            _vowel_start: int = i
            # print(i)
            assert tokens[i] in self._vowels
            while i < n and tokens[i] in self._vowels:
                if i != _vowel_start:
                    logging.warning(
                        (
                            "Found multiple vowels in the nucleus (tokens: %s)."
                            " Perhaps better parsing can be applied or the vowel"
                            " set can be expanded into dipthongs."
                        ),
                        " ".join(tokens[_vowel_start:i]),
                    )

                vowel_phoneme = tokens[i]
                vowel_features = self.get_phoneme_features(vowel_phoneme)
                vowel_bitarray = self.features_to_bitarray(
                    feature_dict=vowel_features,
                    columns=_VOWEL_COLUMNS,
                )
                # print("vowel", "idx", i, vowel_phoneme, "current array", vowel_features)
                nucleus: bitarray = self._elementwise_addition(
                    vector_a=onset,
                    vector_b=vowel_bitarray,
                )

                i += 1  # Consume token

            # 4.
            # Gather coda (all consonants after the vowel until next vowel or end)
            # print(i)
            coda: bitarray = self.make_empty_vector(consonant=True)
            while i < n and tokens[i] not in self._vowels:
                coda_phoneme = tokens[i]
                coda_features = self.get_phoneme_features(phoneme=coda_phoneme)
                coda_bitarray = self.features_to_bitarray(
                    feature_dict=coda_features,
                    columns=_CONSONANT_COLUMNS,
                )
                coda: bitarray = self._elementwise_addition(
                    vector_a=onset,
                    vector_b=coda_bitarray,
                )

                # print("coda", "idx", i, coda_phoneme, "current array", coda_features)
                i += 1
                # TODO: Ensure that no new onset consontant clusters are detected.
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
            # 5.
            # Store the result for this syllable
            if sum(coda) > 0 and sum(onset) > 0:
                results.append(
                    {
                        "onset": onset,
                        "nucleus": nucleus,
                        "coda": coda,
                    }
                )
            elif sum(coda) != 0:
                results.append(
                    {
                        "coda": coda,
                        "nucleus": nucleus,
                    }
                )

            elif sum(onset) != 0:
                results.append(
                    {
                        "onset": onset,
                        "nucleus": nucleus,
                    }
                )
            # print("Incremental result:", results)

        return results

    def make_empty_vector(self, consonant: bool):
        """Return a zeroed bit-vector (integers) for the length of FEATURE_COLUMNS."""
        if consonant:
            return bitarray([0] * len(_CONSONANT_COLUMNS))
        return bitarray([0] * len(_VOWEL_COLUMNS))

    def _elementwise_addition(self, vector_a, vector_b) -> bitarray:
        """Elementwise addition of two vectors (same length)."""
        return bitarray([int(bool(a + b)) for (a, b) in zip(vector_a, vector_b)])
