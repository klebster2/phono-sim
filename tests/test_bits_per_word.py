from bitarray import bitarray

from phone_similarity.language.en_gb import PHONEME_FEATURES, VOWELS_SET
from phone_similarity.model_factory import BitArrayGenerator

MAX_SYLLABLES = 6


def test_bits_per_word():  # pylint: disable=missing-function-docstring
    consonants_set = set(
        filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES)
    )
    bitarray_generator = BitArrayGenerator(
        vowels=VOWELS_SET,
        consonants=consonants_set,
        phoneme_features=PHONEME_FEATURES,
        max_syllables_per_text=MAX_SYLLABLES,
    )

    examples = [
        ("cat", "kæt"),  # 1 syllable
        ("strong", "strɒŋ"),  # 1 syllable
        ("banana", "bənænə"),  # 3 vowels => 3 syllables
        ("computer", "kəmpjuːtə"),  # 3 vowels => 3 syllables
        ("strings", "strɪŋz"),  # 1 syllable
        ("insight", "ɪnsaɪt"),  # 2 vowels => 2 syllables
    ]
    assert bitarray_generator._max_syllables_per_text_chunk == MAX_SYLLABLES, (
        "Maximum bits per Syllable"
    )

    bitarrays = {}
    for w, p in examples:
        bitarrays.update({w: bitarray_generator.ipa_to_syllable(p)})

    assert bitarrays == {
        "cat": [
            {
                "coda": bitarray("00010010100000"),
                "nucleus": bitarray("0000011010"),
                "onset": bitarray("00000010100000"),
            },
        ],
        "strong": [
            {
                "coda": bitarray("10011010110110"),
                "nucleus": bitarray("1001110011"),
                "onset": bitarray("10011000110010"),
            },
        ],
        "banana": [
            {
                "nucleus": bitarray("1100100110"),
                "onset": bitarray("11000000100000"),
            },
            {
                "nucleus": bitarray("1101011010"),
                "onset": bitarray("11010000100100"),
            },
            {
                "nucleus": bitarray("1101100110"),
                "onset": bitarray("11010000100100"),
            },
        ],
        "computer": [
            {
                "nucleus": bitarray("0000101110"),
                "onset": bitarray("00000010100000"),
            },
            {
                "nucleus": bitarray("1101011011"),
                "onset": bitarray("10000110100010"),
            },
            {
                "nucleus": bitarray("1001111110"),
                "onset": bitarray("10010110100010"),
            },
        ],
        "strings": [
            {
                "coda": bitarray("10011000110010"),
                "nucleus": bitarray("1001101011"),
                "onset": bitarray("10011000110010"),
            },
        ],
        "insight": [
            {
                "nucleus": bitarray("0001001000"),
            },
            {
                "coda": bitarray("00010000110000"),
                "nucleus": bitarray("0101010011"),
                "onset": bitarray("00010000010000"),
            },
        ],
    }
