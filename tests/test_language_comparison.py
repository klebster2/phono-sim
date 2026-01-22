from phone_similarity.bit_array_generator import BitArrayGenerator
from phone_similarity.language.en_gb import FEATURES, PHONEME_FEATURES, VOWELS_SET


def test_splitting():
    consonants_set = set(
        filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES)
    )
    english_bitarray_generator = BitArrayGenerator(
        vowels=VOWELS_SET,
        consonants=consonants_set,
        features_per_phoneme=PHONEME_FEATURES,
        features=FEATURES,
    )

    syllables = english_bitarray_generator.ipa_to_syllable("nɑːsti")
    assert len(syllables) == 2
