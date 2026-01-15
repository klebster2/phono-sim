from phone_similarity.language.en_gb import PHONEME_FEATURES, VOWELS_SET
from phone_similarity.model_factory import BitArrayGenerator


def test_make_empty_vector():  # pylint: disable=missing-function-docstring
    consonants_set = set(
        filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES)
    )
    bitarray_generator = BitArrayGenerator(
        vowels=VOWELS_SET, consonants=consonants_set, phoneme_features=PHONEME_FEATURES
    )

    max_syllable_length = len(
        bitarray_generator.make_empty_vector(consonant=True)
    ) * 2 + len(bitarray_generator.make_empty_vector(consonant=False))
    assert max_syllable_length == 38, "Max Syllable Length (in bits)"
