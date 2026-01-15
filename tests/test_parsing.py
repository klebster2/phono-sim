from phone_similarity.language.en_gb import (
    CONSONANT_COLUMNS,
    PHONEME_FEATURES,
    VOWEL_COLUMNS,
    VOWELS_SET,
)
from phone_similarity.model_factory import BitArrayGenerator
from phone_similarity.utils import print_syl_verbose


def test_parsing():
    consonants_set = set(
        filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES)
    )
    bitarray_generator = BitArrayGenerator(
        vowels=VOWELS_SET, consonants=consonants_set, phoneme_features=PHONEME_FEATURES
    )
    sample_ipa = "strɪŋz"
    sylls = bitarray_generator.ipa_to_syllable(sample_ipa)

    print(f"Parsed {sample_ipa} into {len(sylls)} syllable(s).")
    cons_columns = [
        feat.replace("place=", "").replace("manner=", "") for feat in CONSONANT_COLUMNS
    ]

    for idx, syl in enumerate(sylls, start=1):
        print(f"--- Syllable {idx} ---")
        print_syl_verbose(syl, cons_columns, VOWEL_COLUMNS, False)
        import pdb

        pdb.set_trace()
