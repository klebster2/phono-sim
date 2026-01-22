from phone_similarity.bit_array_generator import BitArrayGenerator
from phone_similarity.language.en_gb import FEATURES, PHONEME_FEATURES, VOWELS_SET

# from phone_similarity.utils import print_syl_verbose


def test_parsing():
    consonants_set = set(
        filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES)
    )
    bitarray_generator = BitArrayGenerator(
        vowels=VOWELS_SET,
        consonants=consonants_set,
        features_per_phoneme=PHONEME_FEATURES,
        features=FEATURES,
    )
    sample_ipa = "strɪŋz"
    sylls = bitarray_generator.ipa_to_syllable(sample_ipa)

    print(f"Parsed {sample_ipa} into {len(sylls)} syllable(s).")
    # cons_columns = [
    #    feat.replace("place=", "").replace("manner=", "") for feat in CONSONANT_COLUMNS
    # ]

    for idx, syl in enumerate(sylls, start=1):
        print(f"--- Syllable {idx} ---")
        # print_syl_verbose(syl, cons_columns, VOWEL_COLUMNS, False)
