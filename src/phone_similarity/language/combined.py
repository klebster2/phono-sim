from phone_similarity.language import de, en_gb, es, fr, nl


LANGUAGES = [de, en_gb, es, fr, nl]


def get_combined_phoneme_features():
    all_features = {}
    for lang in LANGUAGES:
        all_features.update(lang.PHONEME_FEATURES)
    return all_features


def get_combined_vowels():
    all_vowels = set()
    for lang in LANGUAGES:
        all_vowels.update(lang.VOWELS_SET)
    return all_vowels


def get_combined_features():
    consonant_columns = set()
    vowel_columns = set()
    for lang in LANGUAGES:
        consonant_columns.update(lang.FEATURES["consonant"])
        vowel_columns.update(lang.FEATURES["vowel"])
    return {"consonant": consonant_columns, "vowel": vowel_columns}


COMBINED_PHONEME_FEATURES = get_combined_phoneme_features()
COMBINED_VOWELS = get_combined_vowels()
COMBINED_FEATURES = get_combined_features()

COMBINED_CONSONANTS = set(filter(lambda p: p not in COMBINED_VOWELS, COMBINED_PHONEME_FEATURES))
