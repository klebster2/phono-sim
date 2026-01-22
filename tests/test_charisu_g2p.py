from bitarray import bitarray

from phone_similarity.bit_array_generator import BitArrayGenerator
from phone_similarity.g2p.charisu.generator import CharisuGraphemeToPhonemeGenerator
from phone_similarity.language.en_gb import FEATURES, PHONEME_FEATURES, VOWELS_SET

ISO_689_3_LANGUAGE = "eng"
ISO_3166_1_ALPHA_2 = "us"


def test_charisu_g2p():
    language = f"{ISO_689_3_LANGUAGE}-{ISO_3166_1_ALPHA_2}"
    g2p = CharisuGraphemeToPhonemeGenerator(language)

    words = ["Euthanasia".lower()]

    consonants_set = set(
        filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES)
    )
    bitarray_generator = BitArrayGenerator(
        vowels=VOWELS_SET,
        consonants=consonants_set,
        features_per_phoneme=PHONEME_FEATURES,
        features=FEATURES,
    )

    _syllables = []

    phones_for_words, sequence_probs = g2p.generate(
        words=tuple(words),
        num_beams=5,
        num_return_sequences=5,
        min_p=0.5,
        max_length=50,
        do_sample=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )
    for phones, prob in zip(phones_for_words[0], sequence_probs):
        _syllables.append(
            bitarray_generator.ipa_to_syllable(phones.replace("ˌ", "").replace("ˈ", ""))
        )

    assert _syllables == [
        [
            {"nucleus": bitarray("1000100011"), "onset": bitarray("00100000010001")},
            {"nucleus": bitarray("0100000100"), "onset": bitarray("01011000000000")},
            {"nucleus": bitarray("0001110100"), "onset": bitarray("01000000100001")},
            {"nucleus": bitarray("0100000100"), "onset": bitarray("00001000000101")},
        ],
        [
            {"nucleus": bitarray("1000100011"), "onset": bitarray("00100000010001")},
            {"nucleus": bitarray("0100000100"), "onset": bitarray("01011000000000")},
            {"nucleus": bitarray("0001110100"), "onset": bitarray("01000000100001")},
            {"nucleus": bitarray("0100000100"), "onset": bitarray("00001000000101")},
        ],
        [
            {"nucleus": bitarray("1000100011"), "onset": bitarray("00100000010001")},
            {"nucleus": bitarray("0100000100"), "onset": bitarray("01011000000000")},
            {"nucleus": bitarray("0100000100"), "onset": bitarray("01001000100101")},
        ],
        [
            {"nucleus": bitarray("1000100011"), "onset": bitarray("00100000010001")},
            {"nucleus": bitarray("0100000100"), "onset": bitarray("01011000000000")},
            {"nucleus": bitarray("0001110100"), "onset": bitarray("01000000100001")},
            {"nucleus": bitarray("0101110101"), "onset": bitarray("01001000000001")},
        ],
        [
            {"nucleus": bitarray("1000100011"), "onset": bitarray("00100000010001")},
            {"nucleus": bitarray("0100000100"), "onset": bitarray("01011000000000")},
            {"nucleus": bitarray("0101110101"), "onset": bitarray("01001000100001")},
        ],
    ]
