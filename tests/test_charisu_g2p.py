import torch
from bitarray import bitarray

from phone_similarity.bit_array_generator import BitArrayGenerator
from phone_similarity.clean_phones import clean_phones
from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator
from phone_similarity.language.en_gb import FEATURES, PHONEME_FEATURES, VOWELS_SET
from phone_similarity.phones_product import phones_product

ISO_689_3_LANGUAGE = "eng"
ISO_3166_1_ALPHA_2 = "us"

CONSONANTS_SET = set(filter(lambda phone: phone not in VOWELS_SET, PHONEME_FEATURES))

WORDS_A = ["euthanasia"]
WORDS_B = ["Youth", "in", "asia"]

BITARRAY_GENERATOR = BitArrayGenerator(
    vowels=VOWELS_SET,
    consonants=CONSONANTS_SET,
    features_per_phoneme=PHONEME_FEATURES,
    features=FEATURES,
)


def test_charsiu_g2p_many_hypotheses():
    language = f"{ISO_689_3_LANGUAGE}-{ISO_3166_1_ALPHA_2}"
    g2p = CharsiuGraphemeToPhonemeGenerator(language)

    _syllables = []
    generation_args = dict(  # pylint: disable=use-dict-literal
        num_beams=5,
        num_return_sequences=5,
        min_p=0.5,
        max_length=50,
        do_sample=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )

    phones_for_words, _ = g2p.generate(words=tuple(WORDS_A), **generation_args)

    assert len(phones_for_words[0]) == 5
    assert BITARRAY_GENERATOR.ipa_to_bitarray(phones_for_words[0][0], 6) == bitarray(
        "0000100000010101000001000100000010000100011101000101100"
        "0000000010000010000100000010001100010001100000000000000"
        "0000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000"
        "00000000"
    )

    if isinstance(phones_for_words[0], str):
        phones_for_words = [clean_phones(p) for p in phones_for_words]

    else:
        phones_for_words = list(
            set([clean_phones(p) for word in phones_for_words for p in word])
        )

    phones_for_words_product = phones_product(
        phones_for_words, tokenizer=BITARRAY_GENERATOR.ipa_tokenizer
    )

    for phones in phones_for_words_product:
        assert isinstance(phones, str)
        _syllables.append(BITARRAY_GENERATOR.ipa_to_syllable(phones))

    if torch.cuda.is_available():
        assert len(_syllables) == 12
    else:
        assert len(_syllables) >= 18


def test_charsiu_g2p_one_hypothesis():
    language = f"{ISO_689_3_LANGUAGE}-{ISO_3166_1_ALPHA_2}"
    g2p = CharsiuGraphemeToPhonemeGenerator(language)

    _syllables = []
    generation_args = dict(  # pylint: disable=use-dict-literal
        num_beams=1,
        max_length=50,
    )

    phones_for_words, _ = g2p.generate(words=tuple(WORDS_A), **generation_args)

    if isinstance(phones_for_words[0], str):
        phones_for_words = [clean_phones(p) for p in phones_for_words]
        for phones in phones_for_words:
            assert isinstance(phones, str)
            _syllables.append(BITARRAY_GENERATOR.ipa_to_syllable(phones))

    assert _syllables[0] == [
        {"nucleus": bitarray("1000100011"), "onset": bitarray("00100000010001")},
        {"nucleus": bitarray("0100000100"), "onset": bitarray("01011000000000")},
        {"nucleus": bitarray("0001110100"), "onset": bitarray("01000000100001")},
        {"nucleus": bitarray("0100000100"), "onset": bitarray("00001000000101")},
    ]


def test_charsiu_g2p_similarity_puns():
    language = f"{ISO_689_3_LANGUAGE}-{ISO_3166_1_ALPHA_2}"
    g2p = CharsiuGraphemeToPhonemeGenerator(language)

    _syllables = []
    generation_args = dict(  # pylint: disable=use-dict-literal
        num_beams=5,
        num_return_sequences=5,
        min_p=0.5,
        max_length=50,
        do_sample=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )

    phones_for_words, _ = g2p.generate(words=tuple(WORDS_A), **generation_args)

    assert len(phones_for_words[0]) == 5
    assert BITARRAY_GENERATOR.ipa_to_bitarray(phones_for_words[0][0], 6) == bitarray(
        "0000100000010101000001000100000010000100011101000101100"
        "0000000010000010000100000010001100010001100000000000000"
        "0000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000"
        "00000000"
    )

    if isinstance(phones_for_words[0], str):
        phones_for_words = [clean_phones(p) for p in phones_for_words]

    else:
        phones_for_words = list(
            set([clean_phones(p) for word in phones_for_words for p in word])
        )

    phones_for_words_product = phones_product(
        phones_for_words, tokenizer=BITARRAY_GENERATOR.ipa_tokenizer
    )

    for phones in phones_for_words_product:
        assert isinstance(phones, str)
        _syllables.append(BITARRAY_GENERATOR.ipa_to_syllable(phones))

    assert len(_syllables) == 12
