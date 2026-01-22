from collections import defaultdict
from itertools import permutations, product

from bitarray import bitarray
from jiwer import process_words

from phone_similarity.bit_array_generator import BitArrayGenerator
from phone_similarity.g2p.charisu.generator import CharisuGraphemeToPhonemeGenerator
from phone_similarity.language.en_gb import FEATURES, PHONEME_FEATURES, VOWELS_SET

ISO_689_3_LANGUAGE = "eng"
ISO_3166_1_ALPHA_2 = "us"


def clean_phones(x: str):
    return x.replace("ˌ", "").replace("ˈ", "")


def test_charisu_g2p_many_hypotheses():
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

    phones_for_words, _ = g2p.generate(words=tuple(words), **generation_args)
    assert len(phones_for_words[0]) == 5

    if isinstance(phones_for_words[0], str):
        phones_for_words = [clean_phones(p) for p in phones_for_words]

    else:
        phones_for_words = list(
            set([clean_phones(p) for word in phones_for_words for p in word])
        )

        jiwer_alignments = [
            (
                process_words(
                    bitarray_generator.ipa_tokenizer(a),
                    bitarray_generator.ipa_tokenizer(b),
                )
            )
            for a, b in permutations(phones_for_words, 2)
        ]

        _phonemes = defaultdict(set)
        for word_substitution_alignment_pair in jiwer_alignments:
            reference_tokenized_ipa = bitarray_generator.ipa_tokenizer(
                "".join([r[0] for r in word_substitution_alignment_pair.references])
            )
            hyp_tokenized_ipa = bitarray_generator.ipa_tokenizer(
                "".join([r[0] for r in word_substitution_alignment_pair.hypotheses])
            )

            idx2alignment = {
                idx: alignmentchunk[0]
                for idx, alignmentchunk in enumerate(
                    word_substitution_alignment_pair.alignments, start=0
                )
            }

            previous_label = "NONE"
            start = 0
            idx = 0
            symbols_r = []
            symbols_h = []

            for idx, alignment in idx2alignment.items():

                if alignment.type != previous_label and idx != 0:
                    _phonemes[(start, idx)].add(tuple(symbols_r))
                    _phonemes[(start, idx)].add(tuple(symbols_h))
                    symbols_r = []
                    start = idx

                symbols_r = reference_tokenized_ipa[start : idx + alignment.ref_end_idx]
                symbols_h = hyp_tokenized_ipa[start : idx + alignment.hyp_end_idx]

            _phonemes[(start, idx + 1)].add(tuple(symbols_h))
            _phonemes[(start, idx + 1)].add(tuple(symbols_r))

        phones_for_words = [
            "".join(p)
            for p in product(
                *[
                    [k for j in i[1] for k in j]
                    for i in sorted(_phonemes.items(), key=lambda x: x[0][0])
                ]
            )
        ]

    for phones in phones_for_words:
        assert isinstance(phones, str)
        _syllables.append(bitarray_generator.ipa_to_syllable(phones))

    assert len(_syllables) == 12


def test_charisu_g2p_one_hypothesis():
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
    generation_args = dict(  # pylint: disable=use-dict-literal
        num_beams=1,
        max_length=50,
    )

    phones_for_words, _ = g2p.generate(words=tuple(words), **generation_args)

    if isinstance(phones_for_words[0], str):
        phones_for_words = [clean_phones(p) for p in phones_for_words]
        for phones in phones_for_words:
            assert isinstance(phones, str)
            _syllables.append(bitarray_generator.ipa_to_syllable(phones))

    assert _syllables[0] == [
        {"nucleus": bitarray("1000100011"), "onset": bitarray("00100000010001")},
        {"nucleus": bitarray("0100000100"), "onset": bitarray("01011000000000")},
        {"nucleus": bitarray("0001110100"), "onset": bitarray("01000000100001")},
        {"nucleus": bitarray("0100000100"), "onset": bitarray("00001000000101")},
    ]
