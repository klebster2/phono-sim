from collections import defaultdict
from itertools import permutations, product

from jiwer import process_words


def phones_product(phones_for_words, tokenizer):
    jiwer_alignments = [
        (
            process_words(
                tokenizer(a),
                tokenizer(b),
            )
        )
        for a, b in permutations(phones_for_words, 2)
    ]

    _phonemes = defaultdict(set)
    for word_substitution_alignment_pair in jiwer_alignments:
        reference_tokenized_ipa = tokenizer(
            "".join([r[0] for r in word_substitution_alignment_pair.references])
        )
        hyp_tokenized_ipa = tokenizer(
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
    return phones_for_words
