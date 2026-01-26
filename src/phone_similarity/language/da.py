VOWELS_SET = {
    "a", "aː", "ɑ", "ɑː", "e", "eː", "ɛ", "ɛː", "i", "iː", "o", "oː", "ɔ", "ɔː", "u", "uː", "y", "yː", "ø", "øː", "œ", "œː", "ə",
    "ai", "oi", "ui", "au", "eu", "iu"
}

PHONEME_FEATURES = {
    # Consonants
    "p": {"voiced": False, "labial": True, "manner": "plosive"},
    "b": {"voiced": True, "labial": True, "manner": "plosive"},
    "t": {"voiced": False, "place": "alveolar", "manner": "plosive"},
    "d": {"voiced": True, "place": "alveolar", "manner": "plosive"},
    "k": {"voiced": False, "place": "velar", "manner": "plosive"},
    "g": {"voiced": True, "place": "velar", "manner": "plosive"},
    "f": {"voiced": False, "labial": True, "dental": True, "manner": "fricative"},
    "v": {"voiced": True, "labial": True, "dental": True, "manner": "fricative"},
    "s": {"voiced": False, "place": "alveolar", "manner": "fricative"},
    "z": {"voiced": True, "place": "alveolar", "manner": "fricative"},
    "ʃ": {"voiced": False, "place": "post-alveolar", "manner": "fricative"},
    "ç": {"voiced": False, "place": "palatal", "manner": "fricative"},
    "x": {"voiced": False, "place": "velar", "manner": "fricative"},
    "h": {"voiced": False, "place": "glottal", "manner": "fricative"},
    "m": {"voiced": True, "labial": True, "manner": "nasal"},
    "n": {"voiced": True, "place": "alveolar", "manner": "nasal"},
    "ŋ": {"voiced": True, "place": "velar", "manner": "nasal"},
    "l": {"voiced": True, "place": "alveolar", "manner": "lateral_approximant"},
    "r": {"voiced": True, "place": "uvular", "manner": "trill"},
    "j": {"voiced": True, "place": "palatal", "manner": "approximant"},
    "w": {"voiced": True, "labial": True, "place": "velar", "manner": "approximant"},
    # Vowels
    "iː": {"high": True, "front": True, "round": False, "long": True},
    "yː": {"high": True, "front": True, "round": True, "long": True},
    "uː": {"high": True, "back": True, "round": True, "long": True},
    "eː": {"mid-high": True, "front": True, "round": False, "long": True},
    "øː": {"mid-high": True, "front": True, "round": True, "long": True},
    "oː": {"mid-high": True, "back": True, "round": True, "long": True},
    "ɛː": {"mid-low": True, "front": True, "round": False, "long": True},
    "œː": {"mid-low": True, "front": True, "round": True, "long": True},
    "ɔː": {"mid-low": True, "back": True, "round": True, "long": True},
    "ɑː": {"low": True, "back": True, "round": False, "long": True},
    "i": {"high": True, "front": True, "round": False, "long": False},
    "y": {"high": True, "front": True, "round": True, "long": False},
    "u": {"high": True, "back": True, "round": True, "long": False},
    "e": {"mid-high": True, "front": True, "round": False, "long": False},
    "ø": {"mid-high": True, "front": True, "round": True, "long": False},
    "o": {"mid-high": True, "back": True, "round": True, "long": False},
    "ɛ": {"mid-low": True, "front": True, "round": False, "long": False},
    "œ": {"mid-low": True, "front": True, "round": True, "long": False},
    "ɔ": {"mid-low": True, "back": True, "round": True, "long": False},
    "a": {"low": True, "front": True, "round": False, "long": False},
    "ɑ": {"low": True, "back": True, "round": False, "long": False},
    "ə": {"mid": True, "central": True, "round": False},
    # Diphthongs
    "ai": {"diphthong": True, "start": "a", "end": "i"},
    "oi": {"diphthong": True, "start": "o", "end": "i"},
    "ui": {"diphthong": True, "start": "u", "end": "i"},
    "au": {"diphthong": True, "start": "a", "end": "u"},
    "eu": {"diphthong": True, "start": "e", "end": "u"},
    "iu": {"diphthong": True, "start": "i", "end": "u"},
}

CONSONANT_COLUMNS = {
    "voiced", "labial", "dental", "alveolar", "post-alveolar", "palatal", "velar", "glottal", "uvular",
    "plosive", "fricative", "nasal", "lateral_approximant", "trill", "approximant"
}

VOWEL_COLUMNS = {
    "high", "mid-high", "mid", "mid-low", "low",
    "front", "central", "back",
    "round", "long", "diphthong", "start", "end"
}

FEATURES = {"consonant": CONSONANT_COLUMNS, "vowel": VOWEL_COLUMNS}
