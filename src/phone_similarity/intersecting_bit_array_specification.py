from typing import List
from phone_similarity.base_bit_array_specification import BaseBitArraySpecification

class IntersectingBitArraySpecification(BaseBitArraySpecification):
    """
    A specification that combines multiple BitArraySpecifications.
    """

    def __init__(self, specifications: List[BaseBitArraySpecification], max_syllables_per_text: int = 6):
        """
        Initializes the IntersectingBitArraySpecification.

        Args:
            specifications: A list of BitArraySpecification objects to combine.
            max_syllables_per_text: The maximum number of syllables per text chunk.
        """
        vowels = set()
        consonants = set()
        features_per_phoneme = {}

        for spec in specifications:
            vowels.update(spec._vowels)
            consonants.update(spec._consonants)
            features_per_phoneme.update(spec._phoneme_features)

        super().__init__(
            vowels=vowels,
            consonants=consonants,
            features_per_phoneme=features_per_phoneme,
            max_syllables_per_text=max_syllables_per_text,
        )
