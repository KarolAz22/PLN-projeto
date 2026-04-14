from .paraphrase_multilingual_encoder import ParaphraseMultilingualEncoder
from .paraphrase_multilingual_encoder_v2 import ParaphraseMultilingualEncoderV2
from .biobertpt_encoder_v2 import BioBERTptEncoderV2

# Aliases para compatibilidade com código antigo
SentenceBertBaseEncoder = ParaphraseMultilingualEncoder
SentenceBertBaseEncoderV2 = ParaphraseMultilingualEncoderV2

__all__ = [
    'ParaphraseMultilingualEncoder',
    'ParaphraseMultilingualEncoderV2',
    'BioBERTptEncoderV2',
    'SentenceBertBaseEncoder',
    'SentenceBertBaseEncoderV2',
]