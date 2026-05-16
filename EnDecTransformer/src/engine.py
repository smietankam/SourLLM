import logging
from pathlib import Path
from encoder import Encoder
from decoder import Decoder
from tokenizer import Tokenizer
from embedding_resolver import EmbeddingResolver

import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


# TODO: Refactor to use a dataclass for configuration (approach 3)
def generate():
    logger.info("SourLLM is here to save a day")

    # given this is a simple reference implementation the input
    # is strictly limited by the vocabulary size (embedding table) and input string is specific
    # for this implementation
    input_sequence = "I see cats"

    base_dir = Path(__file__).parent

    ## paths to parameters - weights, vocab, model config etc.
    vocabulary_path = base_dir / "params" / "shared" / "vocab.txt"
    embeddings_path = base_dir / "params" / "shared" / "embedding.npy"

    encoder_wq_path = base_dir / "params" / "encoder" / "WQ_enc.npy"
    encoder_wk_path = base_dir / "params" / "encoder" / "WK_enc.npy"
    encoder_wv_path = base_dir / "params" / "encoder" / "WV_enc.npy"


    tokenizer = Tokenizer()
    embedding_resolver = EmbeddingResolver(vocabulary_path, embeddings_path)

    tokens = tokenizer.tokenize(input_sequence)
    embeddings = embedding_resolver.get_embeddings(tokens)

    encoder = Encoder(*__load_encoder_weights(encoder_wq_path, encoder_wk_path, encoder_wv_path))
    decoder = Decoder()

    encoder.encode(embeddings)
    decoder.decode()

#todo this should be cleanup up so that engine doesnt depend on numpy libray - this should be intenral details
#       also, encoder should not load data and focus on computations only
def __load_encoder_weights(wq_path, wk_path, wv_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.load(wq_path), np.load(wk_path), np.load(wv_path)

if __name__ == "__main__":
    generate()
