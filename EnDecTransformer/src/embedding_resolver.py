import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingResolver:

    def __init__(self, vocabulary_path, embeddings_path):
        self.__load_vocabulary(vocabulary_path)

        self.__load_embeddings_tensor(embeddings_path)


    def __load_vocabulary(self, vocabulary_path):
        logger.info("Loading vocabulary from: %s", vocabulary_path)

        self.vocabulary = {}

        with open(vocabulary_path, 'r') as f:
            id = 0

            for line in f:
                l = line.rstrip('\n')
                # we build a lookup map Token -> Id  where id is row number
                # this is an approach that with additional optimizations is used in
                # some LLMS. Memory is not constrain because multilingual LLMs with 260K tokens
                # vocab utilize approx 14MB of RAM
                self.vocabulary[l] = id

                id += 1

    def __load_embeddings_tensor(self, embedding_path):
        logger.info("Loading embeddings from: %s", embedding_path)

        self.embeddings = np.load(embedding_path)

        logger.debug("Embeddings loaded = " + str(self.embeddings))


    def get_embeddings(self, tokens) -> np.ndarray:
        logger.debug("Getting embeddings for tokens: %s", tokens)

        token_ids = [self.get_token_id(token) for token in tokens]

        logger.debug("Token ids: %s", token_ids)

        # Perform a vectorized lookup to get the embedding for each token.
        # This is a single, highly-optimized numpy operation that runs in parallel.
        return self.embeddings[token_ids]


    def get_token_id(self, token) -> int:
       return self.vocabulary[token]