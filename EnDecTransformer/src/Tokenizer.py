import logging

logger = logging.getLogger(__name__)

#In this exercise we use a simple tokenizer where 1 token is 1 word in our vocabulary,
#this was a common method for early NLP tokenizers however suffers from vocabulary explosion
class Tokenizer:
    def tokenize(self, sequence) -> list[str]:
        logger.debug("Tokenizing: %s", sequence)
        return sequence.split(" ")