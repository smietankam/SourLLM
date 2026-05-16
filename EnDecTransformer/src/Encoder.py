import logging

import numpy as np

logger = logging.getLogger(__name__)


class Encoder:
    """
    Wq - tensor of weights trained to compute Q
    Wk - tensor of weights trained to compute K
    Wv - tensor of weights trained to compute V
    """
    def __init__(self, Wq, Wk, Wv):
        self.Wq  = Wq
        self.Wk  = Wk
        self.Wv  = Wv


    def encode(self, embeddings: np.ndarray):
        logger.info("Encoder received embeddings of shape: %s", embeddings.shape)

        self.__self_attention(embeddings)


    """
        Here we implement self attention mechanism bas on Scaled-Dot Product attention, which is the 
        most common self attention function currently used in modern LLMS
        Alternatives are:
            - dot-product attention (Laung attention), predecessor  of SCDP introduced for 
              RNN encoder-decoder with attention model
            - State Space Models
            - Linear Attention
            - Sparse/Sliding Window Attention
        SCDP is so commonly used because it is based on matrix multiplication which is highly scalable on GPUs    
    """
    def __self_attention(self, embeddings: np.ndarray) -> np.ndarray:
        logger.debug("Self-attention embeddings: \n %s", embeddings)

        """
            Here we compute Query, Key, Value tensors 
        """
        Q_tensor = embeddings @ self.Wq
        K_tensor = embeddings @ self.Wk
        V_tensor = embeddings @ self.Wv

        """
            Compute Raw scores - determining cross correlation 
        """
        raw_scores = Q_tensor @ K_tensor.T
        logger.debug("Raw scores: \n %s", raw_scores)

        """
            Scale Raw scores to avoid value explosion 
        """
        scaling_scalar = np.sqrt(embeddings.shape[1])
        logger.debug("Scaling scalar: \n %s", scaling_scalar)

        scaled_scores = raw_scores / scaling_scalar
        logger.debug("Scaled scores: \n %s", scaled_scores)

        """
            Apply attendance mask M which drives which words can attend to which other words
            In the encoder we want to compute scores between any two token in the input sequence 
            thus M consists of 0 only and could be skipped. We include this operation to highlight
            that encoder same as decoder applies mask where some of the token should not be attended - meaning
            softmax should return 0
        """
        mask = np.zeros((scaled_scores.shape[0], scaled_scores.shape[1]))
        logger.debug("Mask: \n %s", mask)

        masked_scores = scaled_scores + mask

        """
            Normalize similarity scores so they sum up to 1  representing how much two words in a sequence are related to each other.
            M is a mask tensor defining which words can be attended  - for encoder these are all words so M = 0
        """
        normalized_scores = self.__softmax(masked_scores)

        """
            Compute final attention scores between the token
        """
        attention_scores = normalized_scores @ V_tensor
        logger.debug("Attention scores: \n %s", attention_scores)

        return  attention_scores


    def __softmax(self, tensor_t: np.ndarray, scaling_factor: float = 1.0):
        logger.debug("Softmax scores for input: \n %s", tensor_t)

        nominator =  tensor_t - np.max(tensor_t, axis=--1, keepdims=True)
        nominator = np.exp(nominator)

        denominator = np.sum(nominator, axis=-1, keepdims=True)

        normalized_tensor_t = nominator / denominator

        logger.debug("Softmax scores for input: \n %s", normalized_tensor_t)

        return  normalized_tensor_t













