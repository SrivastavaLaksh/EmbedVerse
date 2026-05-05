import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean


class SimilarityAnalyzer:
    def __init__(self):
        self._similarity_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(euclidean(v1, v2))

    def compute_pairwise_similarity(
        self, embeddings: np.ndarray, method: str = "cosine"
    ) -> np.ndarray:
        n = embeddings.shape[0]
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                if method == "cosine":
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                else:
                    sim = -self.euclidean_distance(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        np.fill_diagonal(similarity_matrix, 1.0)
        return similarity_matrix

    def compute_word_similarities(
        self, model: "EmbeddingModel", target_word: str, comparison_words: List[str]
    ) -> Dict[str, float]:
        target_emb = model.get_embedding(target_word)
        if target_emb is None:
            return {}

        similarities = {}
        for word in comparison_words:
            word_emb = model.get_embedding(word)
            if word_emb is not None:
                similarities[word] = self.cosine_similarity(target_emb, word_emb)
        return similarities

    def compute_cross_model_similarity(
        self, models: Dict[str, "EmbeddingModel"], words: List[str]
    ) -> Dict[str, Dict[str, float]]:
        model_names = list(models.keys())
        cross_similarities = {word: {} for word in words}

        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i >= j:
                    continue
                word_sims = {}
                for model_name, model in models.items():
                    emb1 = model.get_embedding(w1)
                    emb2 = model.get_embedding(w2)
                    if emb1 is not None and emb2 is not None:
                        word_sims[model_name] = self.cosine_similarity(emb1, emb2)
                if word_sims:
                    cross_similarities[w1][w2] = np.mean(list(word_sims.values()))
                    cross_similarities[w2][w1] = cross_similarities[w1][w2]

        return cross_similarities

    def compute_model_agreement(
        self, models: Dict[str, "EmbeddingModel"], words: List[str]
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        pairs = {}
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i >= j:
                    continue
                pair_sims = {}
                for model_name, model in models.items():
                    emb1 = model.get_embedding(w1)
                    emb2 = model.get_embedding(w2)
                    if emb1 is not None and emb2 is not None:
                        pair_sims[model_name] = self.cosine_similarity(emb1, emb2)
                if len(pair_sims) >= 2:
                    pairs[(w1, w2)] = pair_sims
        return pairs

    def rank_similar_words(
        self, model: "EmbeddingModel", target_word: str, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        target_emb = model.get_embedding(target_word)
        if target_emb is None:
            return []

        similarities = []
        for word in model._embeddings.keys():
            if word.lower() != target_word.lower():
                word_emb = model.get_embedding(word)
                if word_emb is not None:
                    sim = self.cosine_similarity(target_emb, word_emb)
                    similarities.append((word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
