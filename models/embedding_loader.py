import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


class EmbeddingModel:
    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        self._embeddings: Dict[str, np.ndarray] = {}

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        word = word.lower().strip()
        return self._embeddings.get(word)

    def get_embeddings_batch(self, words: List[str]) -> Tuple[np.ndarray, List[int]]:
        embeddings = []
        valid_indices = []
        for i, word in enumerate(words):
            emb = self.get_embedding(word)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)
        return np.array(embeddings), valid_indices

    def __contains__(self, word: str) -> bool:
        return word.lower().strip() in self._embeddings


class GensimModel(EmbeddingModel):
    def __init__(self, model_path: str, name: str):
        from gensim.models import KeyedVectors

        try:
            self._model = KeyedVectors.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
        super().__init__(name, self._model.vector_size)
        self._load_embeddings()

    def _load_embeddings(self):
        for word in self._model.key_to_index:
            self._embeddings[word] = self._model[word]


class Word2VecLoader(EmbeddingModel):
    def __init__(self, model_path: str, name: str = "Word2Vec"):
        super().__init__(name, 0)
        from gensim.models import Word2Vec

        try:
            model = Word2Vec.load(model_path)
            self.dimension = model.wv.vector_size
            self._model = model.wv
        except Exception as e:
            raise ValueError(f"Failed to load Word2Vec from {model_path}: {e}")
        self._load_embeddings()

    def _load_embeddings(self):
        for word in self._model.key_to_index:
            self._embeddings[word] = self._model[word]


class GloVeLoader(EmbeddingModel):
    def __init__(self, embeddings_path: str, name: str = "GloVe"):
        self._load_glove(embeddings_path)
        super().__init__(name, self.dimension)

    def _load_glove(self, path: str):
        self._embeddings = {}
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().split()
            self.dimension = len(first_line) - 1
            self._embeddings[first_line[0]] = np.array(
                [float(x) for x in first_line[1:]]
            )
            for line in f:
                parts = line.strip().split()
                if len(parts) == self.dimension + 1:
                    self._embeddings[parts[0]] = np.array([float(x) for x in parts[1:]])


class FastTextLoader(EmbeddingModel):
    def __init__(self, model_path: str, name: str = "FastText"):
        from gensim.models import KeyedVectors

        try:
            self._model = KeyedVectors.load(model_path)
            super().__init__(name, self._model.vector_size)
            self._load_embeddings()
        except Exception as e:
            raise ValueError(f"Failed to load FastText from {model_path}: {e}")

    def _load_embeddings(self):
        for word in self._model.key_to_index:
            self._embeddings[word] = self._model[word]


class SentenceTransformerModel(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        super().__init__(f"SentenceTransformer-{model_name}", dim)
        self._cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        word = word.lower().strip()
        if word not in self._cache:
            emb = self._model.encode(
                word, convert_to_numpy=True, show_progress_bar=False
            )
            self._cache[word] = emb
        return self._cache[word]


class BioBERTModel(EmbeddingModel):
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        super().__init__(f"BioBERT", 768)
        self._cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        import torch

        word = word.lower().strip()
        if word in self._cache:
            return self._cache[word]
        try:
            inputs = self._tokenizer(
                word, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            self._cache[word] = emb
            return emb
        except:
            return None


class FinBERTModel(EmbeddingModel):
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        super().__init__(f"FinBERT", 768)
        self._cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        import torch

        word = word.lower().strip()
        if word in self._cache:
            return self._cache[word]
        try:
            inputs = self._tokenizer(
                word, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            self._cache[word] = emb
            return emb
        except:
            return None


class SciBERTModel(EmbeddingModel):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        super().__init__(f"SciBERT", 768)
        self._cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        import torch

        word = word.lower().strip()
        if word in self._cache:
            return self._cache[word]
        try:
            inputs = self._tokenizer(
                word, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            self._cache[word] = emb
            return emb
        except:
            return None


class MPNetModel(EmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        super().__init__(f"MPNet", dim)
        self._cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        word = word.lower().strip()
        if word not in self._cache:
            emb = self._model.encode(
                word, convert_to_numpy=True, show_progress_bar=False
            )
            self._cache[word] = emb
        return self._cache[word]
