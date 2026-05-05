from typing import Dict, List, Optional
from .embedding_loader import (
    EmbeddingModel,
    SentenceTransformerModel,
    BioBERTModel,
    FinBERTModel,
    SciBERTModel,
    MPNetModel,
    GensimModel,
    GloVeLoader,
)
import warnings


class ModelRegistry:
    _AVAILABLE_MODELS = {
        "sentence-transformers": {
            "all-MiniLM-L6-v2": lambda: SentenceTransformerModel("all-MiniLM-L6-v2"),
            "all-mpnet-base-v2": lambda: MPNetModel("all-mpnet-base-v2"),
            "paraphrase-MiniLM-L6-v2": lambda: SentenceTransformerModel(
                "paraphrase-MiniLM-L6-v2"
            ),
        },
        "domain-specific": {
            "BioBERT": lambda: BioBERTModel(),
            "FinBERT": lambda: FinBERTModel(),
            "SciBERT": lambda: SciBERTModel(),
        },
    }

    _LOADED_MODELS: Dict[str, EmbeddingModel] = {}

    @classmethod
    def list_models(cls) -> Dict[str, List[str]]:
        return {
            cat: list(models.keys()) for cat, models in cls._AVAILABLE_MODELS.items()
        }

    @classmethod
    def get_model(cls, model_key: str) -> EmbeddingModel:
        if model_key in cls._LOADED_MODELS:
            return cls._LOADED_MODELS[model_key]

        for category, models in cls._AVAILABLE_MODELS.items():
            if model_key in models:
                model = models[model_key]()
                cls._LOADED_MODELS[model_key] = model
                return model

        raise ValueError(
            f"Unknown model: {model_key}. Available models: {cls.list_models()}"
        )

    @classmethod
    def load_custom_glove(cls, path: str, name: str = "GloVe-Custom") -> EmbeddingModel:
        if name in cls._LOADED_MODELS:
            return cls._LOADED_MODELS[name]
        model = GloVeLoader(path, name)
        cls._LOADED_MODELS[name] = model
        return model

    @classmethod
    def load_custom_gensim(cls, path: str, name: str) -> EmbeddingModel:
        if name in cls._LOADED_MODELS:
            return cls._LOADED_MODELS[name]
        model = GensimModel(path, name)
        cls._LOADED_MODELS[name] = model
        return model

    @classmethod
    def clear_cache(cls):
        cls._LOADED_MODELS.clear()


class ModelManager:
    def __init__(self):
        self.models: Dict[str, EmbeddingModel] = {}
        self.registry = ModelRegistry

    def add_model(self, model_key: str) -> EmbeddingModel:
        if model_key not in self.models:
            model = self.registry.get_model(model_key)
            self.models[model_key] = model
        return self.models[model_key]

    def remove_model(self, model_key: str):
        if model_key in self.models:
            del self.models[model_key]

    def get_available_models(self) -> Dict[str, List[str]]:
        return self.registry.list_models()

    def get_active_models(self) -> List[str]:
        return list(self.models.keys())

    def check_word_coverage(self, words: List[str]) -> Dict[str, Dict[str, bool]]:
        coverage = {}
        for model_key, model in self.models.items():
            coverage[model_key] = {}
            for word in words:
                coverage[model_key][word] = word.lower().strip() in model
        return coverage
