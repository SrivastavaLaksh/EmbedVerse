import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EmbeddingVisualizer:
    def __init__(self):
        self._tsne_cache: Dict[str, np.ndarray] = {}
        self._umap_cache: Dict[str, np.ndarray] = {}

    def reduce_dimensions_tsne(
        self,
        embeddings: np.ndarray,
        perplexity: float = 5,
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> np.ndarray:
        cache_key = f"tsne_{hash(embeddings.tobytes())}"
        if cache_key in self._tsne_cache:
            return self._tsne_cache[cache_key]

        if embeddings.shape[0] < perplexity:
            perplexity = max(1, embeddings.shape[0] - 1)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            learning_rate="auto",
            init="pca",
        )
        result = tsne.fit_transform(embeddings)
        self._tsne_cache[cache_key] = result
        return result

    def reduce_dimensions_umap(
        self,
        embeddings: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ) -> np.ndarray:
        cache_key = f"umap_{hash(embeddings.tobytes())}"
        if cache_key in self._umap_cache:
            return self._umap_cache[cache_key]

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        result = reducer.fit_transform(embeddings)
        self._umap_cache[cache_key] = result
        return result

    def create_scatter_plot(
        self,
        embeddings_2d: np.ndarray,
        words: List[str],
        labels: Optional[List[str]] = None,
        title: str = "Embedding Visualization",
        colors: Optional[List[str]] = None,
    ) -> go.Figure:
        df = pd.DataFrame(
            {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "word": words}
        )

        if labels:
            df["label"] = labels

        if colors:
            df["color"] = colors

        if labels:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                text="word",
                color="label",
                title=title,
                hover_data={"word": True, "x": False, "y": False},
            )
        else:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                text="word",
                title=title,
                hover_data={"word": True, "x": False, "y": False},
            )

        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(
            height=700,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig

    def create_multi_model_comparison(
        self,
        models_data: Dict[str, Tuple[np.ndarray, List[str]]],
        method: str = "tsne",
        title: str = "Multi-Model Embedding Comparison",
    ) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=list(models_data.keys()),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colors = px.colors.qualitative.Set2

        for idx, (model_name, (embeddings, words)) in enumerate(models_data.items()):
            if method == "tsne":
                reduced = self.reduce_dimensions_tsne(embeddings)
            else:
                reduced = self.reduce_dimensions_umap(embeddings)

            row, col = positions[idx % 4]

            fig.add_trace(
                go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode="markers+text",
                    text=words,
                    textposition="top center",
                    marker=dict(size=10, color=colors[idx % len(colors)]),
                    name=model_name,
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(title=title, height=800, showlegend=True)
        return fig

    def create_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        words: List[str],
        title: str = "Word Similarity Matrix",
    ) -> go.Figure:
        fig = go.Figure(
            data=go.Heatmap(
                z=similarity_matrix,
                x=words,
                y=words,
                colorscale="RdBu",
                zmid=0,
                text=np.round(similarity_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate="%{x} vs %{y}<br>Similarity: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            height=800,
            width=900,
            xaxis={"side": "bottom"},
            yaxis={"autorange": "reversed"},
        )
        return fig

    def create_cluster_comparison(
        self,
        embeddings: np.ndarray,
        words: List[str],
        cluster_labels: List[int],
        title: str = "Word Clusters",
    ) -> go.Figure:
        if len(words) != len(cluster_labels):
            raise ValueError("Words and cluster labels must have same length")

        df = pd.DataFrame(
            {
                "x": embeddings[:, 0],
                "y": embeddings[:, 1],
                "word": words,
                "cluster": [f"Cluster {c}" for c in cluster_labels],
            }
        )

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            text="word",
            title=title,
            hover_data={"word": True},
        )

        fig.update_traces(textposition="top center", marker=dict(size=10))
        fig.update_layout(height=700, showlegend=True)
        return fig

    def create_radar_comparison(
        self,
        words: List[str],
        model_similarities: Dict[str, Dict[str, float]],
        reference_word: str,
    ) -> go.Figure:
        categories = words
        fig = go.Figure()

        colors = px.colors.qualitative.Set1

        for idx, (model_name, similarities) in enumerate(model_similarities.items()):
            values = [similarities.get(w, 0) for w in words if w != reference_word]
            categories_filtered = [w for w in words if w != reference_word]

            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories_filtered + [categories_filtered[0]],
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=colors[idx % len(colors)]),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
            showlegend=True,
            title=f"Similarity to '{reference_word}' across models",
        )
        return fig

    def clear_cache(self):
        self._tsne_cache.clear()
        self._umap_cache.clear()
