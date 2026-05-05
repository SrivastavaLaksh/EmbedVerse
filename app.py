import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_registry import ModelRegistry, ModelManager
from models.similarity import SimilarityAnalyzer
from visualizations.embedding_viz import EmbeddingVisualizer
from data.word_sets import get_all_domains, get_domain_words, TECHNICAL_DOMAIN_WORDS

st.set_page_config(page_title="Embedding Visualizer", page_icon="🔬", layout="wide")

if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager()
if "similarity_analyzer" not in st.session_state:
    st.session_state.similarity_analyzer = SimilarityAnalyzer()
if "visualizer" not in st.session_state:
    st.session_state.visualizer = EmbeddingVisualizer()
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []
if "current_words" not in st.session_state:
    st.session_state.current_words = []

st.title("🔬 Embedding Space Visualizer")
st.markdown("Compare how different models represent words in their embedding spaces")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Visualization",
        "🔍 Similarity Analysis",
        "📈 Model Comparison",
        "📋 Word Coverage",
    ]
)

with tab1:
    st.header("Embedding Visualization")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Settings")

        available_models = st.session_state.model_manager.get_available_models()

        all_models = []
        model_options = {}
        for category, models in available_models.items():
            for model in models:
                full_key = model
                all_models.append(f"{category}: {model}")
                model_options[f"{category}: {model}"] = full_key

        selected_model_names = st.multiselect(
            "Select Models",
            all_models,
            default=all_models[:2] if len(all_models) >= 2 else all_models,
        )

        domain = st.selectbox("Domain", ["custom"] + get_all_domains())

        if domain == "custom":
            words_input = st.text_area(
                "Enter words (comma-separated)",
                value="doctor, hospital, medicine, disease, patient, surgery, nurse, treatment, diagnosis, prescription",
                height=100,
            )
            words = [w.strip() for w in words_input.split(",") if w.strip()]
        else:
            words = get_domain_words(domain)
            st.info(f"Using {len(words)} words from {domain} domain")

        visualization_method = st.selectbox(
            "Dimensionality Reduction", ["t-SNE", "UMAP"]
        )

        perplexity = (
            st.slider("t-SNE Perplexity", 2, 30, 5)
            if visualization_method == "t-SNE"
            else None
        )
        n_neighbors = (
            st.slider("UMAP N Neighbors", 5, 50, 15)
            if visualization_method == "UMAP"
            else None
        )
        min_dist = (
            st.slider("UMAP Min Distance", 0.0, 1.0, 0.1)
            if visualization_method == "UMAP"
            else None
        )

        if st.button("Visualize", type="primary"):
            st.session_state.selected_models = [
                model_options[name] for name in selected_model_names
            ]
            st.session_state.current_words = words

            for model_key in st.session_state.selected_models:
                st.session_state.model_manager.add_model(model_key)

    with col2:
        if st.session_state.selected_models and st.session_state.current_words:
            with st.spinner("Generating visualization..."):
                models_data = {}

                for model_key in st.session_state.selected_models:
                    model = st.session_state.model_manager.models[model_key]
                    embeddings_list = []
                    valid_words = []

                    for word in st.session_state.current_words:
                        emb = model.get_embedding(word)
                        if emb is not None:
                            embeddings_list.append(emb)
                            valid_words.append(word)

                    if embeddings_list:
                        models_data[model_key] = (
                            np.array(embeddings_list),
                            valid_words,
                        )

                if len(models_data) >= 2:
                    fig = st.session_state.visualizer.create_multi_model_comparison(
                        models_data,
                        method=visualization_method.lower(),
                        title=f"{visualization_method} Projection: {', '.join(st.session_state.current_words[:5])}...",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif len(models_data) == 1:
                    model_name = list(models_data.keys())[0]
                    embeddings, words = models_data[model_name]

                    if visualization_method == "t-SNE":
                        reduced = st.session_state.visualizer.reduce_dimensions_tsne(
                            embeddings, perplexity=perplexity
                        )
                    else:
                        reduced = st.session_state.visualizer.reduce_dimensions_umap(
                            embeddings, n_neighbors=n_neighbors, min_dist=min_dist
                        )

                    fig = st.session_state.visualizer.create_scatter_plot(
                        reduced,
                        words,
                        title=f"{model_name}: {visualization_method} Projection",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select models and enter words to visualize embeddings")

with tab2:
    st.header("Word Similarity Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        target_word = st.text_input("Target Word", value="doctor")

        available_models_simple = []
        for (
            category,
            models,
        ) in st.session_state.model_manager.get_available_models().items():
            for model in models:
                available_models_simple.append(model)

        analysis_model = st.selectbox("Model", available_models_simple)

        if st.button("Find Similar Words", type="primary"):
            if analysis_model not in st.session_state.model_manager.models:
                st.session_state.model_manager.add_model(analysis_model)

            model = st.session_state.model_manager.models[analysis_model]
            similar_words = st.session_state.similarity_analyzer.rank_similar_words(
                model, target_word, top_k=20
            )

            if similar_words:
                st.session_state.similar_words_result = similar_words

    with col2:
        if "similar_words_result" in st.session_state:
            words = [w for w, _ in st.session_state.similar_words_result]
            similarities = [s for _, s in st.session_state.similar_words_result]

            df = pd.DataFrame({"Word": words, "Similarity": similarities})
            st.dataframe(df, use_container_width=True)

with tab3:
    st.header("Cross-Model Comparison")

    col1, col2 = st.columns([1, 2])

    with col1:
        comparison_words_input = st.text_area(
            "Words to Compare (comma-separated)",
            value="stock, bond, market, investment, trading",
            height=100,
        )
        comparison_words = [
            w.strip() for w in comparison_words_input.split(",") if w.strip()
        ]

        comparison_models = st.multiselect(
            "Models to Compare",
            available_models_simple[:6],
            default=available_models_simple[:3]
            if len(available_models_simple) >= 3
            else available_models_simple,
        )

        if st.button("Compare Models", type="primary"):
            for model_key in comparison_models:
                st.session_state.model_manager.add_model(model_key)

            st.session_state.comparison_result = True

    with col2:
        if "comparison_result" in st.session_state and comparison_models:
            with st.spinner("Computing similarities..."):
                models_dict = {
                    k: st.session_state.model_manager.models[k]
                    for k in comparison_models
                    if k in st.session_state.model_manager.models
                }

                if len(models_dict) >= 2:
                    similarity_data = []

                    for i, w1 in enumerate(comparison_words):
                        for j, w2 in enumerate(comparison_words):
                            if i >= j:
                                continue
                            row = {"Word Pair": f"{w1} - {w2}"}
                            for model_name, model in models_dict.items():
                                emb1 = model.get_embedding(w1)
                                emb2 = model.get_embedding(w2)
                                if emb1 is not None and emb2 is not None:
                                    sim = st.session_state.similarity_analyzer.cosine_similarity(
                                        emb1, emb2
                                    )
                                    row[model_name] = round(sim, 4)
                            similarity_data.append(row)

                    if similarity_data:
                        df = pd.DataFrame(similarity_data)
                        st.dataframe(df, use_container_width=True)

                        agreements = []
                        for i, w1 in enumerate(comparison_words):
                            for j, w2 in enumerate(comparison_words):
                                if i >= j:
                                    continue
                                sims = []
                                for model_name, model in models_dict.items():
                                    emb1 = model.get_embedding(w1)
                                    emb2 = model.get_embedding(w2)
                                    if emb1 is not None and emb2 is not None:
                                        sim = st.session_state.similarity_analyzer.cosine_similarity(
                                            emb1, emb2
                                        )
                                        sims.append(sim)

                                if len(sims) >= 2:
                                    agreement = 1 - np.std(sims)
                                    agreements.append(
                                        {
                                            "Word Pair": f"{w1} - {w2}",
                                            "Agreement": round(agreement, 4),
                                            "Avg Similarity": round(np.mean(sims), 4),
                                        }
                                    )

                        if agreements:
                            st.subheader("Model Agreement Analysis")
                            df_agreement = pd.DataFrame(agreements)
                            df_agreement = df_agreement.sort_values(
                                "Agreement", ascending=False
                            )
                            st.dataframe(df_agreement, use_container_width=True)
                    else:
                        st.warning("No common words found across selected models")
                else:
                    st.warning("Select at least 2 models for comparison")

with tab4:
    st.header("Word Coverage Analysis")

    if st.session_state.model_manager.models:
        col1, col2 = st.columns([1, 2])

        with col1:
            coverage_words = st.text_area(
                "Words to Check Coverage",
                value="doctor, hospital, medicine, disease, patient, stock, bond, market, atom, molecule",
                height=150,
            )
            coverage_words_list = [
                w.strip() for w in coverage_words.split(",") if w.strip()
            ]

        with col2:
            coverage = st.session_state.model_manager.check_word_coverage(
                coverage_words_list
            )

            coverage_data = []
            for word in coverage_words_list:
                row = {"Word": word}
                total_available = 0
                for model_name, word_coverage in coverage.items():
                    row[model_name] = "✓" if word_coverage.get(word, False) else "✗"
                    if word_coverage.get(word, False):
                        total_available += 1
                row["Coverage"] = f"{total_available}/{len(coverage)}"
                coverage_data.append(row)

            df_coverage = pd.DataFrame(coverage_data)
            st.dataframe(df_coverage, use_container_width=True)

            coverage_rates = {}
            for model_name, word_coverage in coverage.items():
                available = sum(
                    1 for w in coverage_words_list if word_coverage.get(w, False)
                )
                coverage_rates[model_name] = available / len(coverage_words_list) * 100

            st.bar_chart(pd.DataFrame({"Coverage %": coverage_rates}))
    else:
        st.info("Add models first to check word coverage")

st.sidebar.header("Quick Actions")
if st.sidebar.button("Clear All Models"):
    st.session_state.model_manager = ModelManager()
    st.session_state.selected_models = []
    st.rerun()

st.sidebar.info(
    "This tool helps visualize and compare how different embedding models "
    "represent words in vector space. Select domain-specific words to see "
    "how models trained on different corpora capture semantic relationships."
)
