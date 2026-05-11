import time
from typing import List

import streamlit as st
from transformers import pipeline


# =========================
# Streamlit Page Settings
# =========================
st.set_page_config(
    page_title="German → English: Translate & Summarize",
    page_icon="🇩🇪➡️🇬🇧",
    layout="wide",
)
st.title("🇩🇪 ➜ 🇬🇧 German → English: Translate & Summarize")
st.caption("Fast, simple, and reliable — powered by Hugging Face pipelines.")


# =========================
# Model Loaders (cached)
# =========================
@st.cache_resource(show_spinner=True)
def load_translation_pipeline():
    """
    Loads a robust German→English translation model.
    Uses MarianMT: Helsinki-NLP/opus-mt-de-en
    """
    return pipeline(
        task="translation",
        model="Helsinki-NLP/opus-mt-de-en",
        tokenizer="Helsinki-NLP/opus-mt-de-en",
    )


@st.cache_resource(show_spinner=True)
def load_summarization_pipeline():
    """
    Loads an English summarization model.
    BART-Large-CNN is a strong, general-purpose summarizer.
    """
    return pipeline(
        task="summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
    )


translator = load_translation_pipeline()
summarizer = load_summarization_pipeline()


# =========================
# Utilities
# =========================
def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Naive character-based chunking with overlap to stay within model limits.
    Works well enough for translation/summarization pipelines.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap  # step back overlap for context continuity
        if start < 0:
            start = 0
    return chunks


def translate_de_to_en(text_de: str) -> str:
    """
    Translate German → English with chunking for long inputs.
    """
    chunks = chunk_text(text_de, max_chars=2000, overlap=200)
    outputs = []
    for ch in chunks:
        out = translator(ch, max_length=2048)
        outputs.append(out[0]["translation_text"])
    return " ".join(outputs).strip()


def summarize_german_text(text_de: str) -> str:
    """
    Summarize German text into concise English:
      1) Translate German → English
      2) Summarize English
      3) If long, do map-reduce style summarization
    """
    # Step 1: Translate whole text (with chunking inside)
    english_text = translate_de_to_en(text_de)

    # Step 2: If the English text is long, chunk before summarization
    en_chunks = chunk_text(english_text, max_chars=2200, overlap=250)

    partial_summaries = []
    for ch in en_chunks:
        # Typical BART limits; adjust as needed
        result = summarizer(
            ch,
            max_length=180,
            min_length=60,
            do_sample=False,
        )
        partial_summaries.append(result[0]["summary_text"])

    if len(partial_summaries) == 1:
        return partial_summaries[0].strip()

    # Step 3: Reduce — summarize the concatenation of partial summaries
    combined = " ".join(partial_summaries)
    reduced = summarizer(
        combined,
        max_length=200,
        min_length=60,
        do_sample=False,
    )[0]["summary_text"]
    return reduced.strip()


# =========================
# UI
# =========================
st.subheader("Input (German)")
german_text = st.text_area("Paste German text here:", height=220)

col1, col2 = st.columns(2)

with col1:
    if st.button("Translate to English", type="primary", use_container_width=True):
        if not german_text.strip():
            st.warning("Please paste some German text first.")
        else:
            with st.spinner("Translating German → English…"):
                t0 = time.time()
                try:
                    translation = translate_de_to_en(german_text)
                    dt = time.time() - t0
                    st.success(f"Done in {dt:.2f}s")
                    st.markdown("### English Translation")
                    st.write(translation)
                except Exception as e:
                    st.error(f"Translation error: {e}")

with col2:
    if st.button("Summarize in English", use_container_width=True):
        if not german_text.strip():
            st.warning("Please paste some German text first.")
        else:
            with st.spinner("Summarizing (German → English summary)…"):
                t0 = time.time()
                try:
                    summary = summarize_german_text(german_text)
                    dt = time.time() - t0
                    st.success(f"Done in {dt:.2f}s")
                    st.markdown("### English Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Summarization error: {e}")

st.divider()
st.caption(
    "Tips: For best results on very long inputs, use the Summarize function — it will translate and then compress the content."
)
