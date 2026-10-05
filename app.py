import streamlit as st
from model_loader import predict_text
from scrapper import scrape_article

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.write(
    "Powered by a fine-tuned XLM-RoBERTa model (`xlm-roberta-base`)."
)

tab_text, tab_url = st.tabs(["Direct Text Input", "Extract from URL"])

# ----------------- Tab 1: Direct Text Input -----------------
with tab_text:
    title_in = st.text_input("Article Title (Optional):", key="text_title")
    body_in = st.text_area(
        "Article Body / Text:", height=180, key="text_body"
    )

    if st.button("Analyze Article Text", type="primary"):
        # Format text to mirror evaluate.py
        if title_in.strip() and body_in.strip():
            input_text = f"{title_in.strip()}. {body_in.strip()}"
        elif body_in.strip():
            input_text = body_in.strip()
        elif title_in.strip():
            input_text = title_in.strip()
        else:
            input_text = ""

        if not input_text:
            st.warning("Please provide a title or article text.")
        else:
            with st.spinner("Analyzing text..."):
                result = predict_text(input_text)

            if "error" in result:
                st.error(result["error"])
            else:
                label = result["label"]
                score = result["score"]
                if label == "Real":
                    st.success(
                        f"**Prediction:** {label} (Confidence: {score * 100:.2f}%)"
                    )
                else:
                    st.error(
                        f"**Prediction:** {label} (Confidence: {score * 100:.2f}%)"
                    )

# ----------------- Tab 2: URL Scraper -----------------
with tab_url:
    url_in = st.text_input("News Article URL:", key="url_input")

    if st.button("Scrape and Analyze", type="primary"):
        if not url_in.strip():
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Fetching article content..."):
                scraped = scrape_article(url_in.strip())

            if "error" in scraped:
                st.error(scraped["error"])
            else:
                st.subheader("Extracted Content")
                st.text_input(
                    "Title:", value=scraped.get("title", ""), disabled=True
                )
                with st.expander("View Extracted Body Text"):
                    st.write(scraped.get("text", "No body text extracted."))

                with st.spinner("Running classification..."):
                    result = predict_text(scraped["input_text"])

                if "error" in result:
                    st.error(result["error"])
                else:
                    label = result["label"]
                    score = result["score"]
                    if label == "Real":
                        st.success(
                            f"**Prediction:** {label} (Confidence: {score * 100:.2f}%)"
                        )
                    else:
                        st.error(
                            f"**Prediction:** {label} (Confidence: {score * 100:.2f}%)"
                        )