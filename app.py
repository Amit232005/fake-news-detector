# Streamlit Fake News Detector App — Enhanced Version
# File: app.py
# Run with: streamlit run app.py

"""
Enhanced Streamlit Fake News Detector
Features added by assistant automatically:
- About Developer card with Lottie animation
- Text-to-Speech (gTTS) option to play the result
- Confidence gauge (Plotly) visualization
- Wordclouds for Real vs Fake top words (requires wordcloud)
- "Verify on Google News" button (opens search in new tab)
- Theme toggle (light/dark) via CSS

Notes:
- Optional packages: plotly, wordcloud, gtts
- Place tfidf_logreg_model.joblib and tfidf_vectorizer.joblib in the same folder
- Install requirements: pip install -r requirements.txt
  Suggested extras: plotly wordcloud gTTS
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie
import base64
import re
import time
import io
import matplotlib.pyplot as plt

# Optional imports (wrapped)
try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ------------------------- Helpers -------------------------

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().strip()
    return ' '.join(text.split())


@st.cache_resource
def load_model_and_vectorizer(model_path='tfidf_logreg_model.joblib', vec_path='tfidf_vectorizer.joblib'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


# ------------------------- Page config & CSS -------------------------
st.set_page_config(page_title='✨ Fake News Detector', page_icon='📰', layout='wide')

# Theme toggle (simple)
if 'dark' not in st.session_state:
    st.session_state.dark = True

def toggle_theme():
    st.session_state.dark = not st.session_state.dark

st.markdown("""
<style>
:root{--bg:#0b1020; --card:#0f172a; --muted:#94a3b8; --text:#e6eef8}
.light{--bg:#f6f8fb; --card:#ffffff; --muted:#6b7280; --text:#0b1220}
body{background:var(--bg); color:var(--text)}
.header-card{background: linear-gradient(90deg,#0f172a,#1e3a8a);color: white;padding:14px;border-radius:12px}
.card {background-color: rgba(255,255,255,0.02); padding: 12px; border-radius: 10px}
.muted {color: var(--muted)}
.small {font-size:14px}
</style>
""", unsafe_allow_html=True)

if st.session_state.dark:
    st.markdown("<div id='theme' class=''>", unsafe_allow_html=True)
else:
    st.markdown("<div id='theme' class='light'>", unsafe_allow_html=True)

# ------------------------- Header -------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="header-card"><h1 style="margin:0">📰 Fake News Detector</h1><div class="muted small">TF-IDF + Logistic Regression — Enhanced demo</div></div>', unsafe_allow_html=True)
with col2:
    if st.button('Toggle theme'):
        toggle_theme()

# subtitle and animation
st.markdown("<div style='margin-top:8px'><em class='small muted'>Protect feeds · Detect misinformation · Help fact-checks</em></div>", unsafe_allow_html=True)

# layout
main, side = st.columns([3,1])

with side:
    lottie_url = 'https://assets10.lottiefiles.com/packages/lf20_hzgq1iov.json'
    lottie = load_lottieurl(lottie_url)
    if lottie:
        st_lottie(lottie, height=220)
    st.markdown('**About Developer**')
    st.markdown('- Name: Amit Barik, Anshu Kumari, Rohit Prasad')
    st.markdown('- College: Netaji Subhas University')
    st.markdown('- Project: Fake News Detector (TF-IDF & LogReg)')
    st.markdown('')
    st.markdown('**Notes**')
    st.markdown('- Optional: install extras for audio, wordcloud, plotly')
    st.markdown('- gTTS for voice: pip install gTTS')
    st.markdown('- wordcloud: pip install wordcloud')
    st.markdown('- plotly: pip install plotly')

with main:
    tabs = st.tabs(['Predict single', 'Batch CSV', 'Visuals & Explainability'])

    # ------------------------- Tab 1: Single prediction -------------------------
    with tabs[0]:
        st.subheader('Predict a single article')
        st.markdown('Provide a headline and/or article body below.')
        headline = st.text_input('Headline (optional)', '')
        body = st.text_area('Article body (or paste headline here)', height=200)
        short_mode = st.checkbox('Short mode: treat input as headline only', value=False)
        tts_opt = st.checkbox('Play result aloud (Text-to-Speech)', value=False)
        detect_button = st.button('Predict')

        if detect_button:
            model, vectorizer = load_model_and_vectorizer()
            text = (headline + ' ' + body).strip()
            if len(text) == 0:
                st.error('Please provide some text.')
            else:
                with st.spinner('Analyzing...'):
                    cleaned = clean_text(text if not short_mode else headline or body)
                    X = vectorizer.transform([cleaned])
                    probs = model.predict_proba(X)[0]
                    pred = model.predict(X)[0]
                    label = 'REAL' if pred == 1 else 'FAKE'
                    confidence = float(max(probs))

                # Show result card
                color = '#16a34a' if label == 'REAL' else '#dc2626'
                st.markdown(f"<div class='card'><h2>Prediction: <span style='color:{color}'>{label}</span></h2><p class='muted'>Confidence: {confidence*100:.1f}%</p></div>", unsafe_allow_html=True)

                # Confidence gauge (plotly)
                if HAS_PLOTLY:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence*100,
                        title = {'text': "Confidence (%)"},
                        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.progress(int(confidence*100))

                # Show influential words
                st.markdown('**Top influential words (approx.)**')
                coefs = model.coef_[0]
                feat_names = vectorizer.get_feature_names_out()
                tfidf_vals = X.toarray()[0]
                contrib = tfidf_vals * coefs
                top_idx = np.argsort(contrib)[-12:][::-1]
                rows = []
                for idx in top_idx:
                    if tfidf_vals[idx] > 0:
                        rows.append((feat_names[idx], float(tfidf_vals[idx]), float(contrib[idx])))
                if rows:
                    df_rows = pd.DataFrame(rows, columns=['word', 'tfidf', 'contribution']).sort_values('contribution', ascending=False)
                    st.table(df_rows.head(10))
                else:
                    st.info('No prominent TF-IDF words found in this short text.')

                # Text-to-speech (gTTS)
                if tts_opt:
                    if not HAS_GTTS:
                        st.warning('gTTS not installed. Install with: pip install gTTS')
                    else:
                        try:
                            tts = gTTS(text=f"The article appears to be {label} with {confidence*100:.1f} percent confidence.")
                            mp3_fp = io.BytesIO()
                            tts.write_to_fp(mp3_fp)
                            mp3_fp.seek(0)
                            st.audio(mp3_fp.read(), format='audio/mp3')
                        except Exception as e:
                            st.error('Text-to-speech failed: ' + str(e))

                # Verify button (Google News search)
                query = requests.utils.quote(headline or (text[:100]))
                url = f"https://www.google.com/search?q={query}&tbm=nws"
                st.markdown(f"[🔍 Verify on Google News]({url})")

    # ------------------------- Tab 2: Batch CSV -------------------------
    with tabs[1]:
        st.subheader('Batch predict using CSV')
        st.markdown('Upload CSV containing a column named `text`, `content`, `article`, `body`, or `headline`.')
        uploaded = st.file_uploader('Upload CSV', type=['csv'])
        if st.button('Show sample CSV'):
            try:
                sample = pd.read_csv('sample_test_for_streamlit.csv')
                st.dataframe(sample.head())
            except Exception:
                st.warning('Sample file not found. Run the notebook to create sample_test_for_streamlit.csv or upload your own CSV.')

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            candidate_cols = [c for c in df.columns if c.lower() in ('text','content','article','body','headline')]
            if not candidate_cols:
                st.error("Couldn't find a suitable text column (searching for text/content/article/body/headline).")
            else:
                col = st.selectbox('Select text column', candidate_cols)
                if st.button('Run batch prediction'):
                    model, vectorizer = load_model_and_vectorizer()
                    texts = df[col].fillna('').astype(str).apply(clean_text).tolist()
                    st.info('Vectorizing...')
                    X = vectorizer.transform(texts)
                    st.info('Predicting...')
                    preds = model.predict(X)
                    probs = model.predict_proba(X)
                    df['pred_label'] = np.where(preds==1, 'REAL', 'FAKE')
                    df['pred_prob_real'] = probs[:,1]
                    st.dataframe(df.head(50))
                    csv = df.to_csv(index=False).encode('utf-8')
                    b64 = base64.b64encode(csv).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions.csv</a>'
                    st.markdown(href, unsafe_allow_html=True)

    # ------------------------- Tab 3: Visuals & Explainability -------------------------
    with tabs[2]:
        st.subheader('Visuals & Explainability')
        st.markdown('Generate wordclouds for Real vs Fake and inspect top model words.')
        if st.button('Show top model words'):
            model, vectorizer = load_model_and_vectorizer()
            feat_names = vectorizer.get_feature_names_out()
            coefs = model.coef_[0]
            top_real = np.argsort(coefs)[-50:][::-1]
            top_fake = np.argsort(coefs)[:50]
            st.markdown('**Top words indicating REAL**')
            st.write([f'{feat_names[i]} ({coefs[i]:.4f})' for i in top_real[:30]])
            st.markdown('**Top words indicating FAKE**')
            st.write([f'{feat_names[i]} ({coefs[i]:.4f})' for i in top_fake[:30]])

        if HAS_WORDCLOUD:
            st.markdown('**Wordclouds**')
            try:
                model, vectorizer = load_model_and_vectorizer()
                feat_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                # prepare frequencies by exponentiating coefficients (visual effect)
                real_freq = {feat_names[i]: max(coefs[i],0)+0.01 for i in range(len(feat_names))}
                fake_freq = {feat_names[i]: abs(min(coefs[i],0))+0.01 for i in range(len(feat_names))}

                wc_real = WordCloud(width=600, height=300).generate_from_frequencies(real_freq)
                wc_fake = WordCloud(width=600, height=300).generate_from_frequencies(fake_freq)

                fig, axes = plt.subplots(1,2, figsize=(14,6))
                axes[0].imshow(wc_real, interpolation='bilinear')
                axes[0].axis('off')
                axes[0].set_title('Real-indicating words')
                axes[1].imshow(wc_fake, interpolation='bilinear')
                axes[1].axis('off')
                axes[1].set_title('Fake-indicating words')
                st.pyplot(fig)
            except Exception as e:
                st.error('Wordcloud generation failed: ' + str(e))
        else:
            st.info('wordcloud not installed. Install with: pip install wordcloud')

# ------------------------- Footer -------------------------
st.markdown('---')
st.markdown('**Notes:** This demo app is for learning and prototyping. For production use, please add robust data pipelines, continuous retraining, and human review.')

# close theme wrapper
st.markdown('</div>', unsafe_allow_html=True)

# End of file
