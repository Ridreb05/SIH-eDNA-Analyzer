import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
import joblib
import hdbscan
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# --- All your helper functions (load_models, parse_fasta, etc.) go here ---
# (Pasting the full set of functions from our previous version for completeness)

@st.cache_resource
def load_deep_learning_models():
    try:
        model = load_model('taxa_cnn_model.keras')
        label_encoder = joblib.load('label_encoder.joblib')
        return model, label_encoder
    except Exception as e:
        st.error(f"Model files not found or failed to load! Run 'train_cnn_model.py' first. Error: {e}")
        return None, None

def one_hot_encode(sequence, max_len):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    encoded_seq = np.array([mapping.get(base, [0,0,0,0]) for base in sequence.upper()])
    return np.pad(encoded_seq, ((0, max_len - len(sequence)), (0,0)), 'constant')

def parse_fasta(uploaded_file_content):
    stringio = io.StringIO(uploaded_file_content)
    sequences = [{'id': record.id, 'sequence': str(record.seq)} for record in SeqIO.parse(stringio, "fasta")]
    return pd.DataFrame(sequences)

def generate_saliency_map(model, input_sequence):
    max_len = model.input_shape[1]
    input_tensor = tf.convert_to_tensor(one_hot_encode(input_sequence, max_len).reshape(1, max_len, 4), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        top_prediction_index = tf.argmax(predictions[0])
        top_class_prediction = predictions[:, top_prediction_index]
    gradients = tape.gradient(top_class_prediction, input_tensor)
    saliency_scores = tf.reduce_max(tf.abs(gradients), axis=-1)[0]
    saliency_scores = (saliency_scores - tf.reduce_min(saliency_scores)) / (tf.reduce_max(saliency_scores) - tf.reduce_min(saliency_scores) + 1e-8)
    return saliency_scores.numpy()[:len(input_sequence)]

def plot_saliency_map(sequence, scores):
    fig = go.Figure(data=go.Heatmap(z=[scores], x=list(sequence), y=['Importance'], colorscale='Reds', showscale=False))
    fig.update_layout(title='DNA Saliency Map (Importance of Each Nucleotide for Classification)', xaxis_title="DNA Sequence", yaxis_title="")
    return fig

# --- Main Application ---
st.set_page_config(page_title="DeepGene eDNA Analyzer", layout="wide")

# --- Sidebar Navigation ---
with st.sidebar:
    # You can create a simple logo or use text
    st.title("DeepGene")
    page = st.radio("Navigation", ["🌐 About the Project", "🚀 The Application"])
    st.markdown("---")


# --- Page 1: The "Website" ---
if page == "🌐 About the Project":
    st.title("Unveiling the Secrets of the Deep Sea")
    st.subheader("An AI-Powered Solution for eDNA Biodiversity Analysis")
    st.markdown("---")
    
    st.header("The Challenge")
    st.write(
        """
        The deep ocean is Earth's last great frontier, holding a vast reservoir of undiscovered biodiversity. 
        Traditional methods for studying these ecosystems are slow, invasive, and limited by incomplete genetic databases. 
        This prevents scientists at CMLRE from getting a clear, timely picture of deep-sea life, hindering conservation and discovery.
        """
    )

    st.header("Our Solution: DeepGene")
    st.write(
        """
        **DeepGene** is an intelligent, user-friendly web application that revolutionizes eDNA analysis. 
        Our platform uses a state-of-the-art **Hybrid AI Pipeline** to deliver rapid and accurate biodiversity insights.
        """
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 Deep Learning Classification")
        st.write("A Convolutional Neural Network (CNN) instantly identifies known species with high confidence.")
    with col2:
        st.subheader("🔬 Unsupervised Discovery")
        st.write("An HDBSCAN algorithm analyzes unknown sequences to discover and cluster potential novel species.")

    st.subheader("✨ Explainable AI (XAI)")
    st.write("Our unique DNA Saliency Maps allow scientists to see *why* the AI made a decision, turning a black box into a transparent scientific tool.")
    
    st.markdown("---")
    st.header("The Team")
    st.write("`Team Name: DeepGene`")
    st.write("`Debanik Das`")
    st.write("`Aditya Sarkar`")
    st.write("`Suwastik Bhattachraya`")


# --- Page 2: The "Application" ---
elif page == "🚀 The Application":
    model, label_encoder = load_deep_learning_models()
    
    st.title("✨ DeepGene: XAI eDNA Biodiversity Analyzer")
    
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa"])
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
        run_button = st.button("Analyze with XAI", type="primary", use_container_width=True)

    if model and label_encoder:
        if run_button and uploaded_file:
            df = parse_fasta(uploaded_file.getvalue().decode("utf-8"))
            st.session_state.df = df
            st.success(f"Loaded {len(df)} sequences.")

            X_pred = np.array([one_hot_encode(seq, model.input_shape[1]) for seq in df['sequence']])
            probabilities = model.predict(X_pred)
            predictions_int = np.argmax(probabilities, axis=1)
            df['predicted_taxon'] = label_encoder.inverse_transform(predictions_int)
            df['confidence'] = probabilities.max(axis=1)
            st.session_state.df = df
        
        if 'df' in st.session_state:
            df = st.session_state.df
            known_df = df[df['confidence'] >= confidence_threshold]
            unknown_df = df[df['confidence'] < confidence_threshold]
            
            st.header("Analysis Dashboard")
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "✅ Known Taxa (XAI)", "🔍 Novel Taxa (HDBSCAN)"])

            with tab1:
                st.metric("Known Sequences", len(known_df))
                st.metric("Unknown Sequences", len(unknown_df))

            with tab2:
                if not known_df.empty:
                    st.bar_chart(known_df['predicted_taxon'].value_counts())
                    for index, row in known_df.iterrows():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        col1.text(f"ID: {row['id']}")
                        col2.text(f"Prediction: {row['predicted_taxon']} (Conf: {row['confidence']:.2f})")
                        if col3.button("View Saliency Map", key=f"btn_{index}"):
                            with st.spinner("Generating Saliency Map..."):
                                saliency_scores = generate_saliency_map(model, row['sequence'])
                                saliency_plot = plot_saliency_map(row['sequence'], saliency_scores)
                                st.plotly_chart(saliency_plot, use_container_width=True)
                else:
                    st.info("No sequences identified as known taxa.")

            with tab3:
                if not unknown_df.empty:
                    from sklearn.feature_extraction.text import CountVectorizer
                    vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
                    unknown_vectors = vectorizer.fit_transform(unknown_df['sequence'])
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
                    unknown_df['cluster_id'] = clusterer.fit_predict(unknown_vectors)
                    unknown_df['cluster_id'] = unknown_df['cluster_id'].apply(lambda x: f'Novel Cluster {x+1}' if x != -1 else 'Noise')
                    st.dataframe(unknown_df[['id', 'cluster_id']])
                else:
                    st.info("No sequences to analyze for novelty.")
    else:
        st.info("Upload a file and click 'Analyze' to begin.")
