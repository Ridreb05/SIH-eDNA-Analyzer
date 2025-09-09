import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
import joblib
import hdbscan
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from Bio.Blast import NCBIWWW, NCBIXML
import umap.umap_ as umap

# --- Define the custom TransformerBlock layer ---
class TransformerBlock(tf.keras.layers.Layer):
    # (The class definition is the same as before)
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.att.key_dim, 'num_heads': self.att.num_heads, 'ff_dim': self.ffn.layers[0].units, 'rate': self.dropout1.rate,})
        return config

# --- Load Pre-trained Models and Config ---
@st.cache_resource
def load_advanced_models():
    try:
        model = load_model('taxa_transformer_model.keras', custom_objects={"TransformerBlock": TransformerBlock})
        label_encoder = joblib.load('label_encoder.joblib')
        max_len = joblib.load('max_len.joblib') # NEW: Load max_len
        return model, label_encoder, max_len
    except Exception as e:
        st.error(f"Model files not found! Run 'train_transformer_model.py' first. Error: {e}")
        return None, None, None

model, label_encoder, max_len = load_advanced_models()

# --- Helper & Analysis Functions (no changes needed) ---
def dna_to_integers(sequence):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return [mapping.get(base, 0) for base in sequence.upper()]

def parse_fasta(uploaded_file_content):
    stringio = io.StringIO(uploaded_file_content)
    sequences = [{'id': record.id, 'sequence': str(record.seq)} for record in SeqIO.parse(stringio, "fasta")]
    return pd.DataFrame(sequences)

@st.cache_data
def get_blast_annotation(sequence):
    try:
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence, hitlist_size=1)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            return blast_record.alignments[0].title
        else:
            return "No significant match found in NCBI nt database."
    except Exception as e:
        return f"Error connecting to BLAST: {e}"

def calculate_biodiversity(labels):
    if not list(labels): return 0, 0.0
    unique_labels, counts = np.unique(labels, return_counts=True)
    richness = len(unique_labels)
    probabilities = counts / counts.sum()
    shannon_index = -np.sum(probabilities * np.log2(probabilities))
    return richness, shannon_index

# --- Main Application ---
st.set_page_config(page_title="DeepGene Flagship", layout="wide")

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# --- Sidebar and Page Navigation (no changes needed) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/df/Amity_University_Kolkata.svg/800px-Amity_University_Kolkata.svg.png", width=150)
    st.title("DeepGene")
    page = st.radio("Navigation", ["🌐 About the Project", "⚙️ How DeepGene Works", "🚀 The Application"], label_visibility="hidden")
    st.markdown("---")

# --- Pages (no changes needed) ---
if page == "🌐 About the Project":
    st.title("Unveiling the Secrets of the Deep Sea")
    # ... (rest of about page)
elif page == "⚙️ How DeepGene Works":
    st.title("Inside the Engine: The DeepGene AI Pipeline")
    # ... (rest of how-it-works page)

elif page == "🚀 The Application":
    st.title("🛰️ DeepGene: Ecological Discovery Engine")
    
    with st.sidebar:
        st.header("Analysis Controls")
        uploaded_file = st.file_uploader("Upload your FASTA file", type=["fasta", "fa"])
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
        run_button = st.button("Run Full Analysis", type="primary", use_container_width=True)

    if model and label_encoder and max_len: # Check for max_len
        if run_button and uploaded_file:
            df = parse_fasta(uploaded_file.getvalue().decode("utf-8"))
            # The max_len is now loaded from the saved file, ensuring perfect consistency
            X_pred = tf.keras.preprocessing.sequence.pad_sequences([dna_to_integers(seq) for seq in df['sequence']], maxlen=max_len, padding='post')
            with st.spinner("Classifying sequences with Stacked Transformer AI..."):
                probabilities = model.predict(X_pred)
            df['predicted_taxon'] = label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
            df['confidence'] = probabilities.max(axis=1)
            st.session_state.df_results = df
            st.session_state.analysis_complete = True
            st.success("Initial classification complete!")
        
        # (The rest of the display logic is identical)
        if st.session_state.analysis_complete:
            # ... (rest of app)
            pass

