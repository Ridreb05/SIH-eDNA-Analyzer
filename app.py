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

# --- Load Pre-trained Transformer Model ---
@st.cache_resource
def load_advanced_models():
    try:
        model = load_model('taxa_transformer_model.keras', custom_objects={"TransformerBlock": TransformerBlock})
        label_encoder = joblib.load('label_encoder.joblib')
        return model, label_encoder
    except Exception as e:
        st.error(f"Model files not found! Run 'train_transformer_model.py' first. Error: {e}")
        return None, None

model, label_encoder = load_advanced_models()

# --- Helper & Analysis Functions ---
def dna_to_integers(sequence):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return [mapping.get(base, 0) for base in sequence.upper()]

def parse_fasta(uploaded_file_content):
    stringio = io.StringIO(uploaded_file_content)
    sequences = [{'id': record.id, 'sequence': str(record.seq)} for record in SeqIO.parse(stringio, "fasta")]
    return pd.DataFrame(sequences)

@st.cache_data
def get_blast_annotation(sequence):
    """Sends a sequence to NCBI BLAST and returns the top hit."""
    try:
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence, hitlist_size=1)
        blast_record = NCBIXML.read(result_handle)
        if blast_record.alignments:
            return blast_record.alignments[0].title
        else:
            return "No significant match found in NCBI nt database."
    except Exception as e:
        return f"Error connecting to BLAST: {e}"

# --- Main Application ---
st.set_page_config(page_title="DeepGene Flagship", layout="wide")

# (Session state initialization and sidebar are the same)

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# --- Sidebar ---
with st.sidebar:
    st.title("DeepGene")
    page = st.radio("Navigation", ["🌐 About the Project", "🚀 The Application"])
    st.markdown("---")

# --- About Page ---
if page == "🌐 About the Project":
    st.title("Unveiling the Secrets of the Deep Sea")
    st.subheader("A State-of-the-Art AI Research Platform for eDNA Analysis")
    # ... (Rest of about page)

# --- Application Page ---
elif page == "🚀 The Application":
    st.title("🛰️ DeepGene: Ecological Discovery Engine")
    
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa"])
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
        run_button = st.button("Run Full Analysis", type="primary", use_container_width=True)

    if model and label_encoder:
        if run_button and uploaded_file:
            # (Analysis logic is the same)
            df = parse_fasta(uploaded_file.getvalue().decode("utf-8"))
            max_len = model.input_shape[1]
            X_pred = tf.keras.preprocessing.sequence.pad_sequences([dna_to_integers(seq) for seq in df['sequence']], maxlen=max_len, padding='post')
            probabilities = model.predict(X_pred)
            df['predicted_taxon'] = label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
            df['confidence'] = probabilities.max(axis=1)
            st.session_state.df_results = df
            st.session_state.analysis_complete = True
        
        if st.session_state.analysis_complete:
            df = st.session_state.df_results
            known_df = df[df['confidence'] >= confidence_threshold]
            unknown_df = df[df['confidence'] < confidence_threshold].copy() # Use .copy() to avoid warnings
            
            st.header("Analysis Dashboard")
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "✅ Known Taxa (Transformer)", "🔬 Novel Taxa Discovery Engine"])
            
            with tab1:
                # (Overview tab is the same)
                st.metric("Known Sequences", len(known_df))
                st.metric("Unknown Sequences", len(unknown_df))

            with tab2:
                # (Known Taxa tab is the same)
                if not known_df.empty:
                    st.bar_chart(known_df['predicted_taxon'].value_counts())
                else: st.info("No sequences identified as known taxa.")
            
            with tab3:
                st.subheader("Analysis of Unknown Sequences")
                if not unknown_df.empty:
                    # Unsupervised Discovery with HDBSCAN
                    vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
                    unknown_vectors = vectorizer.fit_transform(unknown_df['sequence'])
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
                    unknown_df['cluster_id'] = clusterer.fit_predict(unknown_vectors)
                    
                    # --- NEW: Interactive UMAP Visualization ---
                    st.subheader("Interactive Map of Novel Genetic Space (UMAP)")
                    reducer = umap.UMAP(n_neighbors=max(2, len(unknown_df)//10), min_dist=0.1, n_components=2, random_state=42)
                    embedding = reducer.fit_transform(unknown_vectors)
                    unknown_df['umap_x'] = embedding[:, 0]
                    unknown_df['umap_y'] = embedding[:, 1]
                    
                    # Create a human-readable label
                    unknown_df['display_label'] = unknown_df['cluster_id'].apply(lambda x: f'Cluster {x+1}' if x != -1 else 'Noise')
                    
                    fig = px.scatter(
                        unknown_df, x='umap_x', y='umap_y', color='display_label',
                        hover_data=['id'], title="Discovered Novel Clusters",
                        labels={'color': 'Cluster ID'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- NEW: Live BLAST Annotation ---
                    st.subheader("Annotate Discovered Clusters with NCBI BLAST")
                    st.write("Select a representative sequence from each cluster to find its closest known relative.")
                    
                    representative_df = unknown_df[unknown_df['cluster_id'] != -1].groupby('display_label').first().reset_index()
                    
                    if not representative_df.empty:
                        for index, row in representative_df.iterrows():
                            col1, col2 = st.columns([3, 1])
                            col1.text(f"Representative for {row['display_label']}: {row['id']}")
                            if col2.button("BLAST this sequence", key=f"blast_{index}"):
                                with st.spinner(f"Querying NCBI for {row['id']}... This can take up to a minute."):
                                    annotation = get_blast_annotation(row['sequence'])
                                    st.success(f"**Top NCBI Match:** {annotation}")
                    else:
                        st.info("No stable clusters were found to annotate.")
                else:
                    st.info("No sequences to analyze for novelty.")

