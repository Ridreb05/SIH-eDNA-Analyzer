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
    """Calculates species richness and Shannon diversity index."""
    if not list(labels):
        return 0, 0.0
    unique_labels, counts = np.unique(labels, return_counts=True)
    richness = len(unique_labels)
    probabilities = counts / counts.sum()
    shannon_index = -np.sum(probabilities * np.log2(probabilities))
    return richness, shannon_index

# --- Main Application ---
st.set_page_config(page_title="DeepGene Flagship", layout="wide")

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/df/Amity_University_Kolkata.svg/800px-Amity_University_Kolkata.svg.png", width=150)
    st.title("DeepGene")
    page = st.radio("Navigation", ["🌐 About the Project", "⚙️ How DeepGene Works", "🚀 The Application"], label_visibility="hidden")
    st.markdown("---")

# --- About Page ---
if page == "🌐 About the Project":
    st.title("Unveiling the Secrets of the Deep Sea")
    st.subheader("An AI-Powered Research Platform for eDNA Biodiversity Analysis")
    st.markdown("---")
    st.markdown("### 🌊 The Challenge: A Universe of Unknowns")
    st.write("The deep ocean is Earth's last great frontier... CMLRE scientists need a tool that can navigate this uncharted genetic territory.")
    st.markdown("---")
    st.markdown("### ✨ Our Solution: DeepGene")
    st.write("**DeepGene** is an intelligent, user-friendly web application that revolutionizes eDNA analysis...")
    st.markdown("#### Key Innovations")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("##### 🧠 **Transformer AI Core**")
            st.write("Our model understands the long-range context of DNA, providing superior accuracy for classifying known species.")
    with col2:
        with st.container(border=True):
            st.markdown("##### 🔬 **Unsupervised Discovery Engine**")
            st.write("HDBSCAN and UMAP algorithms work together to find and visualize clusters of potential novel species from unknown data.")
    with st.container(border=True):
        st.markdown("##### 🛰️ **Live NCBI Annotation**")
        st.write("Seamlessly connects our AI's discoveries to the world's largest genetic database, providing real-time biological context to novel findings.")
    st.markdown("---")
    st.markdown("### 👥 The Team")
    st.write("**Team Name:** DeepGene")
    st.write("- **Debanik Das**\n- **Aditya Sarkar**\n- **Suswastik Bhattacharya**")

# --- How it Works Page ---
elif page == "⚙️ How DeepGene Works":
    st.title("Inside the Engine: The DeepGene AI Pipeline")
    st.markdown("---")
    st.image("https://i.imgur.com/8z2gZ2k.png", caption="The DeepGene Data Processing Flowchart")
    with st.expander("Step 1: Data Ingestion & Preprocessing"):
        st.markdown("- **Input:** The process begins when a scientist uploads a standard **FASTA file**...")
    with st.expander("Step 2: The Transformer Core (Supervised Classification)"):
        st.markdown("- **What is a Transformer?** It's the same AI architecture that powers models like GPT...")
    with st.expander("Step 3: The Hybrid Logic Gate"):
        st.markdown("- **High Confidence (e.g., > 0.80):** If the AI is certain... the sequence is classified as **'Known'**.")
    with st.expander("Step 4: The Discovery Engine (Unsupervised Learning)"):
        st.markdown("- **Clustering (HDBSCAN):** The **HDBSCAN** algorithm... groups similar sequences together... into **'Novel Clusters'**.")
    with st.expander("Step 5: Live Annotation (The Final Insight)"):
        st.markdown("- **API Connection:** Our application sends the sequence directly to the **NCBI BLAST server** via its public API.")

# --- Application Page ---
elif page == "🚀 The Application":
    st.title("🛰️ DeepGene: Ecological Discovery Engine")
    
    with st.sidebar:
        st.header("Analysis Controls")
        uploaded_file = st.file_uploader("Upload your FASTA file", type=["fasta", "fa"], help="Upload a standard FASTA file containing your eDNA sequences.")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05, help="The minimum confidence score for our AI to classify a sequence as 'known'.")
        run_button = st.button("Run Full Analysis", type="primary", use_container_width=True)

    if model and label_encoder:
        if run_button and uploaded_file:
            df = parse_fasta(uploaded_file.getvalue().decode("utf-8"))
            max_len = model.input_shape[1]
            X_pred = tf.keras.preprocessing.sequence.pad_sequences([dna_to_integers(seq) for seq in df['sequence']], maxlen=max_len, padding='post')
            with st.spinner("Classifying sequences with Transformer AI..."):
                probabilities = model.predict(X_pred)
            df['predicted_taxon'] = label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
            df['confidence'] = probabilities.max(axis=1)
            st.session_state.df_results = df
            st.session_state.analysis_complete = True
            st.success("Initial classification complete!")
        
        if st.session_state.analysis_complete:
            df = st.session_state.df_results
            known_df = df[df['confidence'] >= confidence_threshold]
            unknown_df = df[df['confidence'] < confidence_threshold].copy()
            
            st.header("Analysis Dashboard")
            tab1, tab2, tab3 = st.tabs(["📊 Overview & Biodiversity", "✅ Known Taxa", "🔬 Novel Taxa Discovery"])
            
            with tab1:
                st.subheader("High-Level Summary")
                col1, col2 = st.columns(2)
                col1.metric("Known Sequences (High Confidence)", len(known_df))
                col2.metric("Unknown Sequences for Discovery", len(unknown_df))
                
                if not known_df.empty:
                    st.subheader("Biodiversity Indices (Known Taxa)")
                    richness, shannon = calculate_biodiversity(known_df['predicted_taxon'])
                    col1, col2 = st.columns(2)
                    col1.metric("Species Richness", f"{richness}")
                    col2.metric("Shannon Index", f"{shannon:.2f}")

            with tab2:
                st.subheader("Composition of Known Taxa")
                if not known_df.empty:
                    st.bar_chart(known_df['predicted_taxon'].value_counts())
                    with st.expander("View Detailed Classification Data"):
                        st.dataframe(known_df[['id', 'predicted_taxon', 'confidence']])
                else: 
                    st.info("No sequences met the confidence threshold to be classified as 'known'.")
            
            with tab3:
                st.subheader("Discovery Engine for Unknown Sequences")
                if not unknown_df.empty:
                    with st.spinner("Mapping the novel genetic space..."):
                        vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
                        unknown_vectors = vectorizer.fit_transform(unknown_df['sequence'])
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
                        unknown_df['cluster_id'] = clusterer.fit_predict(unknown_vectors)
                        
                        # Defensive check for UMAP n_neighbors
                        n_neighbors = max(2, min(15, len(unknown_df)-1))
                        
                        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
                        embedding = reducer.fit_transform(unknown_vectors)
                        unknown_df['umap_x'] = embedding[:, 0]
                        unknown_df['umap_y'] = embedding[:, 1]
                        
                        unknown_df['display_label'] = unknown_df['cluster_id'].apply(lambda x: f'Cluster {x+1}' if x != -1 else 'Noise')
                    
                    st.markdown("#### Interactive Map of Novel Genetic Space (UMAP)")
                    fig = px.scatter(
                        unknown_df, x='umap_x', y='umap_y', color='display_label',
                        hover_data=['id'], title="Hover over points to see Sequence IDs",
                        labels={'color': 'Cluster ID'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### Annotate Discovered Clusters with NCBI BLAST")
                    representative_df = unknown_df[unknown_df['cluster_id'] != -1].groupby('display_label').first().reset_index()
                    
                    if not representative_df.empty:
                        for index, row in representative_df.iterrows():
                            with st.container(border=True):
                                col1, col2 = st.columns([3, 1])
                                col1.text(f"Representative for {row['display_label']}: {row['id']}")
                                if col2.button("BLAST this sequence", key=f"blast_{index}"):
                                    with st.spinner(f"Querying NCBI for {row['id']}... This can take up to a minute."):
                                        annotation = get_blast_annotation(row['sequence'])
                                        st.success(f"**Top NCBI Match:** {annotation}")
                    else:
                        st.info("No stable clusters were found to annotate.")
                else:
                    st.info("No unknown sequences to analyze for novelty.")
    else:
        st.warning("Application requires model files. Please ensure the training script has been run and models are present.")

