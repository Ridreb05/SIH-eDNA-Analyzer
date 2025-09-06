import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
import joblib
import hdbscan
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Define the custom TransformerBlock layer ---
# This is necessary for loading the model if the custom layer isn't automatically recognized.
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super().get_config()
        # Add layer-specific parameters to the config
        # Note: These parameters should match the ones used during initialization
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

# --- Load Pre-trained Transformer Model ---
@st.cache_resource
def load_advanced_models():
    """Loads the pre-trained Transformer model and label encoder."""
    try:
        # Register the custom layer so Keras knows what "TransformerBlock" is when loading the model
        model = load_model('taxa_transformer_model.keras', custom_objects={"TransformerBlock": TransformerBlock})
        label_encoder = joblib.load('label_encoder.joblib')
        return model, label_encoder
    except Exception as e:
        st.error(f"Model files not found or failed to load! Run 'train_transformer_model.py' first. Error: {e}")
        return None, None

model, label_encoder = load_advanced_models()

# --- Helper Functions ---
def dna_to_integers(sequence):
    """Converts a DNA sequence string to a list of integers."""
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4} # 0 is reserved for padding
    return [mapping.get(base, 0) for base in sequence.upper()]

def parse_fasta(uploaded_file_content):
    """Parses a FASTA file from its content and returns a DataFrame."""
    stringio = io.StringIO(uploaded_file_content)
    sequences = [{'id': record.id, 'sequence': str(record.seq)} for record in SeqIO.parse(stringio, "fasta")]
    return pd.DataFrame(sequences)

# --- Main Application ---
st.set_page_config(page_title="DeepGene Transformer", layout="wide")

# --- Initialize session state ---
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = pd.DataFrame()

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("DeepGene")
    page = st.radio("Navigation", ["🌐 About the Project", "🚀 The Application"])
    st.markdown("---")

# --- Page 1: The "Website" ---
if page == "🌐 About the Project":
    st.title("Unveiling the Secrets of the Deep Sea")
    st.subheader("A State-of-the-Art AI Solution for eDNA Biodiversity Analysis")
    st.markdown("---")
    st.header("The Challenge")
    st.write("Traditional eDNA analysis is limited by incomplete genetic databases, preventing the discovery of novel deep-sea organisms and hindering conservation efforts.")
    st.header("Our Solution: DeepGene Transformer")
    st.write(
        """
        **DeepGene** uses a state-of-the-art **Transformer** model, the same AI architecture powering models like GPT. 
        Unlike simpler models that only look at local patterns, our Transformer understands the long-range context and complex relationships within a DNA sequence, leading to more accurate classifications and deeper biological insights.
        """
    )
    st.markdown("---")
    st.header("The Team")
    st.write("`Team Name: DeepGene`")
    st.write("`Debanik Das`")
    st.write("`Aditya Sarkar`")
    st.write("`Suwastik Bhattachraya`")

# --- Page 2: The "Application" ---
elif page == "🚀 The Application":
    st.title("🛰️ DeepGene: Transformer-Powered eDNA Analyzer")
    
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa"])
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
        run_button = st.button("Analyze with Transformer AI", type="primary", use_container_width=True)

    if model and label_encoder:
        if run_button and uploaded_file:
            st.session_state.analysis_complete = False # Reset state
            df = parse_fasta(uploaded_file.getvalue().decode("utf-8"))
            st.success(f"Loaded {len(df)} sequences.")

            with st.spinner("Analyzing with Transformer model..."):
                # Preprocess data for the Transformer
                max_len = model.input_shape[1]
                X_pred = tf.keras.preprocessing.sequence.pad_sequences(
                    [dna_to_integers(seq) for seq in df['sequence']], maxlen=max_len, padding='post'
                )
                
                # Supervised Classification with Transformer
                probabilities = model.predict(X_pred)
                predictions_int = np.argmax(probabilities, axis=1)
                df['predicted_taxon'] = label_encoder.inverse_transform(predictions_int)
                df['confidence'] = probabilities.max(axis=1)
            
            st.session_state.df_results = df
            st.session_state.analysis_complete = True
        
        if st.session_state.analysis_complete:
            df = st.session_state.df_results
            known_df = df[df['confidence'] >= confidence_threshold]
            unknown_df = df[df['confidence'] < confidence_threshold]
            
            st.header("Analysis Dashboard")
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "✅ Known Taxa (Transformer)", "🔍 Novel Taxa (HDBSCAN)"])
            
            with tab1:
                st.metric("Known Sequences", len(known_df))
                st.metric("Unknown Sequences", len(unknown_df))

            with tab2:
                if not known_df.empty:
                    st.bar_chart(known_df['predicted_taxon'].value_counts())
                    st.dataframe(known_df[['id', 'predicted_taxon', 'confidence']])
                    st.info("💡 **Explainable AI:** Our Transformer model uses a sophisticated attention mechanism, allowing researchers to investigate which parts of the DNA sequence were most influential for classification. This is a key area for future development.")
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

