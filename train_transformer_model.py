import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import glob
import random

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding
from tensorflow.keras.utils import to_categorical

print("Starting State-of-the-Art AI (Stacked Transformer) model training process...")

# --- Constants ---
MAX_SEQUENCE_LENGTH = 2000
JUNK_LABELS = ['PREDICTED:', 'TPA_asm:', 'TPA_inf:']
AUGMENTATION_FACTOR = 5 # Create 5 new sequences for each original sequence

# --- NEW: Data Augmentation Function ---
def augment_sequence(sequence, mutation_rate=0.02):
    """
    Creates a new sequence with random point mutations.
    """
    sequence = list(sequence)
    bases = ['A', 'C', 'G', 'T']
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            original_base = sequence[i]
            new_base = random.choice([b for b in bases if b != original_base])
            sequence[i] = new_base
    return "".join(sequence)

# --- Smart Parsing Function ---
def get_label_from_header(header):
    try:
        return header.split(' ')[1]
    except IndexError:
        return header.split('_')[0]

# --- 1. Load, Filter, and Augment Data ---
fasta_files = glob.glob('*.fasta')
training_files = [f for f in fasta_files if 'demo' not in f and 'sample' not in f and 'training_data' not in f]
if not training_files:
    raise FileNotFoundError("No training FASTA files found.")

print(f"Found training files: {training_files}")

all_sequences = []
for file in training_files:
    for record in SeqIO.parse(file, "fasta"):
        if len(record.seq) <= MAX_SEQUENCE_LENGTH:
            label = get_label_from_header(record.description)
            if label not in JUNK_LABELS:
                # Add the original sequence
                all_sequences.append({'label': label, 'sequence': str(record.seq)})
                # Add augmented sequences
                for _ in range(AUGMENTATION_FACTOR):
                    all_sequences.append({'label': label, 'sequence': augment_sequence(str(record.seq))})

if not all_sequences:
    raise ValueError("No valid sequences found after filtering.")

df = pd.DataFrame(all_sequences)
print(f"\nLoaded a total of {len(df)} sequences after filtering and augmentation.")
print("\nUnique labels found for training:")
print(df['label'].value_counts())
print("-" * 30)

# --- 2. Integer Encode DNA Sequences ---
def dna_to_integers(sequence):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return [mapping.get(base, 0) for base in sequence.upper()]

max_len = MAX_SEQUENCE_LENGTH # Use the constant for consistency
X = tf.keras.preprocessing.sequence.pad_sequences(
    [dna_to_integers(seq) for seq in df['sequence']], maxlen=max_len, padding='post'
)

# --- 3. Encode Labels ---
label_encoder = LabelEncoder()
y_integer = label_encoder.fit_transform(df['label'])
y = to_categorical(y_integer)
print("Data has been integer encoded.")

# --- 4. Build the Transformer Block ---
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- 5. Build the Full, Deeper Model ---
embed_dim = 32
num_heads = 2
ff_dim = 32

inputs = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=5, output_dim=embed_dim)
x = embedding_layer(inputs)
# --- NEW: Stacked Transformer Blocks ---
transformer_block_1 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block_2 = TransformerBlock(embed_dim, num_heads, ff_dim) # A second block
x = transformer_block_1(x)
x = transformer_block_2(x) # Pass the output of the first block to the second
# ---
x = GlobalAveragePooling1D()(x)
x = Dropout(0.2)(x) # Increased dropout for a deeper model
x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(y.shape[1], activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- 6. Train the Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nStarting Stacked Transformer model training on augmented data...")
# Added an early stopping callback to prevent overfitting
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=40, batch_size=8, validation_data=(X_test, y_test), callbacks=[callback], verbose=1)
print("Model training complete.")

# --- 7. Evaluate and Save ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Model Accuracy on Test Set: {accuracy*100:.2f}%")

model.save('taxa_transformer_model.keras')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(max_len, 'max_len.joblib') # NEW: Save max_len
print("New, state-of-the-art Transformer model and helper files saved successfully.")

