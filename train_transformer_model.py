import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding
from tensorflow.keras.utils import to_categorical

print("Starting Advanced AI (Transformer) model training process...")

fasta_path = 'training_data.fasta'
sequences = [{'label': record.id.split('_')[0], 'sequence': str(record.seq)} for record in SeqIO.parse(fasta_path, "fasta")]
df = pd.DataFrame(sequences)
print(f"Loaded {len(df)} sequences.")


def dna_to_integers(sequence):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4} 
    return [mapping.get(base, 0) for base in sequence.upper()]

max_len = df['sequence'].str.len().max()
X = tf.keras.preprocessing.sequence.pad_sequences(
    [dna_to_integers(seq) for seq in df['sequence']], maxlen=max_len, padding='post'
)

label_encoder = LabelEncoder()
y_integer = label_encoder.fit_transform(df['label'])
y = to_categorical(y_integer)
print("Data has been integer encoded.")

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

embed_dim = 32  
num_heads = 2  
ff_dim = 32  #
inputs = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=5, output_dim=embed_dim) 
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
outputs = Dense(y.shape[1], activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nStarting Transformer model training...")
model.fit(X_train, y_train, epochs=30, batch_size=4, validation_split=0.1, verbose=0)
print("Model training complete.")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Transformer Model Accuracy: {accuracy*100:.2f}%")

model.save('taxa_transformer_model.keras')
joblib.dump(label_encoder, 'label_encoder.joblib')
print("Transformer model and label encoder saved successfully.")
