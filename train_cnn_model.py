import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib  # <-- ADD THIS LINE

print("Starting Deep Learning (CNN) model training process...")

# --- 1. Load and Prepare Data ---
fasta_path = 'training_data.fasta'
sequences = []
for record in SeqIO.parse(fasta_path, "fasta"):
    label = record.id.split('_')[0]
    sequences.append({'label': label, 'sequence': str(record.seq)})
df = pd.DataFrame(sequences)
print(f"Loaded {len(df)} sequences.")

# --- 2. One-Hot Encode DNA Sequences ---
def one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    encoded_seq = np.array([mapping.get(base, [0,0,0,0]) for base in sequence.upper()])
    return encoded_seq

max_len = df['sequence'].str.len().max()
X = np.array([np.pad(one_hot_encode(seq), ((0, max_len - len(seq)), (0,0)), 'constant') for seq in df['sequence']])

# --- 3. Encode Labels ---
label_encoder = LabelEncoder()
y_integer = label_encoder.fit_transform(df['label'])
y = to_categorical(y_integer)
print("Data has been one-hot encoded.")

# --- 4. Build the CNN Model ---
model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. Train the Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nStarting model training...")
model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.1, verbose=0)
print("Model training complete.")

# --- 6. Evaluate and Save ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Deep Learning Model Accuracy: {accuracy*100:.2f}%")

model.save('taxa_cnn_model.keras')
joblib.dump(label_encoder, 'label_encoder.joblib') # This line will now work
print("CNN model and label encoder saved successfully.")