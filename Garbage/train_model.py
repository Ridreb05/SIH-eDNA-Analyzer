import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # For saving the model

print("Starting model training process...")

# 1. Load and Parse FASTA
fasta_path = 'training_data.fasta'
sequences = []
for record in SeqIO.parse(fasta_path, "fasta"):
    # Extract the label (e.g., 'Porifera') from the header like 'Porifera_Sponge_Seq1'
    label = record.id.split('_')[0]
    sequences.append({'label': label, 'sequence': str(record.seq)})

df = pd.DataFrame(sequences)
print(f"Loaded {len(df)} sequences from {fasta_path}")

# 2. Feature Extraction (K-mer Frequencies)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
X = vectorizer.fit_transform(df['sequence'])
y = df['label']
print("Sequences converted to k-mer vectors.")

# 3. Train the Supervised Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Evaluate the Model (Optional but good practice)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.2f}")

# 5. Save the Model and Vectorizer
joblib.dump(model, 'taxa_classifier.joblib')
joblib.dump(vectorizer, 'kmer_vectorizer.joblib')
print("Model and vectorizer have been saved to files: 'taxa_classifier.joblib' and 'kmer_vectorizer.joblib'")
print("\nTraining process finished successfully!")