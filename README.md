DeepGene: A Deep Learning Pipeline for Deep-Sea eDNA Discovery
A Solution for Smart India Hackathon 2025 - Problem Statement ID: ID25042
DeepGene is an advanced, AI-driven web application designed to tackle the critical challenge of identifying biodiversity from environmental DNA (eDNA) in deep-sea ecosystems. Developed for the Ministry of Earth Sciences (MoES), this tool empowers CMLRE scientists to overcome the limitations of traditional bioinformatics pipelines, enabling rapid, accurate, and insightful analysis of complex eDNA datasets.

🌊 The Problem: Unseen Life in the Deep
The deep ocean is a vast reservoir of undiscovered life. Traditional methods of identifying species from eDNA fail because they rely on databases that are severely incomplete for deep-sea organisms. This leads to:

Misclassification: Incorrectly identifying species.

Unassigned Reads: A large portion of data is unusable.

Slow Analysis: The process is computationally expensive and time-consuming.

This bottleneck hinders vital conservation efforts and the potential discovery of novel species with biotechnological significance.

✨ Our Solution: The Hybrid AI Pipeline
DeepGene introduces a novel Hybrid AI Pipeline that minimizes reliance on databases and embraces a discovery-oriented approach.

Our pipeline intelligently separates the problem into two parts:

🧠 Known Taxa Identification (Supervised Deep Learning): A powerful Convolutional Neural Network (CNN), trained on known genetic markers, instantly classifies sequences with high confidence. This is our rapid identification engine.

🔬 Novel Taxa Discovery (Unsupervised Clustering): Sequences that the CNN cannot confidently classify are passed to an advanced HDBSCAN algorithm. This algorithm groups similar "unknown" sequences into clusters, flagging them as potential novel taxa worthy of further investigation.

⭐ The Masterstroke Feature: Explainable AI (XAI)
We go beyond just providing an answer. DeepGene incorporates DNA Saliency Maps, a cutting-edge XAI technique. This feature creates a heatmap over a DNA sequence, visualizing the exact nucleotides the CNN model focused on to make its classification. This turns the AI from a "black box" into a transparent and trustworthy scientific tool.

🛠️ Technology Stack
Category

Technologies

Core Language

Python 3

Web Dashboard

Streamlit

AI & Deep Learning

TensorFlow (Keras), Scikit-learn, HDBSCAN

Data Processing

Pandas, NumPy, BioPython

Visualization

Plotly, Matplotlib

Collaboration & Tools

Git, GitHub, VS Code, Conda

📂 Project Structure
SIH-eDNA-Analyzer/
│
├── app.py                  # The main Streamlit web application script.
├── train_cnn_model.py      # Script to train and save the Deep Learning model.
├── training_data.fasta     # Labeled FASTA data used to train the CNN.
├── final_demo.fasta        # A sample FASTA file for testing the application.
│
├── taxa_cnn_model.keras    # The saved, pre-trained CNN model file (generated).
├── label_encoder.joblib    # The saved label encoder for the model (generated).
│
└── README.md               # You are here!

🚀 Getting Started: Setup and Installation
Follow these steps to run the project locally. Using Conda is highly recommended to manage the complex dependencies.

1. Clone the Repository
git clone [https://github.com/Ridreb05/SIH-eDNA-Analyzer.git](https://github.com/Ridreb05/SIH-eDNA-Analyzer.git)
cd SIH-eDNA-Analyzer

2. Set Up the Conda Environment
If you don't have Conda, please install Miniconda.

# Create a new Conda environment from scratch
conda create --name sih_env python=3.11
conda activate sih_env

# Install all required packages from conda-forge
conda install -c conda-forge tensorflow streamlit pandas scikit-learn matplotlib biopython hdbscan plotly

3. Train the Deep Learning Model
This is a one-time step. Run the training script to generate the model files.

python train_cnn_model.py

This will create taxa_cnn_model.keras and label_encoder.joblib in your project folder.

4. Run the Web Application
Launch the Streamlit app.

streamlit run app.py

Your web browser will automatically open with the DeepGene application running.

📈 How to Use the Application
Navigate to the App: Select "🚀 The Application" from the sidebar.

Upload Data: Click the "Upload FASTA file" button and select your eDNA data (e.g., final_demo.fasta).

Set Confidence: Adjust the confidence slider to define what the AI should consider "known."

Analyze: Click the "Analyze with XAI" button.

Explore Results:

Overview: View the summary of known vs. unknown sequences.

Known Taxa (XAI): See the classification results and click "View Saliency Map" to understand the AI's reasoning.

Novel Taxa (HDBSCAN): Examine the clusters of unknown sequences that could represent new species.

🔮 Future Scope
Cloud Deployment: Deploy the application on a cloud service (like AWS or Streamlit Community Cloud) for public access.

Scalable Architecture: For production-level use, re-architect the backend with a FastAPI service and a Celery task queue to handle massive datasets asynchronously.

Model Enhancement: Implement a feedback loop where newly discovered and verified taxa can be used to continuously retrain and improve the CNN model.

👥 The Team
Team Name: DeepGene
Debanik Das

Aditya Sarkar

Suwastik Bhattachraya
