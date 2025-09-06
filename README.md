# DeepGene: A Deep Learning Pipeline for Deep-Sea eDNA Discovery

**A Solution for Smart India Hackathon 2025**  
**Problem Statement ID:** ID25042  

DeepGene is an advanced, AI-driven web application designed to tackle the critical challenge of identifying biodiversity from environmental DNA (eDNA) in deep-sea ecosystems. Developed for the Ministry of Earth Sciences (MoES), this tool empowers CMLRE scientists to overcome the limitations of traditional bioinformatics pipelines, enabling rapid, accurate, and insightful analysis of complex eDNA datasets.

---

## 🌊 The Problem: Unseen Life in the Deep

The deep ocean is a vast reservoir of undiscovered life. Traditional methods of identifying species from eDNA fail because they rely on databases that are severely incomplete for deep-sea organisms. This leads to:

- **Misclassification:** Incorrectly identifying species.  
- **Unassigned Reads:** A large portion of data remains unusable.  
- **Slow Analysis:** Computationally expensive and time-consuming.  

These bottlenecks hinder vital conservation efforts and the potential discovery of novel species with biotechnological significance.

---

## ✨ Our Solution: The Hybrid AI Pipeline

DeepGene introduces a novel **Hybrid AI Pipeline** that minimizes reliance on databases and embraces a discovery-oriented approach. The pipeline intelligently separates the problem into two parts:

### 🧠 Known Taxa Identification (Supervised Deep Learning)
A powerful **Convolutional Neural Network (CNN)**, trained on known genetic markers, instantly classifies sequences with high confidence. This serves as the rapid identification engine.

### 🔬 Novel Taxa Discovery (Unsupervised Clustering)
Sequences that the CNN cannot confidently classify are passed to an advanced **HDBSCAN** algorithm. This groups similar "unknown" sequences into clusters, flagging them as potential novel taxa worthy of further investigation.

---

## ⭐ Masterstroke Feature: Explainable AI (XAI)

DeepGene incorporates **DNA Saliency Maps**, a cutting-edge XAI technique. This feature generates a heatmap over a DNA sequence, highlighting the nucleotides the CNN model focused on for its classification. This turns the AI from a "black box" into a transparent and trustworthy scientific tool.

---

## 🛠️ Technology Stack

| Category             | Technologies                                      |
|---------------------|--------------------------------------------------|
| Core Language        | Python 3                                        |
| Web Dashboard        | Streamlit                                       |
| AI & Deep Learning   | TensorFlow (Keras), Scikit-learn, HDBSCAN       |
| Data Processing      | Pandas, NumPy, BioPython                         |
| Visualization        | Plotly, Matplotlib                               |
| Collaboration & Tools| Git, GitHub, VS Code, Conda                       |

---


## 📈 How to Use the Application

Navigate to the App: Select "🚀 The Application" from the sidebar

Upload Data: Click "Upload FASTA file" and select your eDNA data (e.g., final_demo.fasta)

Set Confidence: Adjust the confidence slider to define what the AI should consider "known"

Analyze: Click the "Analyze with XAI" button

Explore Results:

Overview: Summary of known vs. unknown sequences

Known Taxa (XAI): See classification results and click "View Saliency Map" to understand AI reasoning

Novel Taxa (HDBSCAN): Examine clusters of unknown sequences that could represent new species

## 🔮 Future Scope

Cloud Deployment: Deploy on AWS or Streamlit Community Cloud for public access

Scalable Architecture: Re-architect backend with FastAPI and Celery for massive datasets

Model Enhancement: Implement feedback loops for continuous retraining with newly verified taxa

## 👥 The Team

Team Name: DeepGene

Debanik Das

Aditya Sarkar

Suwastik Bhattachraya
