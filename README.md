# DeepGene: A Transformer-Powered Platform for Deep-Sea eDNA Discovery  
### Solution for Smart India Hackathon 2025 - Problem Statement: **ID25042**

DeepGene is an advanced, AI-driven research platform designed to tackle the critical challenge of identifying biodiversity from environmental DNA (eDNA) in deep-sea ecosystems.  

Developed for the **Ministry of Earth Sciences (MoES)**, this tool empowers **CMLRE scientists** to overcome the limitations of traditional bioinformatics, enabling **rapid, accurate, and insightful analysis** of complex eDNA datasets.  

---

## 🌊 The Core Challenge: A Universe of Unknowns  
The deep ocean is Earth's last great frontier, a vast reservoir of undiscovered life. Traditional methods of identifying species from eDNA fail because they rely on genetic databases that are critically incomplete for deep-sea organisms. This leads to:  

- **Misclassification**: Incorrectly identifying species.  
- **Unassigned Reads**: A large portion of valuable data becomes unusable.  
- **Computational Bottlenecks**: The process is slow, expensive, and requires deep bioinformatics expertise.  

This scientific roadblock hinders **vital conservation efforts** and prevents the discovery of **novel species with profound biotechnological potential**.  

---

## ✨ Our Solution: The DeepGene Discovery Engine  

DeepGene introduces a ** Hybrid AI Pipeline** that minimizes reliance on databases and embraces a **discovery-oriented approach**. Our platform intelligently separates the problem into two parallel workflows.  

### 🧠 Known Taxa Identification (Supervised Deep Learning)  
- A powerful **Transformer model** (same architecture powering GPT-like systems) serves as the core classification engine.  
- Unlike CNNs that capture only local patterns, the Transformer captures **long-range context** and **complex relationships** in DNA sequences, ensuring **superior accuracy**.  

### 🔬 Novel Taxa Discovery (Unsupervised Learning)  
- Sequences the Transformer cannot classify are flagged as **"Unknown"** and passed to the Discovery Engine.  
- The engine uses:  
  - **HDBSCAN** → Groups similar unknown sequences into high-confidence clusters (potential novel taxa).  
  - **UMAP** → Creates a stunning, interactive 2D map of this "unknown genetic space," letting scientists visually explore relationships between clusters.  

---

## ⭐ Live NCBI Annotation  
DeepGene bridges AI-driven discovery with biological knowledge.  
- With a single click, scientists can send a **representative sequence** from a new cluster to the **NCBI BLAST server** in real-time.  
- Provides **immediate biological context** → "What is the closest known relative to this potential new species?"  

---

## 🛠️ Technology Stack  

| Category            | Technologies |
|---------------------|--------------|
| **Core Language**   | Python 3 |
| **Web Dashboard**   | Streamlit |
| **AI & Deep Learning** | TensorFlow (Keras) |
| **Machine Learning** | Scikit-learn, HDBSCAN, UMAP-learn |
| **Data Processing** | Pandas, NumPy, BioPython |
| **Visualization**   | Plotly |
| **Collaboration**   | Git, GitHub |
