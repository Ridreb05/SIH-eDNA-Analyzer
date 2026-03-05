# DeepGene: Deep-Sea eDNA Taxonomic Classification Platform
**Smart India Hackathon 2025 Submission | Problem Statement: ID25042**

DeepGene is a bioinformatics platform developed for the Ministry of Earth Sciences (MoES) to assist CMLRE researchers in processing environmental DNA (eDNA) from deep-sea ecosystems. The tool provides a hybrid machine learning pipeline to classify known sequences and cluster unassigned reads for novel taxa discovery.

---

## Problem Statement
Standard eDNA taxonomic assignment relies heavily on sequence alignment against reference databases. Because genomic databases are critically incomplete for deep-sea organisms, traditional pipelines often result in:
* **High Unassigned Read Rates:** Significant portions of sequencing data are discarded as "unknown."
* **Taxonomic Misclassification:** Forcing sequence alignments can lead to incorrect species identification.
* **Computational Overhead:** Processing large FASTQ/FASTA files through standard alignment algorithms is resource-intensive and creates bottlenecks.

---

## Pipeline Architecture
DeepGene minimizes reliance on incomplete databases by utilizing a hybrid machine learning approach, separating the workflow into supervised classification and unsupervised clustering.

### 1. Supervised Classification (Known Taxa)
* The primary classification engine utilizes a **Transformer architecture** rather than traditional k-mer matching or CNNs. 
* By treating DNA sequences as contextual data, the model captures long-range genomic dependencies and complex sequence relationships, allowing for higher accuracy in taxonomic assignment based on available training data.

### 2. Unsupervised Discovery (Novel Taxa)
* Sequences that fall below the Transformer's classification confidence threshold are routed to the discovery pipeline.
* **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise):** Groups similar unclassified sequences into high-confidence clusters, isolating potential novel taxa or variants.
* **UMAP (Uniform Manifold Approximation and Projection):** Reduces the dimensionality of the sequence data to generate a 2D scatter plot, allowing researchers to visually analyze the distance and relationships between unknown genetic clusters.

---

## External Integration: NCBI BLAST API
To bridge the gap between AI clustering and biological validation, DeepGene integrates directly with the NCBI BLAST server. 
* Users can select a consensus sequence from any newly generated HDBSCAN cluster and query it against the NCBI database directly from the dashboard.
* This provides immediate homology data, allowing researchers to identify the closest known phylogenetic relatives of unclassified reads.

---

## Technology Stack

| Component | Technologies Used |
| :--- | :--- |
| **Language** | Python 3.x |
| **Frontend/Dashboard** | Streamlit |
| **Deep Learning Engine** | TensorFlow (Keras) |
| **Machine Learning & Clustering** | Scikit-learn, HDBSCAN, UMAP-learn |
| **Bioinformatics Processing** | BioPython, Pandas, NumPy |
| **Data Visualization** | Plotly |
| **Version Control** | Git, GitHub |
