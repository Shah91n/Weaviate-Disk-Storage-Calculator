# Weaviate Disk Storage Calculator

[![Weaviate](https://img.shields.io/static/v1?label=for&message=Weaviate%20%E2%9D%A4&color=green&style=flat-square)](https://weaviate.io/)
[![GitHub Repo stars](https://img.shields.io/github/stars/Shah91n/Weaviate-Disk-Storage-Calculator?style=social)](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://weaviate-disk-calculator.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)

**Weaviate Disk Storage Calculator** is a tool designed to estimate disk storage requirements for the Weaviate vector database. It provides insights into storage usage based on your data characteristics and allows for extrapolation from existing datasets. The calculator offers a detailed breakdown of storage components, including objects, vector indexes, and inverted indexes, to help you plan capacity accurately.

<img width="1847" height="772" alt="image" src="https://github.com/user-attachments/assets/51149386-a6e2-41bb-98f0-b0ca2d10cb3b" />

<a href="https://weaviate-disk-calculator.streamlit.app/">
  Visit Weaviate Disk Storage Calculator
</a>

## Features

### Comprehensive Storage Calculation
- Estimate storage requirements based on:
  - Number of objects and average object size
  - Vector dimensions and quantization methods
  - **Inverted Indexes**: Factors for filterable and searchable property storage, with defaults derived from real-world dataset analysis.
- Supports all Weaviate quantization methods:
  - **None** (32-bit float)
  - **Scalar Quantization (SQ)**
  - **Product Quantization (PQ)**
  - **Residual Quantization (RQ)**
  - **Binary Quantization (BQ)**
- Provides a detailed breakdown of storage components:
  - Raw Object Storage
  - Vector (HNSW) Index Storage
  - Filterable Properties Index Storage
  - Searchable Properties (BM25) Index Storage

### Extrapolation from Sample Data
- Predict storage requirements for larger datasets based on a sample from your existing Weaviate instance.
- Supports both **linear** and **conservative** extrapolation models to account for growth overhead.

### Interactive Visualizations
- Detailed bar charts showing the full breakdown of storage components.
- Side-by-side comparison of different quantization methods.

---

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator.git
    cd Weaviate-Disk-Storage-Calculator
    ```

2.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run streamlit_app.py
    ```

---

## Usage

1.  Open the application in your browser.
2.  Navigate between the tabs:
    -   **📊 Parameter-Based Calculation**: Input your data characteristics to get a detailed storage breakdown. Adjust inverted index factors in the sidebar for fine-tuning.
    -   **📈 Extrapolate from Dataset**: Use metrics from your current dataset to project future storage needs.
3.  Use the "Clear All & Reset" button in the sidebar to reset all fields to their default values.

---

## Notes on Compression Methods

-   **Scalar Quantization (SQ)**: Reduces vector size by ~75% with minimal quality loss.
-   **Product Quantization (PQ)**: Segments vectors and compresses them for significant storage savings.
-   **Residual Quantization (RQ)**: An advanced form of PQ, often providing better accuracy for a similar storage footprint.
-   **Binary Quantization (BQ)**: Compresses vectors to 1 bit per dimension, offering the highest compression at the cost of accuracy.

---

## License

© 2025 Weaviate Storage Calculator. This tool is for estimation purposes only. Actual storage requirements may vary.
