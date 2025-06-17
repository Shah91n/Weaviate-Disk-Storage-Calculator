# Weaviate Disk Storage Calculator

**Weaviate Disk Storage Calculator** is a designed to estimate storage requirements for the Weaviate vector database. It provides insights into storage needs based on data characteristics and allows for extrapolation from sample datasets.

---

## Features

### Parameter-Based Storage Calculation
- Estimate storage requirements based on:
  - Number of objects
  - Vector dimensions
  - Average object size
  - Vector quantization methods
- Supports different quantization methods:
  - **None** (32-bit float)
  - **Scalar Quantization (SQ)**
  - **Product Quantization (PQ)**
  - **Binary Quantization (BQ)**
- Provides a detailed breakdown of storage components:
  - Raw object storage
  - Vector storage
  - Searchable properties storage

### Compression Comparison
- Compare storage savings and quality impact of different quantization methods
- Visualize storage requirements for each method

### Extrapolation from Sample Data
- Predict storage requirements for larger datasets based on a sample dataset
- Supports both **linear** and **sublinear** extrapolation models

### Interactive Visualizations
- Storage breakdown charts
- Compression comparison charts
- Extrapolation projection charts

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/Weaviate-Disk-Storage-Calculator.git
   cd Weaviate-Disk-Storage-Calculator
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run Streamlit_app.py
   ```

---

## Usage

1. Open the application in your browser
2. Navigate between the tabs:
   - **📊 Parameter-Based Calculation**: Input data characteristics to calculate storage
   - **📈 Extrapolate from current Dataset**: Use your existing dataset to predict storage for growth
3. Explore the visualizations and metrics to understand storage

---

## Notes on Compression Methods

- **Scalar Quantization (SQ)**: Reduces vector size by 75% with minimal quality loss (~5%)
- **Product Quantization (PQ)**: Segments vectors into 8-bit integers for efficient storage
- **Binary Quantization (BQ)**: Compresses vectors to 1 bit per dimension, offering the highest compression but with significant accuracy trade-offs

---

## License

© 2025 Weaviate Storage Calculator. This tool is for estimation purposes only. Actual storage requirements may vary. Please consult Weaviate documentation for production deployments.
