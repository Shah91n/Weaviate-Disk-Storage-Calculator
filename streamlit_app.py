import streamlit as st
import pandas as pd
import altair as alt
from calculator import WeaviateStorageCalculator, format_bytes

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Weaviate Storage Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="assets/weaviate-logo.png"
)

# --- INITIALIZE CALCULATOR & SESSION STATE ---
calculator = WeaviateStorageCalculator()

def init_session_state():
    """Initialize session state to prevent inputs from clearing."""
    defaults = {
        'num_objects': 1_000_000,
        'avg_object_size': 5.0,
        'object_size_unit': "Kilobytes (KB)",
        'vector_dimensions': 1536,
        'quantization': "None (32-bit float)",
        'filterable_factor': 17.00,
        'searchable_factor': 5.00,
        'sample_objects': 100_000,
        'sample_storage': 5.0,
        'sample_size_unit': "Gigabytes (GB)",
        'target_objects': 10_000_000
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

init_session_state()

# --- STYLING ---
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding: 10px;}
    .stTabs [aria-selected="true"] {background-color: #03a1fc; color: white;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("assets/weaviate-logo.png", width=200)
    st.title("Settings")
    st.markdown("Adjust global parameters for the calculator.")
    
    st.header("Inverted Index Factors")
    st.slider("Filterable Index Factor (%)", 5.0, 70.0, key='filterable_factor', step=0.5, help="Estimated size of the filterable index as a percentage of object *property* storage.")
    st.slider("Searchable Index Factor (%)", 1.0, 30.0, key='searchable_factor', step=0.5, help="Estimated size of the searchable (BM25) index as a percentage of object *property* storage.")
    
    st.markdown("---")
    if st.button("Clear All & Reset", use_container_width=True):
        clear_session_state()

# --- APP HEADER ---
st.title("Weaviate Disk Storage Estimator")
st.markdown("An accurate calculator for Weaviate disk usage, correctly modeling storage for vector quantization.")

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Parameter-Based Calculation", "📈 Extrapolate from Dataset"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data & Object Parameters")
        st.number_input("Number of Objects", min_value=1, key='num_objects')
        st.selectbox("Object Property Size Unit", ["Bytes", "Kilobytes (KB)", "Megabytes (MB)"], key='object_size_unit')
        unit = st.session_state.object_size_unit.split(" ")[0]
        st.number_input(f"Avg. Object Property Size ({unit})", 0.001, key='avg_object_size', step=1.0, help="The average size of an object's properties, EXCLUDING the vector.")

    with col2:
        st.subheader("Vector Parameters")
        st.number_input("Vector Dimensions", 1, key='vector_dimensions', step=128, help="The calculator works with any dimension value.")
        st.selectbox("Compression (Vector Quantization)",
            ["None (32-bit float)", "Scalar quantization (SQ)", "Product quantization (PQ)", "Rotational quantization (RQ)", "Binary quantization (BQ)"],
            key='quantization'
        )

    # --- CALCULATIONS ---
    unit_map = {"Bytes": 1, "Kilobytes (KB)": 1024, "Megabytes (MB)": 1024**2}
    avg_object_size_bytes = st.session_state.avg_object_size * unit_map[st.session_state.object_size_unit]

    quant_map = {
        "None (32-bit float)": "none", "Scalar quantization (SQ)": "sq", "Product quantization (PQ)": "pq",
        "Rotational quantization (RQ)": "rq", "Binary quantization (BQ)": "bq"
    }
    quant_code = quant_map[st.session_state.quantization]

    results = calculator.calculate_storage(
        num_objects=st.session_state.num_objects, vector_dimensions=st.session_state.vector_dimensions,
        avg_object_size_bytes=avg_object_size_bytes, quantization=quant_code,
        filterable_index_factor=st.session_state.filterable_factor / 100, searchable_index_factor=st.session_state.searchable_factor / 100
    )
    
    st.markdown("---")
    st.metric(label="Total Estimated Storage", value=f"{format_bytes(results['total_storage_bytes'])}", label_visibility="collapsed")
    
    st.subheader("Storage Breakdown")
    
    storage_components = {'Object Properties': results['object_properties_storage_bytes'], 'Filterable Index': results['filterable_storage_bytes'], 'Searchable Index': results['searchable_storage_bytes']}
    if quant_code == 'none':
        storage_components['HNSW Index (Uncompressed)'] = results['hnsw_index_storage_bytes']
    else:
        storage_components['Uncompressed Vectors'] = results['uncompressed_vector_storage_bytes']
        storage_components['HNSW Index (Quantized)'] = results['hnsw_index_storage_bytes']

    storage_data = pd.DataFrame([{'Component': name, 'Size (GB)': size / (1024**3)} for name, size in storage_components.items() if size > 0])
    chart = alt.Chart(storage_data).mark_bar().encode(
        x=alt.X('Component:N', title=None, sort=None), y=alt.Y('Size (GB):Q', title='Storage (GB)'),
        color=alt.Color('Component:N', scale=alt.Scale(scheme='tableau10'), legend=None),
        tooltip=['Component', alt.Tooltip('Size (GB):Q', format='.3f')]
    ).properties(title='Storage Components Breakdown', height=350)
    
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(storage_data, use_container_width=True, hide_index=True)

with tab2:
    st.header("Extrapolate from an Existing Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Dataset")
        st.number_input("Number of Objects", 1, key='sample_objects', step=1000)
        st.selectbox("Storage Unit", ["Bytes", "Kilobytes (KB)", "Megabytes (MB)", "Gigabytes (GB)"], key='sample_size_unit')
        unit = st.session_state.sample_size_unit.split(" ")[0]
        st.number_input(f"Current Storage Size ({unit})", 0.001, key='sample_storage', step=10.0)
    with col2:
        st.subheader("Target Growth")
        st.number_input("Target Number of Objects", 1, key='target_objects', step=100000)
    
    unit_map_extrap = {"Bytes": 1, "Kilobytes (KB)": 1024, "Megabytes (MB)": 1024**2, "Gigabytes (GB)": 1024**3}
    sample_bytes = st.session_state.sample_storage * unit_map_extrap[st.session_state.sample_size_unit]
    extrap_results = calculator.extrapolate_from_sample(sample_objects=st.session_state.sample_objects, sample_storage_bytes=sample_bytes, target_objects=st.session_state.target_objects)
    
    st.metric("Storage per Object (Current)", f"{format_bytes(extrap_results['storage_per_object'])}", help="Average disk space used by each object in your current dataset.")
    st.markdown("---")
    st.subheader("Extrapolation Results")
    res1, res2 = st.columns(2)
    with res1:
        st.metric("Linear Projection", f"{extrap_results['linear_extrapolation_gb']:.2f} GB")
    with res2:
        st.metric("Conservative Projection", f"{extrap_results['conservative_extrapolation_gb']:.2f} GB")

st.markdown("---")
st.caption("This calculator provides estimates based on Weaviate documentation and dataset analysis. Actual storage may vary.")
st.caption(
    'Created by [Mohamed Shahin](https://github.com/Shah91n). '
    'Source code on [GitHub](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)'
)
