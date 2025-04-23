import streamlit as st
import math
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Weaviate Disk Storage Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8be6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def format_bytes(bytes_value):
    """Format bytes to human-readable format."""
    if bytes_value < 1024:
        return f"{bytes_value:.2f} B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.2f} KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.2f} GB"

def calculate_storage(
    num_objects,
    vector_dimensions,
    avg_object_size_bytes,
    quantization="none",
    searchable_properties_ratio=0.33,
):
    """Calculate estimated storage requirements for Weaviate."""
    # Calculate vector storage (4 bytes per dimension for float32)
    bytes_per_vector = vector_dimensions * 4
    
    if quantization:
        quantization = quantization.lower()
        if quantization == 'sq':  # Scalar Quantization (8-bit)
            bytes_per_vector = vector_dimensions * 1  # 1 byte per dimension
        elif quantization == 'pq':  # Product Quantization
            # Note: In Weaviate, SQ and PQ have similar storage requirements in the simple case
            # Both reduce from 32-bit to 8-bit per dimension
            bytes_per_vector = vector_dimensions * 1  # Approximately 1 byte per dimension
        elif quantization == 'bq':  # Binary Quantization (1-bit)
            bytes_per_vector = math.ceil(vector_dimensions / 8)  # 1 bit per dimension
    
    total_vector_storage_bytes = num_objects * bytes_per_vector
    
    # Calculate object storage
    total_object_storage_bytes = num_objects * avg_object_size_bytes
    
    # Calculate searchable properties storage
    searchable_properties_bytes = total_object_storage_bytes * searchable_properties_ratio
    
    # Total storage
    total_storage_bytes = total_object_storage_bytes + total_vector_storage_bytes + searchable_properties_bytes
    
    return {
        "raw_object_storage_bytes": total_object_storage_bytes,
        "vector_storage_bytes": total_vector_storage_bytes,
        "searchable_properties_bytes": searchable_properties_bytes,
        "total_storage_bytes": total_storage_bytes,
        "total_storage_mb": total_storage_bytes / (1024 * 1024),
        "total_storage_gb": total_storage_bytes / (1024 * 1024 * 1024),
    }

def get_compression_comparison(num_objects, vector_dimensions, avg_object_size_bytes, searchable_properties_ratio=0.33):
    """Get comparison of different compression methods."""
    comparison = {}
    
    for quant_type in ["none", "sq", "pq", "bq"]:
        comp_bytes_per_vector = vector_dimensions * 4  # Default for no compression
        compression_ratio = 1
        
        if quant_type == "sq":
            comp_bytes_per_vector = vector_dimensions * 1  # 1 byte per dimension
            compression_ratio = 4
            display_name = "Scalar Quantization (SQ)"
            description = "Reduces each dimension from 32 bits to 8 bits"
            quality_impact = "~5% loss in retrieval recall"
        elif quant_type == "pq":
            comp_bytes_per_vector = vector_dimensions * 1  # Approximate
            compression_ratio = 4
            display_name = "Product Quantization (PQ)"
            description = "Creates segments stored as 8-bit integers"
            quality_impact = "Varies based on configuration"
        elif quant_type == "bq":
            comp_bytes_per_vector = math.ceil(vector_dimensions / 8)  # 1 bit per dimension
            compression_ratio = 32
            display_name = "Binary Quantization (BQ)"
            description = "Reduces each dimension to a single bit"
            quality_impact = "Significant impact on accuracy"
        else:
            display_name = "No Quantization"
            description = "32-bit floating point vectors"
            quality_impact = "No impact (baseline)"
        
        comp_vector_storage = num_objects * comp_bytes_per_vector
        comp_total_storage = (
            num_objects * avg_object_size_bytes + 
            comp_vector_storage + 
            num_objects * avg_object_size_bytes * searchable_properties_ratio
        )
        
        savings_percent = (1 - 1/compression_ratio) * 100 if compression_ratio > 1 else 0
        
        comparison[quant_type] = {
            "display_name": display_name,
            "description": description,
            "quality_impact": quality_impact,
            "vector_storage_bytes": comp_vector_storage,
            "total_storage_bytes": comp_total_storage,
            "total_storage_mb": comp_total_storage / (1024 * 1024),
            "total_storage_gb": comp_total_storage / (1024 * 1024 * 1024),
            "compression_ratio": compression_ratio,
            "vector_storage_savings_percent": savings_percent
        }
    
    return comparison

def extrapolate_from_sample(
    sample_objects,
    sample_storage_bytes,
    target_objects
):
    """Extrapolate storage requirements from a sample dataset."""
    # Linear extrapolation
    linear_storage = (sample_storage_bytes / sample_objects) * target_objects
    
    # Sublinear extrapolation (using a simple sqrt-based model as an example)
    # This is just an illustrative model - actual sublinear scaling depends on many factors
    sublinear_factor = math.sqrt(target_objects) / math.sqrt(sample_objects)
    sublinear_storage = sample_storage_bytes * sublinear_factor
    
    return {
        "sample_info": {
            "objects": sample_objects,
            "storage_bytes": sample_storage_bytes,
            "storage_per_object": sample_storage_bytes / sample_objects
        },
        "target_objects": target_objects,
        "linear_extrapolation": {
            "storage_bytes": linear_storage,
            "storage_mb": linear_storage / (1024 * 1024),
            "storage_gb": linear_storage / (1024 * 1024 * 1024)
        },
        "sublinear_extrapolation": {
            "storage_bytes": sublinear_storage,
            "storage_mb": sublinear_storage / (1024 * 1024),
            "storage_gb": sublinear_storage / (1024 * 1024 * 1024),
        }
    }

# App Header
st.title("Weaviate Storage Calculator")
st.markdown("""
This app helps you estimate storage requirements for Weaviate vector database based on your data characteristics.
You can either calculate estimates from basic parameters or extrapolate from existing measurements.
""")

# Create tabs
tab1, tab2 = st.tabs(["📊 Parameter-Based Calculation", "📈 Extrapolate from Sample"])

with tab1:
    st.header("Parameter-Based Storage Calculation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Object Parameters")
        
        # Use session state to persist values
        if 'num_objects' not in st.session_state:
            st.session_state.num_objects = 100000
            
        num_objects = st.number_input(
            "Number of Objects", 
            min_value=1,
            value=st.session_state.num_objects,
            step=10000,
            help="Total number of objects to be stored in Weaviate"
        )
        st.session_state.num_objects = num_objects
        
        object_size_unit = st.selectbox(
            "Object Size Unit",
            options=["Bytes", "Kilobytes (KB)", "Megabytes (MB)"],
            index=1,
            help="Unit for specifying object size"
        )
        
        if 'avg_object_size' not in st.session_state:
            st.session_state.avg_object_size = 5.0
            
        avg_object_size = st.number_input(
            f"Average Object Size ({object_size_unit.split()[0]})",
            min_value=0.001,
            value=st.session_state.avg_object_size,
            step=1.0,
            help="Average size of each object (without vector)"
        )
        st.session_state.avg_object_size = avg_object_size
        
        # Convert to bytes based on selected unit
        if object_size_unit == "Kilobytes (KB)":
            avg_object_size_bytes = avg_object_size * 1024
        elif object_size_unit == "Megabytes (MB)":
            avg_object_size_bytes = avg_object_size * 1024 * 1024
        else:
            avg_object_size_bytes = avg_object_size
    
    with col2:
        st.subheader("Vector Parameters")
        
        if 'vector_dimensions' not in st.session_state:
            st.session_state.vector_dimensions = 768
            
        vector_dimensions = st.number_input(
            "Vector Dimensions", 
            min_value=1, 
            value=st.session_state.vector_dimensions,
            step=128,
            help="Dimensionality of the vector embeddings (e.g., 768 for BERT, 1536 for OpenAI's ada-002)"
        )
        st.session_state.vector_dimensions = vector_dimensions
        
        if 'quantization' not in st.session_state:
            st.session_state.quantization = "none"
            
        quantization = st.selectbox(
            "Vector Quantization Method",
            options=[
                "None (32-bit float)",
                "Scalar Quantization (SQ)",
                "Product Quantization (PQ)",
                "Binary Quantization (BQ)"
            ],
            index=0,
            help="Method used to compress vector data"
        )
        
        quant_map = {
            "None (32-bit float)": "none",
            "Scalar Quantization (SQ)": "sq",
            "Product Quantization (PQ)": "pq",
            "Binary Quantization (BQ)": "bq"
        }
        quantization_code = quant_map[quantization]
        st.session_state.quantization = quantization_code
    
    with col3:
        st.subheader("Additional Parameters")
        
        if 'searchable_ratio' not in st.session_state:
            st.session_state.searchable_ratio = 0.33
            
        searchable_ratio = st.slider(
            "Searchable Properties Ratio",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.searchable_ratio,
            step=0.01,
            help="Ratio of index size to object size for searchable properties (Weaviate's recommendation is ~0.33)"
        )
        st.session_state.searchable_ratio = searchable_ratio
        
        st.markdown("---")
        
        bytes_per_vector = vector_dimensions * 4
        if quantization_code == 'sq':
            bytes_per_vector = vector_dimensions * 1
        elif quantization_code == 'pq':
            bytes_per_vector = vector_dimensions * 1
        elif quantization_code == 'bq':
            bytes_per_vector = math.ceil(vector_dimensions / 8)
        
        st.metric(
            "Vector Size", 
            f"{format_bytes(bytes_per_vector)} per vector",
            help="Size of each vector after applying the selected quantization"
        )
        
        total_base_size = num_objects * avg_object_size_bytes
        st.metric(
            "Raw Data Size",
            f"{format_bytes(total_base_size)}",
            help="Total size of all objects without vectors or indices"
        )
    
    # Calculate storage based on parameters
    storage_results = calculate_storage(
        num_objects=num_objects,
        vector_dimensions=vector_dimensions,
        avg_object_size_bytes=avg_object_size_bytes,
        quantization=quantization_code,
        searchable_properties_ratio=searchable_ratio
    )
    
    st.markdown("---")
    
    # Display storage breakdown
    st.subheader("Storage Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Raw Object Storage",
            f"{format_bytes(storage_results['raw_object_storage_bytes'])}",
            help="Total size of all your objects"
        )
    
    with col2:
        st.metric(
            "Vector Storage",
            f"{format_bytes(storage_results['vector_storage_bytes'])}",
            help="Storage required for vector embeddings"
        )
    
    with col3:
        st.metric(
            "Searchable Properties",
            f"{format_bytes(storage_results['searchable_properties_bytes'])}",
            help="Storage required for searchable properties indices"
        )
    
    with col4:
        st.metric(
            "Total Storage",
            f"{format_bytes(storage_results['total_storage_bytes'])}",
            f"{storage_results['total_storage_gb']:.2f} GB",
            help="Total estimated storage requirement"
        )
    
    # Create and display chart for storage breakdown
    storage_data = pd.DataFrame({
        'Component': ['Raw Objects', 'Vector Embeddings', 'Searchable Properties'],
        'Size (bytes)': [
            storage_results['raw_object_storage_bytes'],
            storage_results['vector_storage_bytes'],
            storage_results['searchable_properties_bytes']
        ]
    })
    
    chart = alt.Chart(storage_data).mark_bar().encode(
        x=alt.X('Component:N', title=None),
        y=alt.Y('Size (bytes):Q', title='Storage (bytes)'),
        color=alt.Color('Component:N', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['Component', alt.Tooltip('Size (bytes):Q', format=',')]
    ).properties(
        title='Storage Components Breakdown',
        width=600,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Compression comparison
    st.markdown("---")
    st.subheader("Compression Options Comparison")
    st.markdown("Compare different vector compression methods and their impact on storage requirements")
    
    comparison = get_compression_comparison(
        num_objects=num_objects,
        vector_dimensions=vector_dimensions,
        avg_object_size_bytes=avg_object_size_bytes,
        searchable_properties_ratio=searchable_ratio
    )
    
    # Create data for the chart
    comp_data = []
    for quant_type, data in comparison.items():
        comp_data.append({
            'Method': data['display_name'],
            'Vector Storage (bytes)': data['vector_storage_bytes'],
            'Total Storage (bytes)': data['total_storage_bytes'],
            'Compression Ratio': data['compression_ratio'],
            'Vector Savings %': data['vector_storage_savings_percent']
        })
    
    comp_df = pd.DataFrame(comp_data)
    
    # Add metrics for each compression method
    cols = st.columns(len(comparison))
    
    for i, (quant_type, data) in enumerate(comparison.items()):
        with cols[i]:
            st.markdown(f"**{data['display_name']}**")
            st.markdown(f"*{data['description']}*")
            st.metric(
                "Total Storage",
                f"{format_bytes(data['total_storage_bytes'])}",
                f"{data['compression_ratio']}x compression" if quant_type != "none" else "Baseline"
            )
            st.caption(f"Impact: {data['quality_impact']}")
    
    # Create and display the chart
    comp_chart = alt.Chart(comp_df).mark_bar().encode(
        x=alt.X('Method:N', title=None),
        y=alt.Y('Total Storage (bytes):Q', title='Storage (bytes)'),
        color=alt.Color('Method:N', scale=alt.Scale(scheme='tableau10'), legend=None),
        tooltip=[
            'Method', 
            alt.Tooltip('Total Storage (bytes):Q', format=','),
            alt.Tooltip('Compression Ratio:Q', format='.1f'),
            alt.Tooltip('Vector Savings %:Q', format='.1f')
        ]
    ).properties(
        title='Storage by Compression Method',
        width=600,
        height=300
    )
    
    st.altair_chart(comp_chart, use_container_width=True)
    
    with st.expander("Notes on Compression Methods"):
        st.markdown("""
        ### Compression Options in Weaviate
        
        - **Scalar Quantization (SQ)**: Reduces each dimension from 32 bits to 8 bits, cutting storage by 75% with typically only about 5% loss in retrieval recall.
        
        - **Product Quantization (PQ)**: Reduces vector size by creating segments that are stored as 8-bit integers instead of 32-bit floats.
        
        - **Binary Quantization (BQ)**: Reduces each dimension to a single bit, providing a 1:32 compression rate (best for high-dimensional vectors).
        
        The optimal choice depends on your specific requirements regarding storage constraints and retrieval accuracy.
        """)

with tab2:
    st.header("Extrapolate from Sample Data")
    st.markdown("""
    If you already have a sample dataset loaded in Weaviate, you can use it to predict storage 
    requirements for larger datasets. This approach is often more accurate than parameter-based calculations.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Dataset Information")
        
        if 'sample_objects' not in st.session_state:
            st.session_state.sample_objects = 10000
            
        sample_objects = st.number_input(
            "Objects in Sample", 
            min_value=1,
            value=st.session_state.sample_objects,
            step=1000,
            help="Number of objects in your sample dataset"
        )
        st.session_state.sample_objects = sample_objects
        
        sample_size_unit = st.selectbox(
            "Sample Storage Unit",
            options=["Bytes", "Kilobytes (KB)", "Megabytes (MB)", "Gigabytes (GB)"],
            index=2,
            help="Unit for specifying sample storage size"
        )
        
        if 'sample_storage' not in st.session_state:
            st.session_state.sample_storage = 500.0
            
        sample_storage = st.number_input(
            f"Sample Storage Size ({sample_size_unit.split()[0]})",
            min_value=0.001,
            value=st.session_state.sample_storage,
            step=10.0,
            help="Total storage used by your sample dataset in Weaviate"
        )
        st.session_state.sample_storage = sample_storage
        
        # Convert to bytes based on selected unit
        if sample_size_unit == "Bytes":
            sample_storage_bytes = sample_storage
        elif sample_size_unit == "Kilobytes (KB)":
            sample_storage_bytes = sample_storage * 1024
        elif sample_size_unit == "Megabytes (MB)":
            sample_storage_bytes = sample_storage * 1024 * 1024
        else:  # GB
            sample_storage_bytes = sample_storage * 1024 * 1024 * 1024
    
    with col2:
        st.subheader("Target Dataset Size")
        
        if 'target_objects' not in st.session_state:
            st.session_state.target_objects = 1000000
            
        target_objects = st.number_input(
            "Target Number of Objects", 
            min_value=1,
            value=st.session_state.target_objects,
            step=100000,
            help="Number of objects in your target dataset"
        )
        st.session_state.target_objects = target_objects
        
        # Calculate and display storage per object
        storage_per_object = sample_storage_bytes / sample_objects
        st.metric(
            "Storage per Object",
            f"{format_bytes(storage_per_object)}",
            help="Average storage used by each object in your sample dataset"
        )
        
        # Show sample vs target ratio
        scale_factor = target_objects / sample_objects
        st.metric(
            "Scale Factor",
            f"{scale_factor:.2f}x",
            help="How much larger your target dataset is compared to your sample"
        )
    
    # Calculate extrapolation and show results
    extrapolation_results = extrapolate_from_sample(
        sample_objects=sample_objects,
        sample_storage_bytes=sample_storage_bytes,
        target_objects=target_objects
    )
    
    st.markdown("---")
    
    # Display extrapolation results
    st.subheader("Extrapolation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Linear Extrapolation",
            f"{format_bytes(extrapolation_results['linear_extrapolation']['storage_bytes'])}",
            f"{extrapolation_results['linear_extrapolation']['storage_gb']:.2f} GB",
            help="Estimation assuming storage grows in direct proportion to object count (conservative estimate)"
        )
        
        st.markdown("""
        **Linear extrapolation** assumes storage grows in direct proportion to object count. 
        This provides a conservative estimate with buffer for planning purposes.
        """)
    
    with col2:
        st.metric(
            "Sublinear Estimation (Example)",
            f"{format_bytes(extrapolation_results['sublinear_extrapolation']['storage_bytes'])}",
            f"{extrapolation_results['sublinear_extrapolation']['storage_gb']:.2f} GB",
            help="An illustrative sublinear model (actual behavior may vary)"
        )
        
        st.markdown("""
        **Sublinear estimation** reflects that Weaviate's storage often scales better than linearly
        as data size increases. This example uses a simple square root model for illustration.
        """)
    
    # Create data for extrapolation chart
    extrapolation_data = pd.DataFrame({
        'Objects': [
            sample_objects,
            target_objects,
            target_objects
        ],
        'Storage (bytes)': [
            sample_storage_bytes,
            extrapolation_results['linear_extrapolation']['storage_bytes'],
            extrapolation_results['sublinear_extrapolation']['storage_bytes']
        ],
        'Type': [
            'Sample',
            'Linear Projection',
            'Sublinear Projection'
        ]
    })
    
    # Create and display the chart
    domain = ['Sample', 'Linear Projection', 'Sublinear Projection']
    range_ = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    extrap_chart = alt.Chart(extrapolation_data).mark_circle(size=100).encode(
        x=alt.X('Objects:Q', scale=alt.Scale(type='log'), title='Number of Objects (log scale)'),
        y=alt.Y('Storage (bytes):Q', title='Storage (bytes)'),
        color=alt.Color('Type:N', scale=alt.Scale(domain=domain, range=range_)),
        tooltip=['Type', alt.Tooltip('Objects:Q', format=','), alt.Tooltip('Storage (bytes):Q', format=',')]
    ).properties(
        title='Storage Extrapolation',
        width=600,
        height=400
    )
    
    # Add connecting lines
    line_data = pd.DataFrame({
        'Objects': [sample_objects, target_objects],
        'Linear (bytes)': [sample_storage_bytes, extrapolation_results['linear_extrapolation']['storage_bytes']],
        'Sublinear (bytes)': [sample_storage_bytes, extrapolation_results['sublinear_extrapolation']['storage_bytes']]
    })
    
    line_chart1 = alt.Chart(line_data).mark_line(color='#ff7f0e').encode(
        x='Objects:Q',
        y='Linear (bytes):Q'
    )
    
    line_chart2 = alt.Chart(line_data).mark_line(color='#2ca02c').encode(
        x='Objects:Q',
        y='Sublinear (bytes):Q'
    )
    
    combined_chart = extrap_chart + line_chart1 + line_chart2
    
    st.altair_chart(combined_chart, use_container_width=True)
    
    with st.expander("Notes on Extrapolation"):
        st.markdown("""
        ### About Storage Extrapolation
        
        #### Linear Extrapolation
        Linear extrapolation assumes that storage requirements grow in direct proportion to the number of objects. 
        This is a straightforward calculation: if 100K objects use 5GB, then 1M objects would use 50GB.
        
        While this approach is simple, it provides a conservative estimate with some buffer for planning, 
        which is often useful for infrastructure provisioning.
        
        #### Sublinear Scaling
        In practice, Weaviate's storage often scales sublinearly, meaning that as your data size increases, 
        the storage requirements don't increase at the same rate - they grow more slowly than a direct proportion 
        would suggest.
        
        This happens because:
        - Index structures become more efficient at scale
        - Compression techniques work better with larger datasets
        - Overhead becomes proportionally smaller
        
        The sublinear model shown here is illustrative only and uses a simple square root-based scaling factor.
        Actual sublinear behavior depends on many factors including data distribution, index configuration, 
        and Weaviate version.
        
        #### Best Practice
        The recommended approach is to:
        1. Start with a representative sample (~100K objects or more)
        2. Measure actual storage usage
        3. Use linear extrapolation for conservative planning
        4. Monitor actual usage as you scale
        """)

st.markdown("---")
st.caption("""
© 2025 Weaviate Storage Calculator | This is a simplified estimation tool and actual storage requirements may vary.
For production deployments, please consult Weaviate documentation and conduct appropriate testing.
""")