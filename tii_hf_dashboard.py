import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="TII Hugging Face Statistics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for TII branding
st.markdown("""
<style>
    .main {
        background-color: #f5f5f9;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1e1b4b;
        font-weight: bold;
    }
    h2, h3 {
        color: #4c1d95;
    }
    .stButton>button {
        background-color: #7c3aed;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #6d28d9;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🚀 TII Hugging Face Live Statistics")
st.markdown("**Real-time analytics from the Technology Innovation Institute's AI models**")
st.markdown("---")

# Sidebar
with st.sidebar:
    # st.image("tii.png", use_container_width=True)
    st.image("tii.png", width=50)
    st.markdown("### Dashboard Controls")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if st.button("🔄 Refresh Now"):
        st.rerun()
    st.markdown("---")
    st.markdown("### About")
    st.info("This dashboard displays live statistics from TII's Hugging Face organization, including model downloads, trending models, and family breakdowns.")

# Function to fetch TII models from Hugging Face API
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_tii_models():
    """Fetch all models from TII organization"""
    try:
        url = "https://huggingface.co/api/models?author=tiiuae&limit=200"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models = response.json()
        return models
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

# Function to categorize models by family
def categorize_model(model_id):
    """Categorize model into Falcon family"""
    model_name = model_id.lower()
    # Check for Falcon H1-R (Reasoning) first
    if "falcon-h1r" in model_name or "falcon_h1r" in model_name or "h1-reasoning" in model_name or "h1r" in model_name:
        return "Falcon H1-R"
    elif "falcon-h1" in model_name or "falcon_h1" in model_name:
        return "Falcon H1"
    elif "falcon3" in model_name:
        return "Falcon 3"
    elif "falcon-11b" in model_name or "falcon-2" in model_name or "falcon2" in model_name:
        return "Falcon 2"
    elif "falcon-arabic" in model_name:
        return "Falcon Arabic"
    elif "falcon-mamba" in model_name or "mamba" in model_name:
        return "Falcon Mamba"
    elif "falcon-e" in model_name or "edge" in model_name:
        return "Falcon Edge"
    # Original Falcon models - check specific sizes
    elif "falcon-180b" in model_name or "falcon-7b" in model_name or "falcon-40b" in model_name:
        return "Falcon"
    elif "falcon-refinedweb" in model_name or "falcon-rw" in model_name:
        return "Falcon RefinedWeb"
    elif "viscon" in model_name:
        return "VisCon"
    elif "visper" in model_name:
        return "ViSpeR"
    elif "dense" in model_name:
        return "Dense Architecture"
    else:
        return "Other"

# Fetch data
with st.spinner("Fetching live data from Hugging Face..."):
    models_data = fetch_tii_models()

if models_data:
    # Process data
    df = pd.DataFrame(models_data)
    
    # Extract relevant fields
    df['model_id'] = df['id']
    df['downloads'] = df['downloads'].fillna(0).astype(int)
    df['likes'] = df['likes'].fillna(0).astype(int)
    df['family'] = df['model_id'].apply(categorize_model)
    
    # Calculate statistics
    total_models = len(df)
    total_downloads = df['downloads'].sum()
    total_likes = df['likes'].sum()
    avg_downloads = int(df['downloads'].mean())
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", f"{total_models:,}")
    with col2:
        st.metric("Total Downloads", f"{total_downloads:,}")
    with col3:
        st.metric("Total Likes", f"{total_likes:,}")
    with col4:
        st.metric("Avg Downloads/Model", f"{avg_downloads:,}")
    
    st.markdown("---")
    
    # Two column layout for charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Top 10 Most Downloaded Models")
        top_models = df.nlargest(10, 'downloads')[['model_id', 'downloads']]
        top_models['model_name'] = top_models['model_id'].str.replace('tiiuae/', '')
        
        fig_top = px.bar(
            top_models,
            x='downloads',
            y='model_name',
            orientation='h',
            color='downloads',
            color_continuous_scale=['#ddd6fe', '#7c3aed'],
            labels={'downloads': 'Downloads', 'model_name': 'Model'},
        )
        fig_top.update_layout(
            showlegend=False,
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Downloads by Model Family")
        family_stats = df.groupby('family')['downloads'].sum().reset_index()
        family_stats = family_stats.sort_values('downloads', ascending=False)
        
        fig_family = px.pie(
            family_stats,
            values='downloads',
            names='family',
            color_discrete_sequence=px.colors.sequential.Purples_r,
        )
        fig_family.update_traces(textposition='inside', textinfo='percent+label')
        fig_family.update_layout(
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_family, use_container_width=True)
    
    st.markdown("---")
    
    # Model family breakdown
    st.subheader("📈 Model Family Breakdown")
    family_breakdown = df.groupby('family').agg({
        'model_id': 'count',
        'downloads': 'sum',
        'likes': 'sum'
    }).reset_index()
    family_breakdown.columns = ['Family', 'Model Count', 'Total Downloads', 'Total Likes']
    family_breakdown = family_breakdown.sort_values('Total Downloads', ascending=False)
    
    st.dataframe(
        family_breakdown,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Family": st.column_config.TextColumn("Model Family", width="medium"),
            "Model Count": st.column_config.NumberColumn("Models", format="%d"),
            "Total Downloads": st.column_config.NumberColumn("Downloads", format="localized"),
            "Total Likes": st.column_config.NumberColumn("Likes", format="localized"),
        }
    )
    
    st.markdown("---")
    
    # Full model list (expandable)
    with st.expander("📋 View All Models"):
        display_df = df[['model_id', 'downloads', 'likes', 'family']].copy()
        display_df.columns = ['Model ID', 'Downloads', 'Likes', 'Family']
        display_df = display_df.sort_values('Downloads', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: Hugging Face API")
    
else:
    st.error("Failed to fetch data from Hugging Face. Please try again later.")

# Auto-refresh logic
if auto_refresh:
    time.sleep(30)
    st.rerun()
