import streamlit as st
from segmentation import (
    load_and_prepare_data,
    perform_clustering,
    cluster_stability_score,
    cluster_summary,
    assign_personas,
    simulate_strategy
)

st.set_page_config(
    page_title="Customer Segmentation Tool",
    layout="wide"
)

st.title("Customer Segmentation & Business Decision Support Tool")

st.markdown("""
This application segments customers based on historical purchasing behavior
and enables business teams to simulate the impact of targeted engagement strategies
before deploying them in the real world.
""")

# Load and process data
rfm = load_and_prepare_data("data/online_retail.csv")
rfm, rfm_scaled = perform_clustering(rfm)
ari_score = cluster_stability_score(rfm_scaled)
rfm, personas = assign_personas(rfm)

# Sidebar controls
st.sidebar.header("Strategy Simulation Controls")

cluster_id = st.sidebar.selectbox(
    "Select Customer Segment",
    sorted(rfm['Cluster'].unique())
)

freq_increase = st.sidebar.slider(
    "Assumed Increase in Engagement for Selected Segment (%)",
    0, 50, 20
)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Segment Overview (Historical RFM Averages)")
    st.dataframe(cluster_summary(rfm))

    st.subheader("🧩 Cluster Personas")
    for k, v in personas.items():
        st.write(f"Cluster {k}: **{v}**")

with col2:
    st.subheader("📈 Projected Revenue Impact (What-If Analysis)")
    simulated = simulate_strategy(
        rfm,
        cluster_id,
        1 + freq_increase / 100
    )
    st.dataframe(simulated.style.format({
    'Before': '₹{:,.0f}',
    'After': '₹{:,.0f}',
    'Change (%)': '{:.2f}%'
}))

st.markdown("---")

st.info(
    f"🔍 **Cluster Stability (ARI Score):** {ari_score:.3f} "
    "(Higher values indicate more stable and reliable segmentation)"
)

st.warning("""
⚠️ This tool is designed to support business decision-making.
All simulated results are indicative and should be reviewed with
human judgment, domain knowledge, and ethical considerations.
""")
