import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# -----------------------------
# Load & Prepare Data
# -----------------------------
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'UnitPrice': lambda x: (df.loc[x.index, 'Quantity'] * x).sum()
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm


# -----------------------------
# Clustering
# -----------------------------
def perform_clustering(rfm, n_clusters=4):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    return rfm, rfm_scaled


# -----------------------------
# Cluster Stability (ARI)
# -----------------------------
def cluster_stability_score(rfm_scaled, n_clusters=4):
    labels_1 = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    ).fit_predict(rfm_scaled)

    labels_2 = KMeans(
        n_clusters=n_clusters, random_state=99, n_init=10
    ).fit_predict(rfm_scaled)

    return adjusted_rand_score(labels_1, labels_2)


# -----------------------------
# Cluster Summary
# -----------------------------
def cluster_summary(rfm):
    numeric_cols = ['Recency', 'Frequency', 'Monetary']
    return rfm.groupby('Cluster')[numeric_cols].mean()



# -----------------------------
# Persona Mapping
# -----------------------------
def assign_personas(rfm):
    summary = rfm.groupby('Cluster').mean()

    # Rank clusters
    summary['Recency_rank'] = summary['Recency'].rank(ascending=False)
    summary['Frequency_rank'] = summary['Frequency'].rank(ascending=False)
    summary['Monetary_rank'] = summary['Monetary'].rank(ascending=False)

    personas = {}

    for cluster, row in summary.iterrows():

        # Highest spenders
        if row['Monetary_rank'] == 1:
            personas[cluster] = "VIP Customers"

        # Most frequent buyers
        elif row['Frequency_rank'] == 1:
            personas[cluster] = "Loyal High-Value Customers"

        # Long time since last purchase
        elif row['Recency_rank'] == 1:
            personas[cluster] = "Dormant Customers"

        # Lowest overall engagement
        else:
            personas[cluster] = "Low-Value Lost Customers"

    rfm['Persona'] = rfm['Cluster'].map(personas)
    return rfm, personas





# -----------------------------
# Strategy Simulation
# -----------------------------
def simulate_strategy(rfm, target_cluster, freq_multiplier):
    baseline = rfm.groupby('Cluster')['Monetary'].sum()

    simulated = rfm.copy()
    mask = simulated['Cluster'] == target_cluster

    efficiency = 0.6 + (0.4 / freq_multiplier)
    simulated.loc[mask, 'Monetary'] *= (1 + (freq_multiplier - 1) * efficiency)

    after = simulated.groupby('Cluster')['Monetary'].sum()

    result = pd.DataFrame({
        'Before': baseline,
        'After': after,
        'Change (%)': ((after - baseline) / baseline) * 100
    })

    return result.round(2)

