import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
import warnings

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

aploc_url = "https://github.com/23CSE362-edge-computing-2025-26-odd/capstone-project-03_cloud9/raw/refs/heads/main/src/dataset/aploc.csv"
nodeloc_url = "https://github.com/23CSE362-edge-computing-2025-26-odd/capstone-project-03_cloud9/raw/refs/heads/main/src/dataset/nodeloc_ss.csv"

def load_csv_safely(url, usecols=None, header=None, dtype=None, low_memory=False):
    try:
        df = pd.read_csv(url, usecols=usecols, header=header, dtype=dtype, low_memory=low_memory)
        print(f"Loaded {url} — Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {url}: {e}. Using fallback data.")
        return None

def normalize_positions(data, scale=10.0):
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.zeros((0, 2))
    min_vals = np.nanmin(data, axis=0)
    max_vals = np.nanmax(data, axis=0)
    ranges = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    norm = (data - min_vals) / ranges * scale
    return np.nan_to_num(norm, nan=0.0)


print("\nLoading gateway positions...")
gateway_df = load_csv_safely(aploc_url, header=None, dtype=float)
if gateway_df is not None and not gateway_df.empty:
    gateway_positions = gateway_df.values
else:
    gateway_positions = np.array([[0,0],[20,0],[20,10],[0,10]])
    print("Using fallback gateway positions.")


print("\nLoading tag positions...")
tag_df_full = load_csv_safely(nodeloc_url, header=None, low_memory=False)

if tag_df_full is not None and not tag_df_full.empty:
    tag_df_numeric = tag_df_full.apply(pd.to_numeric, errors='coerce')
    variances = tag_df_numeric.var(axis=0, skipna=True)
    coord_cols = variances[variances > 1e-6].index.tolist()

    if len(coord_cols) >= 2:
        tag_positions = tag_df_numeric[coord_cols[:2]].dropna().values
        print(f"Detected coordinate columns: {coord_cols[:2]}")
    else:
        numeric_cols = tag_df_numeric.columns[:2]
        tag_positions = tag_df_numeric[numeric_cols].dropna().values

    tag_positions = normalize_positions(tag_positions, scale=10.0)
    
    max_plot = 500
    total_tags = tag_positions.shape[0]
    if total_tags > max_plot:
        print(f"Large dataset ({total_tags} rows). Sampling first {max_plot} for visualization.")
        tag_positions_plot = tag_positions[:max_plot]
    else:
        tag_positions_plot = tag_positions

    unique_tags = [f"Tag_{i}" for i in range(tag_positions_plot.shape[0])]
    print(f"Loaded {total_tags} total tag positions ({len(unique_tags)} shown on plot).")
else:
    tag_positions = np.array([
        [1.2,1.2],[5.1,1.2],[9,1.2],
        [1.2,5.1],[5.1,5.1],[9,5.1],
        [5.1,9],[9,9],[5.7,0.6],[2.4,4.5]
    ])
    tag_positions_plot = tag_positions
    tag_positions = normalize_positions(tag_positions, scale=10.0)
    unique_tags = [f"Tag_{i}" for i in range(len(tag_positions))]
    print("Using fallback tag positions.")

def rssi_from_distance(dist, P0=-59, n=2.0, d0=1.0, shadow_sigma=2.0, rng=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        path_loss = -10 * n * np.log10(np.maximum(dist, d0) / d0)
    shadow = 0.0 if rng is None else rng.normal(0, shadow_sigma, size=dist.shape)
    return P0 + path_loss + shadow


def get_filtered_rssi_vectorized(
    tag_positions,
    gateway_positions,
    steps=10,
    P0=-59.0,
    n=2.0,
    shadow_sigma=2.0,
    Q=0.1,
    R=1.0,
    rng_seed=42,
    export_csv_path=None
):
    rng = np.random.default_rng(rng_seed)
    tag_positions = np.asarray(tag_positions)
    n_tags, n_gws = tag_positions.shape[0], gateway_positions.shape[0]

    # Initialize Kalman states
    x = np.full((n_tags, n_gws), P0, dtype=float) 
    P = np.ones((n_tags, n_gws), dtype=float)      
    
    history = []

    for t in range(steps):
        dists = cdist(tag_positions, gateway_positions)
        meas = rssi_from_distance(dists, P0=P0, n=n, shadow_sigma=shadow_sigma, rng=rng)

        x_pred = x
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (meas - x_pred)
        P = (1 - K) * P_pred

        history.append(x.copy())

    filtered_rssi = x  

    if export_csv_path:
        df = pd.DataFrame(np.array(history).transpose(1,2,0).reshape(n_tags*n_gws, steps))
        df.index = [f"T{i}_G{j}" for i in range(n_tags) for j in range(n_gws)]
        df.to_csv(export_csv_path)
        print(f"RSSI time series exported to {export_csv_path}")

    return filtered_rssi, history


def plot_2d_positions(gateway_positions, tag_positions, unique_tags=None, title="2D Layout", max_tags=300):
    plt.figure(figsize=(12, 10))
    plt.scatter(gateway_positions[:,0], gateway_positions[:,1], marker='*', s=200, c='red', label='Gateways')
    for i, (x, y) in enumerate(gateway_positions):
        plt.annotate(f"GW_{i}", (x, y), xytext=(5,5), textcoords='offset points', fontsize=10, fontweight='bold')

    n_tags = tag_positions.shape[0]
    if n_tags > max_tags:
        idx = np.random.choice(n_tags, max_tags, replace=False)
        plot_tags = tag_positions[idx]
    else:
        plot_tags = tag_positions

    plt.scatter(plot_tags[:,0], plot_tags[:,1], c='blue', s=5, alpha=0.4, label=f'Tags (n={n_tags})')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\nPlotting 2D positions...")
    plot_2d_positions(gateway_positions, tag_positions_plot, unique_tags)

    print("\nRunning RSSI simulation and filtering for large dataset...")
    filtered_rssi, hist = get_filtered_rssi_vectorized(
        tag_positions,
        gateway_positions,
        steps=20,
        export_csv_path="rssi_timeseries.csv"
    )

    print("\nFinal Filtered RSSI Map (dBm):")
    print(np.round(filtered_rssi, 2))
