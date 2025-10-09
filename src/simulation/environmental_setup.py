import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
from matplotlib.animation import FuncAnimation
import json
import warnings

# Suppress warnings (after fixes, they shouldn't appear, but safety net)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# GitHub URLs (replace with local paths if needed, e.g., r'C:\path\to\file.csv')
aploc_url = "https://github.com/23CSE362-edge-computing-2025-26-odd/capstone-project-03_cloud9/raw/refs/heads/main/src/dataset/aploc.csv"
nodeloc_url = "https://github.com/23CSE362-edge-computing-2025-26-odd/capstone-project-03_cloud9/raw/refs/heads/main/src/dataset/nodeloc_ss.csv"

def load_csv_safely(url, usecols=None, header=None, dtype=None, low_memory=False):
    """Safely load CSV, handling mixed types and errors."""
    try:
        df = pd.read_csv(url, usecols=usecols, header=header, dtype=dtype, low_memory=low_memory)
        print(f"Successfully loaded {url}. Shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        return df
    except Exception as e:
        print(f"Error loading {url}: {e}. Using fallback data.")
        return None

def normalize_positions(data, scale=10.0):
    """Normalize positions to [0, scale], handling division by zero and NaNs."""
    data = np.asarray(data)
    if data.size == 0:
        print("Warning: Empty data for normalization. Returning zeros.")
        return np.zeros((0, data.shape[1]) if len(data.shape) > 1 else (0,))

    with np.errstate(divide='ignore', invalid='ignore'):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        range_vals = max_vals - min_vals

        normalized = np.zeros_like(data, dtype=float)
        for i in range(data.shape[1]):
            if range_vals[i] == 0 or np.isnan(range_vals[i]):
                normalized[:, i] = 0.0  # Or min_vals[i]: Set to constant
                print(f"Warning: Constant/NaN range in column {i} (range={range_vals[i]}). Set to 0.")
            else:
                normalized[:, i] = (data[:, i] - min_vals[i]) / range_vals[i] * scale

        # Replace any remaining NaN/inf
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=scale, neginf=0.0)
        return normalized

# Load gateway positions (only cols 0-1, force float)
print("Loading gateway positions...")
gateway_df = load_csv_safely(aploc_url, usecols=[0, 1], header=None, dtype={0: float, 1: float}, low_memory=False)
if gateway_df is not None and not gateway_df.empty:
    gateway_positions = gateway_df.values
    # Normalize if needed
    # gateway_positions = normalize_positions(gateway_positions, scale=10.0)
    print("Gateway positions:\n", gateway_positions)
else:
    gateway_positions = np.array([[0, 0], [20, 0], [20, 10], [0, 10]])
    print("Using fallback gateway positions.")

# Load tag positions (only cols 0-1 for initial positions, force float)
print("\nLoading tag positions...")
tag_df_raw = load_csv_safely(nodeloc_url, usecols=[0, 1], header=None, dtype={0: float, 1: float}, low_memory=False)
last_positions = np.empty((0, 2))
tag_trajectories = {}
unique_tags = []

if tag_df_raw is not None and not tag_df_raw.empty:
    tag_positions = tag_df_raw.values
    print("Raw tag positions shape:", tag_positions.shape)
    
    # Normalize
    last_positions = normalize_positions(tag_positions, scale=10.0)
    print("Normalized tag positions shape:", last_positions.shape)
    
    # For unique tags: Load full CSV with str dtype to handle mixed types, then parse
    print("\nLoading full tag dataset for unique tags...")
    full_tag_df = load_csv_safely(nodeloc_url, dtype=str, low_memory=False)
    if full_tag_df is not None and not full_tag_df.empty:
        print("Full columns:", full_tag_df.columns.tolist())
        print("Full shape:", full_tag_df.shape)
        
        # Assume col 0: Node ID (str or num), cols 1-2: X,Y (parse to float)
        try:
            full_tag_df['Node'] = full_tag_df.iloc[:, 0].astype(str).str.strip()  # Clean Node ID
            unique_tags = full_tag_df['Node'].unique()
            print(f"Unique tags found: {len(unique_tags)}")
            
            # Group by tag, get positions (limit to 10 tags for demo)
            for tag_id in unique_tags[:10]:
                tag_data = full_tag_df[full_tag_df['Node'] == tag_id]
                if len(tag_data) > 0:
                    pos_cols = tag_data.iloc[:, 1:3].apply(pd.to_numeric, errors='coerce').dropna()
                    if len(pos_cols) > 0:
                        positions = pos_cols.values
                        tag_trajectories[tag_id] = normalize_positions(positions, scale=10.0)
                        print(f"Tag {tag_id}: {len(positions)} positions")
            
            # Use last positions per tag for 2D plot
            if tag_trajectories:
                last_positions = np.vstack([traj[-1] for traj in tag_trajectories.values()])
            unique_tags = list(tag_trajectories.keys())
        except Exception as e:
            print(f"Error processing unique tags: {e}. Using all positions as unique tags.")
            unique_tags = range(len(last_positions))
else:
    # Fallback sample positions
    sample_pos_str = "1.2,1.2 5.1,1.2 9,1.2 1.2,5.1 5.1,5.1 9,5.1 5.1,9 9,9 5.7,0.6 2.4,4.5"
    sample_positions = [list(map(float, pair.split(','))) for pair in sample_pos_str.split()]
    last_positions = normalize_positions(np.array(sample_positions), scale=10.0)
    unique_tags = [f"Tag_{i}" for i in range(len(last_positions))]
    print("Using fallback normalized tag positions.")

if len(last_positions) == 0:
    last_positions = np.array([[2, 3], [5, 8], [10, 5]])  # Minimal fallback
    unique_tags = ["Tag1", "Tag2", "Tag3"]
    print("Using minimal fallback positions.")

print(f"\nFinal unique tags/positions: {len(unique_tags)}")

# Simulation functions (with error handling)
def rssi_from_distance(dist, P0=-59, n=2.0, d0=1.0, shadow_sigma=2.0, rng=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        path_loss = -10 * n * np.log10(np.maximum(dist, d0) / d0)
    shadow = 0 if rng is None else rng.normal(0, shadow_sigma, size=dist.shape)
    return P0 + path_loss + shadow

def make_kalman_filter(x0=-60, Q=0.1, R=1.0):
    try:
        kf = KalmanFilter(dim_x=1, dim_z=1)  # Ensure integers
        kf.x = np.array([[x0]], dtype=float)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.P = np.array([[1.0]])
        kf.R = np.array([[float(R)]])
        kf.Q = np.array([[float(Q)]])
        return kf
    except Exception as e:
        print(f"Error creating KalmanFilter: {e}. Using simple filter.")
        return None

class MovingAverage:
    def __init__(self, window=5):
        self.window = window
        self.values = []

    def update(self, val):
        self.values.append(float(val))
        if len(self.values) > self.window:
            self.values.pop(0)
        return np.mean(self.values)

def simulate_tags(tag_positions, steps=20, method="kalman", rng_seed=42, export=None, real_trajectories=None):  # Reduced steps for speed
    try:
        rng = np.random.default_rng(rng_seed)
        n_tags = tag_positions.shape[0]
        n_gws = gateway_positions.shape[0]
        print(f"Simulating {n_tags} tags, {n_gws} gateways, {steps} steps...")

        if method == "kalman":
            filters = {}
            for i in range(n_tags):
                for j in range(n_gws):
                    kf = make_kalman_filter()
                    if kf is None:
                        print("Kalman filter failed; falling back to Moving Average.")
                        method = "ma"  # Fallback
                        break
                    filters[(i, j)] = kf
            if method != "kalman":
                filters = {(i, j): MovingAverage(window=5) for i in range(n_tags) for j in range(n_gws)}
        else:
            filters = {(i, j): MovingAverage(window=5) for i in range(n_tags) for j in range(n_gws)}

        rssi_hist = {key: [] for key in filters}

        if real_trajectories:
            tag_traj = real_trajectories
            steps = len(tag_traj)
        else:
            tag_traj = [tag_positions.copy() for _ in range(steps)]

        for t in range(steps):
            dists = cdist(tag_traj[t], gateway_positions)
            rssi_vals = rssi_from_distance(dists, rng=rng)
            for i in range(n_tags):
                for j in range(n_gws):
                    meas = rssi_vals[i, j]
                    f = filters[(i, j)]
                    if method == "kalman":
                        f.predict()
                        f.update(meas)
                        val = f.x[0, 0]
                    else:
                        val = f.update(meas)
                    rssi_hist[(i, j)].append(val)

        filtered_map = np.array([
            [rssi_hist[(i, j)][-1] for j in range(n_gws)]
            for i in range(n_tags)
        ])

        if export == "csv":
            df = pd.DataFrame({
                f"T{i}_G{j}": rssi_hist[(i, j)]
                for i in range(n_tags) for j in range(n_gws)
            })
            df.to_csv("rssi_timeseries.csv", index_label="Step")
            print("Exported RSSI to rssi_timeseries.csv")

        return filtered_map, tag_traj, rssi_hist
    except Exception as e:
        print(f"Simulation error: {e}")
        return None, None, None

def plot_2d_positions(gateway_positions, tag_positions, unique_tags, title="2D Visualization of Unique Tags and Gateways"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(gateway_positions[:, 0], gateway_positions[:, 1], marker='*', s=200, c='red', label='Gateways')
    scatter = ax.scatter(tag_positions[:, 0], tag_positions[:, 1], c='blue', s=80, label=f'Unique Tags (n={len(unique_tags)})', alpha=0.7)
    # Label tags if few
    if len(unique_tags) <= 10:
        for i, tag in enumerate(unique_tags):
            ax.annotate(tag, (tag_positions[i, 0], tag_positions[i, 1]), xytext=(5, 5), textcoords='offset points')
    ax.set_xlim(-1, max(np.max(gateway_positions[:, 0]), np.max(tag_positions[:, 0])) + 1)
    ax.set_ylim(-1, max(np.max(gateway_positions[:, 1]), np.max(tag_positions[:, 1])) + 1)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def animate_simulation(tag_traj, gateway_positions):
    if len(tag_traj) == 0:
        print("No trajectory for animation.")
        return
    if not isinstance(tag_traj, list):
        tag_traj = [tag_traj] * 5  # Minimal repeat
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(gateway_positions[:, 0], gateway_positions[:, 1], marker='*', s=200, c='red', label='Gateways')
    scat = ax.scatter([], [], c='blue', s=80, label='Tags')

    def init():
        max_x = max(np.max(gateway_positions[:, 0]), np.max(tag_traj[0][:, 0])) + 1
        max_y = max(np.max(gateway_positions[:, 1]), np.max(tag_traj[0][:, 1])) + 1
        ax.set_xlim(-1, max_x)
        ax.set_ylim(-1, max_y)
        ax.set_title("Tag Positions Over Time")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return scat,

    def update(frame):
        scat.set_offsets(tag_traj[frame])
        ax.set_title(f"Tag Positions (step {frame})")
        return scat,

    ani = FuncAnimation(fig, update, frames=min(10, len(tag_traj)), init_func=init, blit=True, interval=500, repeat=True)
    plt.show()

if __name__ == "__main__":
    # 2D plot
    plot_2d_positions(gateway_positions, last_positions, unique_tags)

    # Simulation (limit tags if many)
    if len(last_positions) > 10:
        last_positions_limited = last_positions[:10]
        print("Limiting to 10 tags for simulation.")
    else:
        last_positions_limited = last_positions
    
    real_trajectories = None  # Use tag_trajectories if you want dynamic paths (e.g., interpolate to steps)
    filt_rssi, traj, hist = simulate_tags(last_positions_limited, steps=20, method="kalman", export="csv", real_trajectories=real_trajectories)
    if filt_rssi is not None:
        print("Final Filtered RSSI Map:\n", np.round(filt_rssi, 2))
    else:
        print("Simulation failed; skipping RSSI output.")

    # Animate simulation
    animate_simulation(traj, gateway_positions)
    print("Script completed successfully!")
