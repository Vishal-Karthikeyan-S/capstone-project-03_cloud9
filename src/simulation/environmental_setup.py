#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
import io
import sys
import urllib.request
import json
from typing import Tuple, List

URL1 = "https://raw.githubusercontent.com/jinyi-yoon/CampusRSSI/refs/heads/main/mediumObs_9.9x9.9/aploc.csv"
URL2 = "https://raw.githubusercontent.com/23CSE362-edge-computing-2025-26-odd/capstone-project-03_cloud9/refs/heads/main/src/dataset/aploc.csv"

def download_aploc(url: str) -> np.ndarray:
    try:
        with urllib.request.urlopen(url) as r:
            data = r.read().decode('utf-8').strip()
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        raise
    tokens = data.split()
    coords = []
    for tok in tokens:
        if "," not in tok:
            continue
        x_str,y_str = tok.split(",",1)
        try:
            x = float(x_str)
            y = float(y_str)
            coords.append([x,y])
        except:
            continue
    return np.array(coords, dtype=float)

# RSSI model,Kalman,MovingAverage
def rssi_from_distance(dist: np.ndarray, P0: float = -59, n: float = 2.0, d0: float = 1.0, shadow_sigma: float = 2.0, rng: np.random.Generator = None):
    
    with np.errstate(divide='ignore'):
        path_loss = -10 * n * np.log10(np.maximum(dist, d0) / d0)
    shadow = 0 if rng is None else rng.normal(0, shadow_sigma, size=dist.shape)
    return P0 + path_loss + shadow

def make_kalman_filter(x0=-60.0, Q=0.1, R=1.0):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[x0]], dtype=float)
    kf.F = np.array([[1.]], dtype=float)
    kf.H = np.array([[1.]], dtype=float)
    kf.P = np.array([[1.]], dtype=float)
    kf.R = np.array([[R]], dtype=float)
    kf.Q = np.array([[Q]], dtype=float)
    return kf

class MovingAverage:
    def __init__(self, window=5):
        self.window = int(window)
        self.values: List[float] = []
    def update(self, val: float) -> float:
        self.values.append(val)
        if len(self.values) > self.window:
            self.values.pop(0)
        return float(np.mean(self.values))

#Simulation
def simulate_tags(
    gateway_positions: np.ndarray,
    start_tags: np.ndarray,
    steps: int = 50,
    method: str = "kalman",
    rng_seed: int = 42,
    export: str = None
):
    rng = np.random.default_rng(rng_seed)
    n_tags = start_tags.shape[0]
    n_gws = gateway_positions.shape[0]

    #Create filters
    if method == "kalman":
        filters = {(i, j): make_kalman_filter() for i in range(n_tags) for j in range(n_gws)}
    else:
        filters = {(i, j): MovingAverage(window=5) for i in range(n_tags) for j in range(n_gws)}

    rssi_hist = {key: [] for key in filters}
    tag_traj = [start_tags.copy() for _ in range(steps)]  # static tags for this example

    for t in range(steps):
        # compute distances tag->gateway(n_tags x n_gws)
        dists = cdist(tag_traj[t], gateway_positions)
        rssi_vals = rssi_from_distance(dists, rng=rng)
        for i in range(n_tags):
            for j in range(n_gws):
                meas = float(rssi_vals[i, j])
                f = filters[(i, j)]
                if method == "kalman":
                    f.predict()
                    f.update(np.array([[meas]]))
                    val = float(f.x[0, 0])
                else:
                    val = f.update(meas)
                rssi_hist[(i, j)].append(val)

    filtered_map = np.array([[rssi_hist[(i, j)][-1] for j in range(n_gws)] for i in range(n_tags)])

    # Export
    if export == "csv":
        cols = []
        data = {}
        for i in range(n_tags):
            for j in range(n_gws):
                colname = f"T{i}_G{j}"
                cols.append(colname)
                data[colname] = rssi_hist[(i, j)]
        df = pd.DataFrame(data)
        df.index.name = "Step"
        outname = "rssi_timeseries.csv"
        df.to_csv(outname)
        print(f"Exported RSSI timeseries to {outname}")

    if export == "json":
        out = {f"T{i}": {f"G{j}": rssi_hist[(i, j)] for j in range(n_gws)} for i in range(n_tags)}
        with open("rssi_timeseries.json", "w") as f:
            json.dump(out, f, indent=2)
        print("Exported RSSI timeseries to rssi_timeseries.json")

    return filtered_map, tag_traj, rssi_hist

#Animation
def animate_simulation(gateway_positions: np.ndarray, tag_traj: List[np.ndarray]):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(gateway_positions[:, 0], gateway_positions[:, 1], marker='*', s=200, label='Gateways (APs)')
    scat = ax.scatter([], [], s=100, label='Tags')

    def init():
        minx = min(np.min(gateway_positions[:, 0]), np.min(tag_traj[0][:, 0])) - 1
        maxx = max(np.max(gateway_positions[:, 0]), np.max(tag_traj[0][:, 0])) + 1
        miny = min(np.min(gateway_positions[:, 1]), np.min(tag_traj[0][:, 1])) - 1
        maxy = max(np.max(gateway_positions[:, 1]), np.max(tag_traj[0][:, 1])) + 1
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Tag Simulation (static positions shown over time)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return scat,

    def update(frame):
        scat.set_offsets(tag_traj[frame])
        ax.set_title(f"Tag Simulation (step {frame})")
        return scat,

    ani = FuncAnimation(fig, update, frames=len(tag_traj), init_func=init, blit=True, interval=250, repeat=False)
    plt.show()

#main
if __name__ == "__main__":
    print("Downloading AP location files...")
    ap_coords1 = download_aploc(URL1)
    ap_coords2 = download_aploc(URL2)
    print(f"Loaded {ap_coords1.shape[0]} AP coordinates from URL1 and {ap_coords2.shape[0]} from URL2")
    gateway_positions = ap_coords1.copy()
    n_tags = min(6, ap_coords2.shape[0])
    rng = np.random.default_rng(123)
    pick_idx = rng.choice(np.arange(ap_coords2.shape[0]), size=n_tags, replace=False)
    start_tags = ap_coords2[pick_idx] + rng.normal(scale=0.3, size=(n_tags, 2)) 

    print("Gateway positions (first 8):\n", np.round(gateway_positions[:8], 2))
    print("Start tag positions:\n", np.round(start_tags, 2))

    # Run simulation
    filtered_rssi, traj, hist = simulate_tags(
        gateway_positions=gateway_positions,
        start_tags=start_tags,
        steps=40,
        method="kalman",   
        rng_seed=2025,
        export="csv"
    )

    print("\nFinal filtered RSSI matrix (tags x gateways), rounded:")
    print(np.round(filtered_rssi[:, :8], 2))  # printing first 8 gateways columns for readability

    pd.DataFrame(gateway_positions, columns=["x", "y"]).to_csv("gateway_positions.csv", index=False)
    pd.DataFrame(start_tags, columns=["x", "y"]).to_csv("start_tag_positions.csv", index=False)
    print("Saved gateway_positions.csv and start_tag_positions.csv")

    animate_simulation(gateway_positions, traj)
