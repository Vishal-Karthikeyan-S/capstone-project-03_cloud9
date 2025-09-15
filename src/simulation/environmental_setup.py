import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
import pandas as pd
gateway_positions = np.array([
    [0, 0],
    [20, 0],
    [20, 10],
    [0, 10]
])
def rssi_from_distance(dist,P0=-59,n=2.0,d0=1.0,shadow_sigma=2.0,rng=None):
    with np.errstate(divide='ignore'):
        path_loss = -10*n*np.log10(np.maximum(dist, d0)/d0)
    shadow=0 if rng is None else rng.normal(0,shadow_sigma,size=dist.shape)
    return P0 + path_loss + shadow

def make_kalman_filter(x0=-60,Q=0.1,R=1.0):
    kf = KalmanFilter(dim_x=1,dim_z=1)
    kf.x = np.array([[x0]])   #initial state
    kf.F = np.array([[1]])    #state transition
    kf.H = np.array([[1]])    #measurement function
    kf.P = np.array([[1]])    #covariance
    kf.R = np.array([[R]])    #measurement noise
    kf.Q = np.array([[Q]])    #process noise
    return kf

class MovingAverage:
    def __init__(self,window=5):
        self.window=window
        self.values=[]
    def update(self,val):
        self.values.append(val)
        if len(self.values)>self.window:
            self.values.pop(0)
        return np.mean(self.values)

def simulate_tags(tag_positions,steps=50,method="kalman",rng_seed=42,export=None):
    rng=np.random.default_rng(rng_seed)
    n_tags=tag_positions.shape[0]
    n_gws=gateway_positions.shape[0]

    if method=="kalman":
        filters={(i,j):make_kalman_filter() for i in range(n_tags) for j in range(n_gws)}
    else:
        filters={(i,j):MovingAverage(window=5) for i in range(n_tags) for j in range(n_gws)}

    rssi_hist={key: [] for key in filters}

    tag_traj=[tag_positions.copy() for _ in range(steps)]

    for t in range(steps):
        dists=cdist(tag_traj[t], gateway_positions)
        rssi_vals=rssi_from_distance(dists,rng=rng)
        for i in range(n_tags):
            for j in range(n_gws):
                meas=rssi_vals[i,j]
                f=filters[(i,j)]
                if method=="kalman":
                    f.predict()
                    f.update(meas)
                    val=f.x[0,0]
                else:
                    val=f.update(meas)
                rssi_hist[(i,j)].append(val)

    filtered_map=np.array([
        [rssi_hist[(i,j)][-1] for j in range(n_gws)]
        for i in range(n_tags)
    ])

    if export=="csv":
        df = pd.DataFrame({
            f"T{i}_G{j}":rssi_hist[(i,j)]
            for i in range(n_tags)
            for j in range(n_gws)
        })
        df.to_csv("rssi_timeseries.csv",index_label="Step")
        print("Exported full RSSI time series to rssi_timeseries.csv")

    elif export=="json":
        import json
        out={
            f"T{i}":{f"G{j}": rssi_hist[(i,j)] for j in range(n_gws)}
            for i in range(n_tags)
        }
        with open("rssi_timeseries.json","w") as f:
            json.dump(out,f,indent=2)
        print("Exported full RSSI time series to rssi_timeseries.json")
    return filtered_map,tag_traj,rssi_hist

def animate_simulation(tag_traj):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(gateway_positions[:,0],gateway_positions[:,1],marker='*',s=200,c='red',label='Gateways')
    scat=ax.scatter([],[],c='blue',s=80,label='Tags')

    def init():
        ax.set_xlim(-1,21);ax.set_ylim(-1,11)
        ax.set_title("Tag Simulation (fixed positions)")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.legend();ax.grid(True, alpha=0.3)
        return scat,

    def update(frame):
        scat.set_offsets(tag_traj[frame])
        ax.set_title(f"Tag Simulation (step {frame})")
        return scat,

    ani = FuncAnimation(fig,update,frames=len(tag_traj),init_func=init,blit=True,interval=300,repeat=False)
    plt.show()

if __name__ == "__main__":
    start_tags = np.array([
        [2, 3],
        [5, 8],
        [10, 5],
        [15, 2],
        [8, 9],
        [18, 6]
    ])
    filt_rssi,traj,hist = simulate_tags(start_tags,steps=40,method="kalman",export="csv")
    print("Final Filtered RSSI:\n",np.round(filt_rssi,2))
    animate_simulation(traj)

