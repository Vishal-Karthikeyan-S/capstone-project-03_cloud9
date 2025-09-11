import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
gateway_positions = np.array([
    [0, 0],
    [20, 0],
    [20, 10],
    [0, 10]
])
tag_positions = np.array([
    [2, 3],
    [5, 8],
    [10, 5],
    [15, 2],
    [12, 9]
])

steps = 40
step_size = 1.0
area = (20, 10)  
rng = np.random.default_rng(42)

positions_over_time = []
tags = tag_positions.copy()
for t in range(steps):
    tags = tags + rng.uniform(-step_size, step_size, tags.shape) 
    tags[:,0] = np.clip(tags[:,0], 0, area[0])  
    tags[:,1] = np.clip(tags[:,1], 0, area[1])
    positions_over_time.append(tags.copy())
positions_over_time = np.array(positions_over_time)  

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(gateway_positions[:,0], gateway_positions[:,1],
           marker='*', s=250, c='red', label='Gateways')
for idx,(x,y) in enumerate(gateway_positions):
    ax.text(x+0.3,y+0.3,f"G{idx}",color="red")

colors = ['blue','green','orange','purple','brown']
scat = ax.scatter(tag_positions[:,0], tag_positions[:,1], c=colors, s=80)

ax.set_xlim(-1, area[0]+1)
ax.set_ylim(-1, area[1]+1)
ax.set_aspect("equal","box")
ax.set_title("2D Layout with 5 Moving Tags and Gateways")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend()
ax.grid(True, alpha=0.3)

def update(frame):
    scat.set_offsets(positions_over_time[frame])
    ax.set_title(f"2D Layout with 5 Moving Tags (step {frame+1}/{steps})")
    return scat,

ani = FuncAnimation(fig, update, frames=steps, interval=500, blit=True)

plt.show()
