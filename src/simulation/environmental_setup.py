import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist #
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
#rssi
def rssi_from_distance(dist, P0=-59, n=2.0, d0=1.0, shadow_sigma=2.0, rng=None):
    with np.errstate(divide='ignore'):
        path_loss = -10 * n * np.log10(np.maximum(dist, d0) / d0)
    shadow = 0 if rng is None else rng.normal(0, shadow_sigma, size=dist.shape)
    return P0 + path_loss + shadow

rng = np.random.default_rng(42)
dists = cdist(tag_positions, gateway_positions)   #shape:(N_tags×N_gateways)
rssi_vals = rssi_from_distance(dists, rng=rng)
print("Distances (m):")
print(np.round(dists, 2))
print("\nRSSI values (dBm):")
print(np.round(rssi_vals, 2))

plt.figure(figsize=(8,5))

plt.scatter(gateway_positions[:,0], gateway_positions[:,1],
            marker='*', s=250, c='red', label='Gateways')
for idx,(x,y) in enumerate(gateway_positions):
    plt.text(x+0.3,y+0.3,f"G{idx}",color="red")

# Tags
colors = ['blue','green','orange','purple','brown']
plt.scatter(tag_positions[:,0], tag_positions[:,1], c=colors, s=80, label='Tags')
for idx,(x,y) in enumerate(tag_positions):
    plt.text(x+0.3,y+0.3,f"T{idx}",color=colors[idx])

plt.title("2D Layout with Fixed Tags (Materials) and Gateways")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().set_aspect("equal","box")
plt.show()