import numpy as np
import matplotlib.pyplot as plt
import random


edge_gateways = {
    'Zone A': [0, 0],
    'Zone B': [100, 0],
    'Zone C': [0, 100],
    'Zone D': [100, 100],
    'Zone E': [50, 50]
} # fixed edge gateway locations

# door and user position
door_position = np.array([0, 50])
user_pos = np.array([5, 50])  

res_x = (0, 20)     # restricted near door(to place materials)
res_y = (40, 60)

material_locations = {}
for i in range(50):
    while True:
        x = random.randint(0, 100) # random loaction to place materials
        y = random.randint(0, 100)
        if not (res_x[0] <= x <= res_x[1] and res_y[0] <= y <= res_y[1]):
            material_locations[f"MAT{str(i+1).zfill(3)}"] = [x, y]
            break

def get_rssi(tag_position,add_noise=True):
    rssi_values = {}

    for zone, gw_pos in edge_gateways.items():
        distance = np.linalg.norm(tag_position - np.array(gw_pos)) # get tag's position
        rnoise = np.random.normal(0, 3) if add_noise else 0
        rssi = -59 - 10 * 2 * np.log10(distance + 1e-5) + rnoise # -> closer , less neg val. larger distance , positive val. and random noise
        rssi_values[zone] = rssi
    return rssi_values

def get_coarse_location(tag_position):
    rssi_data = get_rssi(tag_position) # max of rssi -> strongest signal
    max_rssi = max(rssi_data.values())
    
    tied_zones = [zone for zone, rssi in rssi_data.items() if abs(rssi - max_rssi) <= 1.0]#to find which edge gateway has same rssi value
    
    if len(tied_zones) == 1:
        coarse_location = tied_zones[0]

    else: # if multiple zones there, calculate the distance from user positon and edge gateway
        distances = [np.linalg.norm(user_pos - np.array(edge_gateways[zone])) for zone in tied_zones]
        nearest_idx = np.argmin(distances)
        coarse_location = tied_zones[nearest_idx]
    
    return coarse_location, max_rssi, rssi_data

def fingerprint_dataset(step):
    dataset = []
    for x in range(0, 101, step):
        for y in range(0, 101, step):
            pos = np.array([x, y])
            rssi_vector = list(get_rssi(pos).values())
            dataset.append({"pos": (x, y), "rssi": rssi_vector})
    return dataset


dataset_ = fingerprint_dataset(step=1)

from sklearn.neighbors import NearestNeighbors
X = np.array([entry["rssi"] for entry in dataset_])
y = np.array([entry["pos"] for entry in dataset_])


knn = NearestNeighbors(n_neighbors=7, metric='euclidean')
knn.fit(X,y)


def cloud_computation(tag_position): 
    print("Computing the exact location....waiting for cloud response...")
    

    tag_rssi = list(get_rssi(tag_position, add_noise=False).values()) # curr rssi values

    tag_rssi = np.array(tag_rssi).reshape(1, -1)

    distances, position = knn.kneighbors(tag_rssi) # nearest neighbors

    neighbor_positions = y[position[0]]


    print(f"\nDistances to nearest neighbors:", np.round(distances[0], 2))

    print(f"\nNeighbor positions:", neighbor_positions)

    weights = 1 / (distances[0] + 1e-5)  
    refined_location = np.average(neighbor_positions, axis=0, weights=weights)


    return refined_location

def plot_all(material_positions, selected_tag_pos, coarse_zone,refined_location=None): # plotting all values in graph
    plt.figure(figsize=(10, 10))
    
    
    for mat_id, pos in material_positions.items():
        plt.scatter(pos[0], pos[1], c='gray', marker='o', s=50, alpha=0.6) # plot in circle shape 
        plt.text(pos[0] + 0.5, pos[1] + 0.5, mat_id, fontsize=6)
    
    for zone, pos in edge_gateways.items():
        plt.scatter(pos[0], pos[1], marker='s', s=200,
                    color='green' if zone == coarse_zone else 'blue')
        plt.text(pos[0] + 1, pos[1] + 1, zone, fontsize=9)
    
   
    plt.scatter(door_position[0], door_position[1], c='purple', marker='D', s=150)
    plt.text(door_position[0] + 1, door_position[1] + 1, 'Door', fontsize=10, color='purple')

    plt.scatter(user_pos[0], user_pos[1], c='orange', marker='*', s=200)

    plt.scatter(selected_tag_pos[0], selected_tag_pos[1], c='red', marker='x', s=150)

    if refined_location is not None:
        print(f"Plotting Refined Location: {refined_location}")
        plt.scatter(refined_location[0], refined_location[1], c='olive', marker='X', s=200)
    
    plt.grid(True)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)

    plt.show(block=False)

def simulation():
    print(f"\nAvailable Material IDs :")
    print(", ".join(list(material_locations.keys())))
    
    material_id = input("\nEnter Material ID : ").strip().upper()
    
    if material_id in material_locations:
        tag_pos = np.array(material_locations[material_id])
        
        coarse_zone, max_rssi, rssi_details = get_coarse_location(tag_pos)
        
        print(f"\nMaterial ID: {material_id}")
        print("RSSI values:")
        for zone, rssi in rssi_details.items():
          print(f"  {zone}: {rssi:.2f} dBm\n")

        print(f"\nMax RSSI: {max_rssi:.2f} dBm\n")
        print(f"Coarse Location Zone: {coarse_zone}")

        # for cloud
        c_input = input("\nDo you want to compute the exact location via Cloud? (yes/no): ").strip().lower()
        if c_input == 'yes':
            refined_zone = cloud_computation(tag_pos)
            print(f"Refined location: {refined_zone}")
        else:
            refined_zone = None
            print("Skipping Cloud computation.")


        plot_all(material_locations, tag_pos, coarse_zone,refined_zone)


        m_input = input("\nHas the material picked up? (yes/no): ").strip().lower()
        if m_input == 'yes':
            del material_locations[material_id]   
            print(f"Material {material_id} has been picked up and removed from the material list.")
        
        else:
            print(f"Material is not picked up. It's still in list.")


    else:
        print(f"Material ID '{material_id}' not found!")


simulation() 