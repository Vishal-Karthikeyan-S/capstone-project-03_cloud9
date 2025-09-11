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


material_locations = {
    f"MAT{str(i+1).zfill(3)}": [random.randint(0, 100), random.randint(0, 100)] # random material location
    for i in range(50)
}

def get_rssi(tag_position):
    rssi_values = {}

    for zone, gw_pos in edge_gateways.items():
        distance = np.linalg.norm(tag_position - np.array(gw_pos)) # get tag's position
        rssi = -59 - 10 * 2 * np.log10(distance + 1e-5) + np.random.normal(0, 3) # -> closer , less neg val. larger distance , positive val. and random noise
        rssi_values[zone] = rssi
    return rssi_values

def get_coarse_location(tag_position):
    rssi_data = get_rssi(tag_position) # max of rssi -> strogest signal
    max_rssi = max(rssi_data.values())
    
    tied_zones = [zone for zone, rssi in rssi_data.items() if abs(rssi - max_rssi) <= 1.0]#to find which edge gateway has same rssi value
    
    if len(tied_zones) == 1:
        coarse_location = tied_zones[0]

    else: # if multiple zones there, calculate the distance from user positon and edge gateway
        distances = [np.linalg.norm(user_pos - np.array(edge_gateways[zone])) for zone in tied_zones]
        nearest_idx = np.argmin(distances)
        coarse_location = tied_zones[nearest_idx]
    
    return coarse_location, max_rssi, rssi_data

def plot_all(material_positions, selected_tag_pos, coarse_zone): # plotting all values in graph
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
    
    plt.grid(True)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    
    plt.show()

def simulation():
    print("Available Material IDs:")
    print(", ".join(list(material_locations.keys())))
    
    material_id = input("\nEnter Material ID (e.g., MAT001): ").strip().upper()
    
    if material_id in material_locations:
        tag_pos = np.array(material_locations[material_id])
        
        coarse_zone, max_rssi, rssi_details = get_coarse_location(tag_pos)
        
        print(f"\nMaterial ID: {material_id}")
        print(f"Material Position: {tag_pos.tolist()}")
        print(f"RSSI values: {rssi_details}")
        print(f"Coarse Location Zone: {coarse_zone} (Max RSSI: {max_rssi:.2f} dBm)")
        
        plot_all(material_locations, tag_pos, coarse_zone)
    else:
        print(f"Material ID '{material_id}' not found!")

simulation()
