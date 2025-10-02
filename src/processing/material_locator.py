import numpy as np
import matplotlib.pyplot as plt
import random
import json,os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from heapq import heappush, heappop
import time

edge_gateways = {
'Zone A': [20, 20],
'Zone B': [60, 20],
'Zone C': [100, 20],
'Zone D': [100, 60],
'Zone E': [100, 100],
'Zone F': [60, 100],
'Zone H': [20, 100],
'Zone I': [20, 60],
'ZONE J': [60,60]
} # fixed edge gateway locations

# door and user position
door_position = np.array([-2, 50])
user_pos = np.array([0, 50])

res_x = (0, 20) # restricted near door(to place materials)
res_y = (40, 60)

# materials can't be placed on grid lines
def is_on_grid_line(x, y, grid_spacing=10):
    return x % grid_spacing == 0 or y % grid_spacing == 0

def valid_pos(x, y):

    if res_x[0] <= x <= res_x[1] and res_y[0] <= y <= res_y[1]:
        return False
    
    if is_on_grid_line(x, y, grid_spacing=10):
        return False
    return True


if os.path.exists("mat_loc.json"):
    with open("mat_loc.json", "r") as f:
        mat_loc = json.load(f)

    materials_to_regenerate = [] # regenerate materials if they're on grid lines
    for mat_id, pos in mat_loc.items():
        if not valid_pos(pos[0], pos[1]):
            materials_to_regenerate.append(mat_id)
    
    for mat_id in materials_to_regenerate: # for mat_id placed on grid, replace it
        while True:
            x = random.randint(1, 99) 
            y = random.randint(1, 99)
            if valid_pos(x, y):
                mat_loc[mat_id] = [x, y]
                break
    
    # save if any materials were regenerated
    if materials_to_regenerate:
        with open("mat_loc.json", "w") as f:
            json.dump(mat_loc, f)
        print(f"Regenerated {len(materials_to_regenerate)} materials away from grid lines.")
else:
    mat_loc = {}
    for i in range(200):
        while True:
            x = random.randint(1, 99)  
            y = random.randint(1, 99)
            if valid_pos(x, y):
                mat_loc[f"MAT{str(i+1).zfill(3)}"] = [x, y]
                break
    with open("mat_loc.json", "w") as f:
        json.dump(mat_loc, f)

def get_rssi(tag_position,add_noise=True):
    rssi_values = {}

    for zone, gw_pos in edge_gateways.items():
        distance = np.linalg.norm(tag_position - np.array(gw_pos)) # get tag's position
        # rnoise = np.random.normal(0, 3) if add_noise else 0
        rssi = -59 - 10 * 2 * np.log10(distance + 1e-5) #+ rnoise # -> closer , less neg val. larger distance , positive val. and random noise
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

if os.path.exists("fingerprint.json"):
    with open("fingerprint.json", "r") as f:
        dataset_ = json.load(f)
    print(f"\nLoading existing fingerprint dataset...")
    
else:
    dataset_ = fingerprint_dataset(step=1)
    with open("fingerprint.json", "w") as f:
        json.dump(dataset_, f)
    print(f"\nGenerated new fingerprint dataset....")



# models for analysis : knn, RandomForestReg. , mlp
X = np.array([entry["rssi"] for entry in dataset_], dtype=float)
y = np.array([entry["pos"] for entry in dataset_], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

knn_model = KNeighborsRegressor(n_neighbors=7, weights='distance')
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
mlp_model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42, early_stopping=True)

knn_model.fit(X_train_s, y_train)
rf_model.fit(X_train_s, y_train)
mlp_model.fit(X_train_s, y_train)



def cloud_computation(tag_position):
    print("Computing the exact location....waiting for cloud response...")

    tag_rssi = list(get_rssi(tag_position, add_noise=False).values())
    tag_rssi = np.array(tag_rssi).reshape(1, -1)
    tag_rssi_scaled = scaler.transform(tag_rssi)

    models = {
        "KNN": knn_model,
        "Random Forest": rf_model,
        "MLP (NN)": mlp_model
    }

    results = {}
    for name, model in models.items():
        pred = model.predict(tag_rssi_scaled)[0]
        error = np.linalg.norm(pred - tag_position)
        results[name] = {"pred": pred, "error": error}
        print(f"\n{name} → Predicted: {np.round(pred,2)}, Error: {error:.2f} meters")

    best_model = min(results, key=lambda k: results[k]["error"])
    print(f"\n ==>> Best Model: {best_model} with error {results[best_model]['error']:.2f} meters")

    # bar chart for errors by ecul. dist.
    plt.figure(figsize=(6,4))
    model_names = list(results.keys())
    errors = [results[m]["error"] for m in model_names]
    plt.bar(model_names, errors, color=['blue','green','orange'])
    plt.ylabel("Localization Error (m)")
    plt.title("Model Comparison for This Material")
    plt.show(block=False)

    
    best_pred = results[best_model]["pred"]
    return {best_model: best_pred} 



def find_path_a_star(start, end):   
    
    start_grid = (round(start[0]//10)*10, round(start[1]//10)*10) # user pos(nearest to grid point)
    end_exact = (end[0], end[1]) # material place
    end_grid = (round(end[0]//10)*10, round(end[1]//10)*10)  # nearest grid point to the material 
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) # manhattan
    
    def get_neighbors(pos):
        neighbors = []
        # only move along grid lines (* of 10), not through diagonals
        for dx, dy in [(0,10), (10,0), (0,-10), (-10,0)]:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if 0 <= new_x <= 100 and 0 <= new_y <= 100:
                neighbors.append((new_x, new_y))
        return neighbors
    
    # A* algorithm
    start_time = time.time()
    
    open_set = [(0, start_grid)] # keeps track of which nodes to explore next
    came_from = {} # for shortest path
    g_score = {start_grid: 0} # actual cost from start to each node
    f_score = {start_grid: heuristic(start_grid, end_grid)} # tracks estimated total cost
    nodes_explored = 0
    
    while open_set:
        current = heappop(open_set)[1]
        nodes_explored += 1
        
        if current == end_grid: # curr path == mat_pos, we will will reverse it and find path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_grid)
            path = path[::-1]  
            
            # ensures final point
            if path[-1] != end_exact:
                path.append(end_exact)
            
            elapsed = (time.time() - start_time)*1000
            path_length = sum(np.linalg.norm(np.array(path[i])-np.array(path[i-1])) for i in range(1,len(path)))
            return path, nodes_explored, elapsed, path_length
        
        for neighbor in get_neighbors(current):
            tot_g_score = g_score[current] + 10  
            
            if neighbor not in g_score or tot_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tot_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end_grid)
                heappush(open_set, (f_score[neighbor], neighbor))
    
    elapsed = (time.time() - start_time)*1000
    return [start_grid,end_exact], nodes_explored, elapsed, 0


def find_path_dijkstra(start, end): # Diijkstra's algorihthm
    start_grid = (round(start[0]//10)*10, round(start[1]//10)*10)
    end_exact = (end[0], end[1])
    end_grid = (round(end[0]//10)*10, round(end[1]//10)*10)

    def get_neighbors(pos):
        neighbors = []
        for dx, dy in [(0,10),(10,0),(0,-10),(-10,0)]:
            new_x, new_y = pos[0]+dx, pos[1]+dy
            if 0<=new_x<=100 and 0<=new_y<=100:
                neighbors.append((new_x,new_y))
        return neighbors

    start_time = time.time()
    open_set = [(0, start_grid)]
    came_from = {}
    g_score = {start_grid:0}
    nodes_explored = 0

    while open_set:
        current = heappop(open_set)[1]
        nodes_explored += 1
        if current == end_grid:
            path=[]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_grid)
            path = path[::-1]
            if path[-1]!=end_exact:
                path.append(end_exact)
            elapsed = (time.time()-start_time)*1000
            path_length = sum(np.linalg.norm(np.array(path[i])-np.array(path[i-1])) for i in range(1,len(path)))
            return path, nodes_explored, elapsed, path_length
        
        for neighbor in get_neighbors(current):
            tot_g_score = g_score[current]+10
            if neighbor not in g_score or tot_g_score<g_score[neighbor]:
                came_from[neighbor]=current
                g_score[neighbor]=tot_g_score
                heappush(open_set,(g_score[neighbor],neighbor))

    elapsed = (time.time()-start_time)*1000
    return [start_grid,end_exact], nodes_explored, elapsed, 0



def path_analysis(user_pos, material_pos, material_id):
   # calling both algo
    path_a, nodes_a, time_a, length_a = find_path_a_star(user_pos, material_pos)
    path_d, nodes_d, time_d, length_d = find_path_dijkstra(user_pos, material_pos)

    
    print(f"\nA* Path     --> Nodes Explored: {nodes_a}, Path Length: {length_a:.2f}, Time: {time_a:.2f}ms")
    print(f"Dijkstra Path --> Nodes Explored: {nodes_d}, Path Length: {length_d:.2f}, Time: {time_d:.2f}ms")

    
    metrics = ['Nodes Explored', 'Path Length', 'Execution Time (ms)']
    astar_values = [nodes_a, length_a, time_a]
    dijk_values = [nodes_d, length_d, time_d]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, astar_values, width, label='A*', color='lightcoral')
    plt.bar(x + width/2, dijk_values, width, label='Dijkstra', color='skyblue')
    plt.xticks(x, metrics)
    plt.ylabel("values")
    plt.title(f"Pathfinding Analysis to Material {material_id}")
    plt.legend()
    plt.show(block=False)

    return path_a, path_d



def plot_all(material_positions, selected_tag_pos, coarse_zone, refined_location=None, show_path=False, path_a=None):

    plt.figure(figsize=(12, 10))
    
    for i in range(0, 101, 10):
        plt.axvline(x=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # plotting materials by avoiding grid lines
    for mat_id, pos in material_positions.items():
        plt.scatter(pos[0], pos[1], c='gray', marker='o', s=50, alpha=0.6)
        plt.text(pos[0] + 0.5, pos[1] + 0.5, mat_id, fontsize=6)
    
    
    for zone, pos in edge_gateways.items():
        plt.scatter(pos[0], pos[1], marker='s', s=200,
                   color='green' if zone == coarse_zone else 'blue')
        plt.text(pos[0] + 1, pos[1] + 1, zone, fontsize=9)
    
    plt.scatter(door_position[0], door_position[1], c='purple', marker='D', s=150)
    plt.text(door_position[0] - 2, door_position[1] + 2, 'Door', fontsize=10, color='purple')
    
    plt.scatter(user_pos[0], user_pos[1], c='orange', marker='*', s=200)
    plt.text(door_position[0] + 3, door_position[1] - 3, 'User', fontsize=10, color='purple')
    
    plt.scatter(selected_tag_pos[0], selected_tag_pos[1], c='red', marker='x', s=150)
    
    if refined_location is not None:
        best_model = min(refined_location, key=lambda m: np.linalg.norm(refined_location[m] - selected_tag_pos))
        best_pos = refined_location[best_model]

        colors = {"KNN": "olive", "Random Forest": "cyan", "MLP (NN)": "magenta"}
        markers = {"KNN": "X", "Random Forest": "P", "MLP (NN)": "D"}
        for model_name, pos in refined_location.items():
            plt.scatter(pos[0], pos[1], c=colors[model_name], marker=markers[model_name],s=150, label=f"{model_name} Prediction")
    
        
    if show_path and path_a is not None:
        plt.plot([p[0] for p in path_a], [p[1] for p in path_a],'lightcoral', linewidth=3, label='A* Path')

    plt.legend()      
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.show(block=False)


def simulation():
    print(f"\nAvailable Material IDs :")
    print(", ".join(list(mat_loc.keys())))
    material_id = input("\nEnter Material ID : ").strip().upper()
    
    if material_id in mat_loc:
        tag_pos = np.array(mat_loc[material_id])
        coarse_zone, max_rssi, rssi_details = get_coarse_location(tag_pos)
        
        print(f"\nMaterial ID: {material_id}")
        print("RSSI values:")

        for zone, rssi in rssi_details.items():
            print(f" {zone}: {rssi:.2f} dBm\n")
        print(f"\nMax RSSI: {max_rssi:.2f} dBm\n")
        print(f"Coarse Location Zone: {coarse_zone}")
        
        # for cloud
        c_input = input("\nDo you want to compute the exact location via Cloud? (yes/no): ").strip().lower()
        if c_input == 'yes':
            refined_zone = cloud_computation(tag_pos)
            print(f"Refined location: {refined_zone}")
            
            # path
            path_input = input("\nDo you want to see the path to the material? (yes/no): ").strip().lower()
            show_path = path_input == 'yes'
        else:
            refined_zone = None
            show_path = False
            print("Skipping Cloud computation.")

        path_a, _ = path_analysis(user_pos, tag_pos, material_id)
        plot_all(mat_loc, tag_pos, coarse_zone, refined_zone, show_path=show_path,path_a=path_a)
        
        m_input = input("\nHas the material picked up? (yes/no): ").strip().lower()
        if m_input == 'yes':
            del mat_loc[material_id]
            print(f"Material {material_id} has been picked up and removed from the material list.")
            
            with open("mat_loc.json", "w") as f:
                json.dump(mat_loc, f)
        else:
            print(f"Material is not picked up. It's still in list.")
    else:
        print(f"Material ID '{material_id}' not found!")


simulation()