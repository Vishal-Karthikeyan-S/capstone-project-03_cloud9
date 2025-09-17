import math, random, uuid, statistics, time
import simpy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

edge_gateways = {
    'Zone A': [0, 0],
    'Zone B': [100, 0],
    'Zone C': [0, 100],
    'Zone D': [100, 100],
    'Zone E': [50, 50]
} 

door_position = np.array([0, 50])
user_pos = np.array([5, 50])

material_locations = {
    f"MAT{str(i+1).zfill(3)}": [random.randint(0, 100), random.randint(0, 100)]
    for i in range(50)
}


def get_rssi(tag_position):
    rssi_values = {}
    for zone, gw_pos in edge_gateways.items():
        distance = np.linalg.norm(tag_position - np.array(gw_pos))
        rssi = -59 - 10 * 2 * np.log10(distance + 1e-5) + np.random.normal(0, 3)
        rssi_values[zone] = rssi
    return rssi_values


def get_coarse_location(tag_position):
    rssi_data = get_rssi(tag_position)
    max_rssi = max(rssi_data.values())
    tied_zones = [zone for zone, rssi in rssi_data.items()
                  if abs(rssi - max_rssi) <= 1.0]
    if len(tied_zones) == 1:
        coarse_location = tied_zones[0]
    else:
        distances = [np.linalg.norm(user_pos - np.array(edge_gateways[zone]))
                     for zone in tied_zones]
        nearest_idx = np.argmin(distances)
        coarse_location = tied_zones[nearest_idx]
    return coarse_location, max_rssi, rssi_data


def plot_all(material_positions, selected_tag_pos, coarse_zone):
    plt.figure(figsize=(10, 10))

    for mat_id, pos in material_positions.items():
        plt.scatter(pos[0], pos[1], c='gray', marker='o', s=50, alpha=0.6)
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


# simulation() 

random_seed = 123
simulation_time = 60.0
arr_rate = 30
edge_nodes = [
    {"id": "edge-1", "cpu": 2, "mem": 256, "throughput": 2e6, "uplink_ms": 20, "downlink_ms": 20, "gw_id": "gw1"},
    {"id": "edge-2", "cpu": 2, "mem": 256, "throughput": 2e6, "uplink_ms": 25, "downlink_ms": 25, "gw_id": "gw2"},
]
cloud_nodes = {"id": "cloud", "cpu": 16, "mem": 16384, "throughput": 50e6, "uplink_ms": 120, "downlink_ms": 100}


def ms_to_sec(v): return max(0.0, v) / 1000.0
def service_time(size_bytes, throughput): return size_bytes / throughput


class MockBus:
    def __init__(self, env):
        self.env = env
        self.subs = defaultdict(list)

    def subscribe(self, topic, cb):
        self.subs[topic].append(cb)

    def publish(self, topic, msg, delay=0.0):
        def _deliver():
            for cb in list(self.subs.get(topic, [])):
                try:
                    cb(self.env, topic, msg)
                except Exception as e:
                    print("Callback error:", e)
        self.env.process(self._delayed_call(delay, _deliver))

    def _delayed_call(self, delay, fn):
        yield self.env.timeout(delay)
        fn()


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def knn_predict(train_X, train_y, query, k=3):
    dists = [(euclidean(x, query), y) for x, y in zip(train_X, train_y)]
    dists.sort(key=lambda t: t[0])
    neighbors = dists[:k]
    eps = 1e-6
    numx = 0.0
    numy = 0.0
    denom = 0.0
    for dist, pos in neighbors:
        w = 1.0 / (dist + eps)
        numx += pos[0] * w
        numy += pos[1] * w
        denom += w
    return (numx / denom, numy / denom)


class Node:
    def __init__(self, env, spec, bus, fingerprint_db=None):
        self.env = env
        self.id = spec["id"]
        self.cpu = simpy.Resource(env, capacity=spec["cpu"])
        self.mem = simpy.Container(env, init=spec["mem"], capacity=spec["mem"])
        self.throughput = spec["throughput"]
        self.uplink = spec["uplink_ms"]
        self.downlink = spec["downlink_ms"]
        self.queue_len = 0
        self.completed = 0
        self.busy_time = 0.0
        self.bus = bus
        self.spec = spec
        self.fingerprint_db = fingerprint_db

    def estimate_finish_time(self, expected_size):
        avg_service = service_time(expected_size, self.throughput)
        return max(0, self.queue_len) * avg_service + avg_service

    def process_task(self, task, result_topic):
        size = task["size_bytes"]
        mem_need = min(64, max(1, size // 1024))
        self.queue_len += 1
        with self.cpu.request() as req:
            yield req
            yield self.mem.get(mem_need)
            start = self.env.now
            if self.spec.get("gw_id") is not None:
                rmap = task.get("rssi_map", {})
                if rmap:
                    best_gw = max(rmap.items(), key=lambda kv: kv[1])[0]
                    coarse = {"task_id": task["task_id"], "node": self.id,
                              "coarse_loc": best_gw, "ts": self.env.now}
                    self.bus.publish("location/coarse", coarse, delay=0.0)
            work = service_time(size, self.throughput)
            yield self.env.timeout(work)
            if self.id == "cloud" and self.fingerprint_db is not None:
                gw_order = list(edge_gateways.keys())
                query = [task["rssi_map"].get(gid, -120.0) for gid in gw_order]
                X = [f["rssi_vector"] for f in self.fingerprint_db]
                Y = [f["pos"] for f in self.fingerprint_db]
                refined_pos = knn_predict(X, Y, query, k=3)
                refined = {"task_id": task["task_id"], "node": self.id,
                           "refined_pos": refined_pos, "true_pos": task["true_pos"], "ts": self.env.now}
                self.bus.publish("location/refined", refined, delay=ms_to_sec(self.downlink))
            self.mem.put(mem_need)
            self.completed += 1
            self.busy_time += (self.env.now - start)
        self.queue_len -= 1
        self.bus.publish(result_topic,
                         {"task_id": task["task_id"], "node": self.id, "finish": self.env.now},
                         delay=ms_to_sec(self.downlink))

class Scheduler:
    def __init__(self, env, bus, edges, cloud, policy='FCFS'):
        self.env = env
        self.bus = bus
        self.policy = policy
        self.nodes = {}
        for n in edges:
            self.nodes[n["id"]] = Node(env, n, bus, fingerprint_db=None)
        self.nodes[cloud["id"]] = Node(env, cloud, bus, fingerprint_db=None)
        self.edge_ids = [n["id"] for n in edges]
        self.cloud_id = cloud["id"]
        self.arrivals = 0
        self.completed = 0
        self.latencies = []
        self.route_counts = {"edge": 0, "cloud": 0}
        self.loc_errors = []
        self.start_times = {} 

        bus.subscribe("edge/coarse", self.on_task)
        bus.subscribe("node/result", self.on_result)
        bus.subscribe("location/refined", self.on_refined)

    def set_cloud_db(self, fingerprint_db):
        self.nodes[self.cloud_id].fingerprint_db = fingerprint_db

    def on_task(self, env, topic, msg):
        self.arrivals += 1
        task = dict(msg)
        task["created_sim_ts"] = self.env.now
        self.start_times[task["task_id"]] = self.env.now
        node_id, uplink_delay = self.decide(task)

        def assign():
            node = self.nodes[node_id]
            result_topic = "node/result"
            env.process(node.process_task(task, result_topic))

        self.env.process(self._delay_and_call(uplink_delay, assign))

    def _delay_and_call(self, delay, fn):
        yield self.env.timeout(delay)
        fn()

    def on_result(self, env, topic, msg):
        self.completed += 1
        start = self.start_times.pop(msg["task_id"], None)
        if start is not None:
            latency = self.env.now - start
            self.latencies.append(latency)

    def on_refined(self, env, topic, msg):
        true = msg.get("true_pos")
        pred = msg.get("refined_pos")
        if true and pred:
            err = euclidean(true, pred)
            self.loc_errors.append(err)

    def decide(self, task):
        size = task["size_bytes"]
        if self.policy == 'FCFS':
            node = self.nodes[self.cloud_id]
            self.route_counts["cloud"] += 1
            return node.id, ms_to_sec(node.uplink)
        elif self.policy == 'LB':
            candidates = [self.nodes[self.cloud_id]] + [self.nodes[eid] for eid in self.edge_ids]
            best = min(candidates, key=lambda n: n.estimate_finish_time(size))
            if best.id == self.cloud_id:
                self.route_counts["cloud"] += 1
            else:
                self.route_counts["edge"] += 1
            return best.id, ms_to_sec(best.uplink)
        
        edges = [self.nodes[eid] for eid in self.edge_ids]
        best_edge = min(edges,
                        key=lambda n: n.estimate_finish_time(size) + ms_to_sec(n.uplink) + ms_to_sec(n.downlink))
        cloud = self.nodes[self.cloud_id]
        edge_e2e = best_edge.estimate_finish_time(size) + ms_to_sec(best_edge.uplink) + ms_to_sec(best_edge.downlink)
        cloud_e2e = cloud.estimate_finish_time(size) + ms_to_sec(cloud.uplink) + ms_to_sec(cloud.downlink)
        choice = best_edge if edge_e2e < cloud_e2e else cloud
        if choice.id == self.cloud_id:
            self.route_counts["cloud"] += 1
        else:
            self.route_counts["edge"] += 1
        return choice.id, ms_to_sec(choice.uplink)


def path_loss_rssi(tx_pos, rx_pos, tx_power_dbm=0.0, path_loss_exponent=2.0, sigma=2.0):
    dx = tx_pos[0] - rx_pos[0]
    dy = tx_pos[1] - rx_pos[1]
    d = math.sqrt(dx * dx + dy * dy)
    d = max(d, 0.1)
    rssi = tx_power_dbm - 10 * path_loss_exponent * math.log10(d) + random.gauss(0, sigma)
    return rssi


def material_task_source(env, bus, material_id, rate_per_min=10):
    inter = 60.0 / rate_per_min
    tag_pos = np.array(material_locations[material_id])
    while True:
        yield env.timeout(random.expovariate(1.0 / inter))
        rssi_map = get_rssi(tag_pos)
        coarse_zone, max_rssi, _ = get_coarse_location(tag_pos)
        task = {
            "task_id": str(uuid.uuid4()),
            "tag_id": material_id,
            "size_bytes": random.randint(2000, 8000),
            "rssi_map": rssi_map,
            "true_pos": tag_pos.tolist(),
            "source_gw": coarse_zone,
            "created_real_ts": env.now
        }
        bus.publish("edge/coarse", task)


def build_fingerprint_db(num_points=200):
    db = []
    gw_positions = list(edge_gateways.values())
    for _ in range(num_points):
        pos = (random.uniform(0, 100), random.uniform(0, 100))
        rvec = [path_loss_rssi(pos, gw_pos) for gw_pos in gw_positions]
        db.append({"pos": pos, "rssi_vector": rvec})
    return db


def run_demo_for_policy(policy):
    random.seed(random_seed)
    env = simpy.Environment()
    bus = MockBus(env)
    sched = Scheduler(env, bus, edge_nodes, cloud_nodes, policy=policy)
    fp_db = build_fingerprint_db(300)
    sched.set_cloud_db(fp_db)

    material_id = "MAT017"
    env.process(material_task_source(env, bus, material_id, rate_per_min=arr_rate))
    env.run(until=simulation_time)

    completed = sum(n.completed for n in sched.nodes.values())
    node_utils = {nid: node.busy_time / simulation_time for nid, node in sched.nodes.items()}
    avg_util = np.mean(list(node_utils.values()))
    avg_loc_err = np.mean(sched.loc_errors) if sched.loc_errors else 0.0
    avg_lat = np.mean(sched.latencies) if sched.latencies else 0.0

    print("\n Policy:", policy, " ")
    print("Arrivals:", sched.arrivals,
          " Cloud:", sched.route_counts['cloud'],
          " Edge:", sched.route_counts['edge'],
          " Completed:", completed)
    for nid, node in sched.nodes.items():
        print(f"  Node {nid}: completed={node.completed}, util={node.busy_time / simulation_time * 100:.2f}%")

    return {
        "policy": policy,
        "avg_latency": avg_lat,
        "avg_util": avg_util,
        "avg_loc_err": avg_loc_err
    }


def run_all_policies():
    all_results = []
    for policy in ['FCFS', 'LB', 'LatencyAware']:
        res = run_demo_for_policy(policy)
        all_results.append(res)

    print("\nSummary:")
    for res in all_results:
        print(f"{res['policy']:12s} Latency={res['avg_latency']:.3f}  Util={res['avg_util']:.3f}  LocErr={res['avg_loc_err']:.3f}")

    plt.figure(figsize=(6, 5))
    for res in all_results:
        plt.scatter(res['avg_latency'], res['avg_loc_err'], label=res['policy'], s=100)
    plt.xlabel("Average Latency")
    plt.ylabel("Average Location Error")
    plt.title("Policy Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
run_all_policies()
