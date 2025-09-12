import math, random, uuid, statistics, time
import simpy  
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

## integrate processing folder code 

random_seed= 123
simulation_time= 60.0            
arr_rate= 30          
sched_policy= 'LATENCY'   

edge_nodes= [
    {"id":"edge-1","cpu":2,"mem":256,"throughput":2e6,"uplink_ms":20,"downlink_ms":20, "gw_id":"gw1"},
    {"id":"edge-2","cpu":2,"mem":256,"throughput":2e6,"uplink_ms":25,"downlink_ms":25, "gw_id":"gw2"},
]
cloud_nodes= {"id":"cloud","cpu":16,"mem":16384,"throughput":50e6,"uplink_ms":120,"downlink_ms":100}

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
def euclidean(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def knn_predict(train_X, train_y, query, k=3):
    dists = [(euclidean(x, query), y) for x,y in zip(train_X, train_y)]
    dists.sort(key=lambda t: t[0])
    neighbors = dists[:k]
    eps = 1e-6
    numx = 0.0; numy = 0.0; denom = 0.0
    for dist, pos in neighbors:
        w = 1.0/(dist+eps)
        numx += pos[0]*w
        numy += pos[1]*w
        denom += w
    return (numx/denom, numy/denom)
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
                    coarse = {"task_id": task["task_id"], "node": self.id, "coarse_loc": best_gw, "ts": self.env.now}
                    self.bus.publish("location/coarse", coarse, delay=0.0)
            work = service_time(size, self.throughput)
            yield self.env.timeout(work)
            if self.id == "cloud" and self.fingerprint_db is not None:
                gw_order = [g["id"] for g in edge_gateways]
                query = [task["rssi_map"].get(gid, -120.0) for gid in gw_order]
                X = [f["rssi_vector"] for f in self.fingerprint_db]
                Y = [f["pos"] for f in self.fingerprint_db]
                refined_pos = knn_predict(X, Y, query, k=3)
                refined = {"task_id": task["task_id"], "node": self.id, "refined_pos": refined_pos, "ts": self.env.now}
                self.bus.publish("location/refined", refined, delay=ms_to_sec(self.downlink))
            self.mem.put(mem_need)
            self.completed += 1
            self.busy_time += (self.env.now - start)
        self.queue_len -= 1
        self.bus.publish(result_topic, {"task_id": task["task_id"], "node": self.id, "finish": self.env.now}, delay=ms_to_sec(self.downlink))

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
        self.route_counts = {"edge":0, "cloud":0}
        bus.subscribe("edge/coarse", self.on_task)
        bus.subscribe("node/result", self.on_result)

    def set_cloud_db(self, fingerprint_db):
        self.nodes[self.cloud_id].fingerprint_db = fingerprint_db

    def on_task(self, env, topic, msg):
        self.arrivals += 1
        task = dict(msg)
        task["created_sim_ts"] = self.env.now
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
        self.latencies.append(0.0)

    def decide(self, task):
        size = task["size_bytes"]
        if self.policy == 'FCFS':
            node = self.nodes[self.cloud_id]
            self.route_counts["cloud"] += 1
            return node.id, ms_to_sec(node.uplink)
        if self.policy == 'LB':
            candidates = [self.nodes[self.cloud_id]] + [self.nodes[eid] for eid in self.edge_ids]
            best = min(candidates, key=lambda n: n.estimate_finish_time(size))
            if best.id == self.cloud_id:
                self.route_counts["cloud"] += 1
            else:
                self.route_counts["edge"] += 1
            return best.id, ms_to_sec(best.uplink)
        edges = [self.nodes[eid] for eid in self.edge_ids]
        best_edge = min(edges, key=lambda n: n.estimate_finish_time(size) + ms_to_sec(n.uplink) + ms_to_sec(n.downlink))
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
    dx = tx_pos[0]-rx_pos[0]; dy = tx_pos[1]-rx_pos[1]
    d = math.sqrt(dx*dx+dy*dy)
    d = max(d, 0.1)
    rssi = tx_power_dbm - 10*path_loss_exponent*math.log10(d) + random.gauss(0, sigma)
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
            "source_gw": coarse_zone,   # strongest coarse zone
            "created_real_ts": env.now
        }

        
        bus.publish("edge/coarse", task)

def build_fingerprint_db(num_points=200):
    db = []
    zones = list(edge_gateways.keys())  
    gw_positions = list(edge_gateways.values()) 

    for _ in range(num_points):
        pos = (random.uniform(0, 100), random.uniform(0, 100))  
        rvec = [path_loss_rssi(pos, gw_pos) for gw_pos in gw_positions]
        db.append({"pos": pos, "rssi_vector": rvec})
    return db
def run_demo():
    random.seed(random_seed)
    env = simpy.Environment()
    bus = MockBus(env)
    sched = Scheduler(env, bus, edge_nodes, cloud_nodes, policy=sched_policy)
    fp_db = build_fingerprint_db(300)
    sched.set_cloud_db(fp_db)
    def on_coarse(env, topic, msg):
        print(f"[{env.now:.3f}] COARSE -> task={msg['task_id'][:8]} coarse_loc={msg['coarse_loc']}")
    def on_refined(env, topic, msg):
        print(f"[{env.now:.3f}] REFINED -> task={msg['task_id'][:8]} refined_pos=({msg['refined_pos'][0]:.2f},{msg['refined_pos'][1]:.2f})")
    bus.subscribe("location/coarse", on_coarse)
    bus.subscribe("location/refined", on_refined)
    material_id = "MAT017"
    env.process(material_task_source(env, bus, material_id, rate_per_min=arr_rate))

    env.run(until=simulation_time)
    print("Policy:",sched_policy)
    print("Arrivals (scheduler saw):",sched.arrivals)
    print("Routed to cloud:", sched.route_counts["cloud"],"|", " Routed to edge:", sched.route_counts["edge"])
    completed = sum(n.completed for n in sched.nodes.values())
    print("Completed tasks (node counts):", completed)
    for nid,node in sched.nodes.items():
        util = node.busy_time / simulation_time if simulation_time>0 else 0.0
        print(f"Node{nid}:completed={node.completed},utilization={util*100:.2f}%")
    return {"sched":sched, "fp_db_size": len(fp_db)}

def main():
    run_demo()
main()
