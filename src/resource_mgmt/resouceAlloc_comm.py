import math, random, uuid, time
import simpy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from material_locator import (
    edge_gateways, door_position, user_pos,
    is_on_grid_line, valid_pos, simulation,
    get_rssi, get_coarse_location, cloud_computation,
    fingerprint_dataset, mat_loc
)

def ms_to_sec(v): 
    return max(0.0, v) / 1000.0

def service_time(size_bytes, throughput): 
    return size_bytes / throughput

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

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
                refined_pos = cloud_computation(np.array(task["true_pos"]))
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

        bus.subscribe("task/new", self.on_task)
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
        if true is not None and pred is not None:
            true_arr = np.array(true)
            pred_arr = np.array(pred)
            err = euclidean(true_arr, pred_arr)
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

def material_task_source(env, bus, material_id, rate_per_min=10):
    inter = 60.0 / rate_per_min
    tag_pos = np.array(mat_loc[material_id])
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
        bus.publish("task/new", task)

random_seed = 123
simulation_time = 60.0
arr_rate = 30
edge_nodes = [
    {"id": "edge-1", "cpu": 2, "mem": 256, "throughput": 2e6, "uplink_ms": 20, "downlink_ms": 20, "gw_id": "gw1"},
    {"id": "edge-2", "cpu": 2, "mem": 256, "throughput": 2e6, "uplink_ms": 25, "downlink_ms": 25, "gw_id": "gw2"},
]
cloud_nodes = {"id": "cloud", "cpu": 16, "mem": 16384, "throughput": 50e6, "uplink_ms": 120, "downlink_ms": 100}

def run_demo_for_policy(policy):
    random.seed(random_seed)
    env = simpy.Environment()
    bus = MockBus(env)
    sched = Scheduler(env, bus, edge_nodes, cloud_nodes, policy=policy)
    fp_db = fingerprint_dataset(300)
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

def run_code():
    run_all_policies
    
