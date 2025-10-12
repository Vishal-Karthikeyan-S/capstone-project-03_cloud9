from flask import Flask, jsonify, request, send_file, render_template
from threading import Thread
import matplotlib.pyplot as plt
import os
import sys
from resourceAlloc_comm import (
    run_demo_for_policy,
    edge_nodes,
    cloud_nodes,
    simulation_time,
    arr_rate
)

app = Flask(__name__)

# Storage for simulation results
simulation_results = {}

@app.route("/")
def home():
    # Load your main dashboard (make sure you have templates/resource_allocation.html)
    return render_template("resource_allocation.html")

@app.route("/api/run-simulation", methods=["POST"])
def run_simulation():
    """Run a single simulation for the selected policy."""
    try:
        data = request.get_json()
        policy = data.get("policy", "FCFS")

        def background_task():
            # Run simulation in background to avoid blocking
            result = run_demo_for_policy(policy)
            simulation_results[policy] = result

        Thread(target=background_task).start()

        return jsonify({
            "success": True,
            "message": f"Simulation for {policy} started successfully."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/run-all-policies", methods=["POST"])
def run_all_policies():
    """Run all available scheduling policies."""
    try:
        policies = ["FCFS", "RoundRobin", "Priority", "Random"]

        def background_task():
            for policy in policies:
                result = run_demo_for_policy(policy)
                simulation_results[policy] = result

        Thread(target=background_task).start()

        return jsonify({
            "success": True,
            "message": "All policies started successfully."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/simulation-plot.png")
def simulation_plot():
    """Generate and return the latest simulation plot as an image."""
    if not simulation_results:
        return jsonify({"error": "No simulation results available yet."}), 404

    # Create plot
    plt.figure(figsize=(8, 5))
    for policy, result in simulation_results.items():
        # Expecting result to have 'x' and 'y' data (customize if needed)
        if isinstance(result, dict) and "x" in result and "y" in result:
            plt.plot(result["x"], result["y"], label=policy)
        else:
            plt.plot([], [], label=policy)  # placeholder if data missing

    plt.title("Simulation Results by Policy")
    plt.xlabel("Time (s)")
    plt.ylabel("Performance Metric")
    plt.legend()
    plt.grid(True)

    # Save image
    plot_path = "static/simulation_plot.png"
    os.makedirs("static", exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    return send_file(plot_path, mimetype="image/png")


@app.route("/api/simulation-status", methods=["GET"])
def simulation_status():
    """Check which policies have finished running."""
    status = {p: "completed" for p in simulation_results.keys()}
    return jsonify(status)


if __name__ == "__main__":
    app.run(debug=True)
