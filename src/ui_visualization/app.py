import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import io, base64, json, os, sys
import matplotlib.pyplot as plt

# Set up paths before importing modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import material_locator as ml
import resouceAlloc_comm as ra

# Flask setup
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Login credentials
users = {
    "admin": "admin",
    "worker1": "1234",
    "worker2": "5678"
}

# ---------- Login ----------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users and password == users[username]:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("search"))
        else:
            flash("Invalid username or password", "error")
    return render_template("login.html")


# ---------- Logout ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- Search Page ----------
@app.route("/search", methods=["GET", "POST"])
def search():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    location = None
    searched_id = None
    latency = None
    error = None

    if request.method == "POST":
        searched_id = request.form.get("item_id").upper().strip()
        if searched_id in ml.mat_loc:
            material_pos = np.array(ml.mat_loc[searched_id])
            coarse_zone, _, _ = ml.get_coarse_location(material_pos)
            location = coarse_zone

            # Run Edge–Cloud simulation
            try:
                sim_result = ra.run_demo_for_policy("LatencyAware")
                latency = round(sim_result["avg_latency"], 3)
                error = round(sim_result["avg_loc_err"], 3)
            except Exception as e:
                print(f"Simulation error: {e}")
                latency = "N/A"
                error = "N/A"

        else:
            location = "Not Found"

    return render_template(
        "search.html",
        location=location,
        searched_id=searched_id,
        latency=latency,
        error=error
    )


# ---------- Map Visualization ----------
@app.route("/map/<item_id>")
def map_view(item_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if item_id not in ml.mat_loc:
        flash("Invalid Material ID")
        return redirect(url_for("search"))

    material_pos = np.array(ml.mat_loc[item_id])
    coarse_zone, _, _ = ml.get_coarse_location(material_pos, material_id=item_id)
    refined_location = ml.cloud_computation(material_pos, material_id=item_id)
    path_a, _ = ml.path_analysis(ml.user_pos, material_pos, item_id)

    # Plot everything
    plt.ioff()
    plt.clf()
    ml.plot_all(ml.mat_loc, material_pos, coarse_zone,
                refined_location, show_path=True,
                path_a=path_a, selected_mat_id=item_id)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return render_template("resourcemap.html",
                           plot_url=plot_url,
                           item_id=item_id,
                           coarse_zone=coarse_zone)


# ---------- Pick Material ----------
@app.route("/picked/<item_id>")
def picked(item_id):
    try:
        if item_id in ml.mat_loc:
            del ml.mat_loc[item_id]
            # Save to JSON file
            mat_loc_path = os.path.join(BASE_DIR, "mat_loc.json")
            with open(mat_loc_path, "w") as f:
                json.dump(ml.mat_loc, f)
        flash(f"Material {item_id} marked as picked up", "success")
    except Exception as e:
        flash(f"Error updating material list: {e}", "error")
    return redirect(url_for("search"))


# ---------- Debug Route (Optional) ----------
@app.route("/debug")
def debug_info():
    import os
    debug_data = {
        "current_dir": os.getcwd(),
        "app_dir": os.path.dirname(os.path.abspath(__file__)),
        "mat_loc_count": len(ml.mat_loc),
        "sample_materials": list(ml.mat_loc.keys())[:10],
        "edge_gateways": ml.edge_gateways
    }
    return f"<pre>{json.dumps(debug_data, indent=2)}</pre>"


# ---------- Run ----------
if __name__ == "__main__":
    print("\n" + "="*60)
    print(f"Flask server starting...")
    print(f"Materials loaded: {len(ml.mat_loc)}")
    print(f"Sample IDs: {', '.join(list(ml.mat_loc.keys())[:5])}")
    print("="*60 + "\n")
    app.run(debug=True)
