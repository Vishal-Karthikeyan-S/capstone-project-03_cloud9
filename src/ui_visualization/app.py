from flask import Flask, render_template, request, redirect, url_for, flash, Response
import matplotlib.pyplot as plt
import io

import resource_locator as rl

app = Flask(__name__)
app.secret_key = "edge_project_demo"

# Dummy credentials
users = {"worker1": "1234", "worker2": "abcd"}

# ---------------- Login ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            flash(f"Welcome {username}!", "success")
            return redirect(url_for("search"))
        else:
            flash("Invalid Username or Password", "error")
    return render_template("login.html")


# ---------------- Search ----------------
@app.route("/search", methods=["GET", "POST"])
def search():
    location = None
    material_id = None
    if request.method == "POST":
        material_id = request.form["item_id"].strip().upper()
        
        # ✅ Check if this material exists in real data
        if material_id in rl.mat_loc:
            tag_pos = rl.mat_loc[material_id]
            coarse_zone, max_rssi, rssi_details = rl.get_coarse_location(rl.np.array(tag_pos))
            location = f"{coarse_zone} (approx.)"
        else:
            location = "Material Not Found"
            
    return render_template("search.html", location=location, material_id=material_id)


# ---------------- Map ----------------
@app.route("/map")
def map_view():
    return render_template("resourcemap.html")


@app.route("/plot.png")
def plot_png():
    fig, ax = plt.subplots()

    # ✅ Gateways (red squares)
    for zone, pos in rl.edge_gateways.items():
        ax.plot(pos[0], pos[1], 'rs', markersize=8)
        ax.text(pos[0] + 1, pos[1], zone, fontsize=8)

    # ✅ Materials (blue dots)
    for mat_id, (x, y) in rl.mat_loc.items():
        ax.plot(x, y, 'bo')
        ax.text(x + 1, y + 1, mat_id, fontsize=6)

    ax.set_title("Factory Resource Map (Live Layout)")
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    output = io.BytesIO()
    plt.savefig(output, format='png', bbox_inches="tight")
    plt.close(fig)
    output.seek(0)
    return Response(output.getvalue(), mimetype='image/png')


# ---------------- Item Found ----------------
@app.route("/itemfound")
def itemfound():
    return render_template("itemfound.html")


if __name__ == "__main__":
    app.run(debug=True)
