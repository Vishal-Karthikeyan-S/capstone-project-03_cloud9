from flask import Flask, render_template, request, redirect, url_for, flash, Response
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io



app = Flask(__name__)
app.secret_key = "edge_project_demo"

# Dummy credentials
users = {"worker1": "1234", "worker2": "abcd"}

# Dummy material locations (Zone names for display)
items = {
    "M001": "Zone A - Near Assembly Line",
    "M002": "Zone B - Storage Area",
    "M003": "Zone C - Loading Dock"
}

# Coordinates for plotting
item_coords = {
    "M001": (3, 4),
    "M002": (7, 5),
    "M003": (4, 7)
}

# Gateways
gateways = [(2, 2), (8, 2), (5, 8)]


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



@app.route("/search", methods=["GET", "POST"])
def search():
    location = None
    if request.method == "POST":
        item_id = request.form["item_id"].strip()
        if item_id in items:
            location = items[item_id]   # Zone name shown here
        else:
            location = "Not Found"
    return render_template("search.html", location=location)



@app.route("/map")
def map_view():
    return render_template("resourcemap.html")

@app.route("/plot.png")
def plot_png():
    fig, ax = plt.subplots()

    # Plot gateways (red squares)
    for gx, gy in gateways:
        ax.plot(gx, gy, 'rs', markersize=10, label="Gateway")

    # Plot items (blue dots with IDs)
    for item, (x, y) in item_coords.items():
        ax.plot(x, y, 'bo')
        ax.text(x + 0.2, y, item)

    ax.set_title("Factory Resource Map")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Save figure to PNG
    output = io.BytesIO()
    plt.savefig(output, format='png')
    plt.close(fig)
    output.seek(0)
    return Response(output.getvalue(), mimetype='image/png')



@app.route("/itemfound")
def itemfound():
    return render_template("itemfound.html")


if __name__ == "__main__":
    app.run(debug=True)
