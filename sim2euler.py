import sys
import matplotlib.pyplot as plt
import numpy as np

def initialize(args=None):
    if args is None:
        if len(sys.argv) == 2:
            filename = sys.argv[1]
            try:
                with open(filename, "r") as f:
                    values = [float(line.strip()) for line in f.readlines()]
                    if len(values) >= 9:
                        x, z, vx, vz, u, dt, tf, m, g = values[:9]
                    else:
                        raise ValueError("File does not contain enough values.")
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
        elif len(sys.argv) >= 10:
            x, z, vx, vz, u, dt, tf, m, g = map(float, sys.argv[1:10])
        else:
            print("Missing input data. Using default values.")
            x, z, vx, vz, u, dt, tf, m, g = 0, 0, 10, 10, 0.5, 0.01, 5, 1, 9.8
    else:
        x, z, vx, vz, u, dt, tf, m, g = args

    state = {
        "x": x, "z": z, "vx": vx, "vz": vz,
        "evolX": [x], "evolZ": [z],
        "evolvx": [vx], "evolvz": [vz],
        "time": [0]
    }

    print("Initial Conditions:")
    print(f"x = {x} m")
    print(f"z = {z} m")
    print(f"vx = {vx} m/s")
    print(f"vz = {vz} m/s")
    print(f"Drag coefficient (u) = {u}")
    print(f"Time step (dt) = {dt} s")
    print(f"Total time (tf) = {tf} s")
    print(f"Mass (m) = {m} kg")
    print(f"Gravitational acceleration (g) = {g} m/s²")

    return state, u, dt, tf, m, g

def observe(state, dt):
    state["evolX"].append(state["x"])
    state["evolZ"].append(state["z"])
    state["evolvx"].append(state["vx"])
    state["evolvz"].append(state["vz"])
    state["time"].append(state["time"][-1] + dt)

def update(state, u, dt, m, g):
    x, z, vx, vz = state["x"], state["z"], state["vx"], state["vz"]
    
    # Euler forward method
    ax = -np.sign(vx) * (u * vx**2) / m
    az = -g - np.sign(vz) * (u * vz**2) / m

    nx = x + vx * dt 
    nz = z + vz * dt
    nvx = vx + ax * dt
    nvz = vz + az * dt

    state["x"] = nx
    state["z"] = nz
    state["vx"] = nvx
    state["vz"] = nvz

def get_simulation_data(state):
    return state["evolX"], state["evolZ"], state["evolvx"], state["evolvz"], state["time"]

if __name__ == "__main__":
    state, u, dt, tf, m, g = initialize()
    steps = int(tf / dt)
    for _ in range(steps):
        update(state, u, dt, m, g)
        observe(state, dt)

    state["evolX"], state["evolZ"], state["evolvx"], state["evolvz"], state["time"] = get_simulation_data(state)
    
    print('-----------------------------------------')
    print("Euler method simulation")
    print('-----------------------------------------')
    print("Final position: (x, z) = ({:.2f}, {:.2f})".format(state["x"], state["z"]))
    print("Final velocity: (vx, vz) = ({:.2f}, {:.2f})".format(state["vx"], state["vz"]))
    print("Final time: {:.2f} seconds".format(state["time"][-1]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(state["evolX"],state["evolZ"], label="Projectile Path")
    plt.xlabel("x position")
    plt.ylabel("z position")
    plt.title("Object Trajectory with Drag- Euler")
    plt.axhline(0, color="gray", linestyle="--")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(state["time"], state["evolvx"], label="Vx (horizontal)", linestyle="--")
    plt.plot(state["time"], state["evolvz"], label="Vz (vertical)", linestyle="-")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity Evolution")
    plt.legend()

    plt.tight_layout()
    plt.show()
    