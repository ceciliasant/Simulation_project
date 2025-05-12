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

    
    def dvx_dt(vx): return -np.sign(vx) * (u * vx**2) / m
    def dvz_dt(vz): return -g - np.sign(vz) * (u * vz**2) / m

    K1_x = vx * dt
    K1_z = vz * dt
    K1_vx = dvx_dt(vx) * dt
    K1_vz = dvz_dt(vz) * dt

    K2_x = (vx + K1_vx / 2) * dt
    K2_z = (vz + K1_vz / 2) * dt
    K2_vx = dvx_dt(vx + K1_vx / 2) * dt
    K2_vz = dvz_dt(vz + K1_vz / 2) * dt

    K3_x =(vx + K2_vx / 2) * dt
    K3_z = (vz + K2_vz / 2) * dt    
    K3_vx = dvx_dt(vx + K2_vx / 2) * dt
    K3_vz = dvz_dt(vz + K2_vz / 2) * dt

    K4_x = (vx + K3_vx) * dt
    K4_z = (vz + K3_vz) * dt
    K4_vx = dvx_dt(vx + K3_vx) * dt
    K4_vz = dvz_dt(vz + K3_vz) * dt

    state["x"] += (K1_x + 2*K2_x + 2*K3_x + K4_x) / 6
    state["z"] += (K1_z + 2*K2_z + 2*K3_z + K4_z) / 6
    state["vx"] += (K1_vx + 2*K2_vx + 2*K3_vx + K4_vx) / 6
    state["vz"] += (K1_vz + 2*K2_vz + 2*K3_vz + K4_vz) / 6

def get_simulation_data(state):
    return state["evolX"], state["evolZ"], state["evolvx"], state["evolvz"], state["time"]

if __name__ == "__main__":
    state, u, dt, tf, m, g = initialize()
    steps = int(tf / dt)
    for _ in range(steps):
        update(state, u, dt, m, g)
        observe(state, dt)

    print('-----------------------------------------')
    print("Runge-kutta method simulation")
    print('-----------------------------------------')

    print("Final position: (x, z) = ({:.2f}, {:.2f})".format(state["x"], state["z"]))
    print("Final velocity: (vx, vz) = ({:.2f}, {:.2f})".format(state["vx"], state["vz"]))
    print("Final time: {:.2f} seconds".format(state["time"][-1]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(state["evolX"],state["evolZ"], label="Projectile Path")
    plt.xlabel("x position")
    plt.ylabel("z position")
    plt.title("Object Trajectory with Drag- Runge-Kutta")
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