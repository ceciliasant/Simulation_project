import sim2euler
import sim2runge
import matplotlib.pyplot as plt

import pandas as pd
from tabulate import tabulate

initial_conditions = initial_conditions = [
    (0,   0,   10,  10, 0.5, 0.01, 5, 1, 9.8),   # standard dt
    (0,   0,   10,  10, 0.5, 0.001,5, 1, 9.8),   # smaller dt
    (0,   0,   10,  10, 0.5, 0.1,  5, 1, 9.8),   # larger dt
    (0,   0,   100, 100,0.5, 0.01, 5, 1, 9.8),   # very high speed
    (0,   0,   100, 100,0.5, 0.001, 5, 1, 9.8),   # very high speed
    (0,   0,   10,  10, 0.5, 0.01, 100, 1, 9.8),   # standard dt
]

results = []
for i, initial_args in enumerate(initial_conditions):
    
    state_euler, u, dt, tf, m, g = sim2euler.initialize(initial_args)
    state_runge, _, _, _, _, _ = sim2runge.initialize(initial_args)

    steps = int(tf / dt)
    for _ in range(steps):
        sim2euler.update(state_euler, u, dt, m, g)
        sim2euler.observe(state_euler, dt)

        sim2runge.update(state_runge, u, dt, m, g)
        sim2runge.observe(state_runge, dt)

    x_e, z_e, vx_e, vz_e, t_e = sim2euler.get_simulation_data(state_euler)
    x_r, z_r, vx_r, vz_r, t_r = sim2runge.get_simulation_data(state_runge)

    xf_e, zf_e = x_e[-1], z_e[-1]
    xf_r, zf_r = x_r[-1], z_r[-1]
    vxe, vze = vx_e[-1], vz_e[-1]
    vxr, vzr = vx_r[-1], vz_r[-1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Simulation Comparison: dt={dt}, vx0={initial_args[2]}, vz0={initial_args[3]}")

    # Trajectory
    ax1 = axs[0]
    ax1.plot(x_e, z_e, '--', label="Euler")
    ax1.plot(x_r, z_r,  '-', label="RK4")
    ax1.scatter([xf_e, xf_r], [zf_e, zf_r], 
                color=['blue','orange'], marker='o')
    ax1.annotate(f"E({xf_e:.1f},{zf_e:.1f})", (xf_e, zf_e),
                 textcoords="offset points", xytext=(5,-10))
    ax1.annotate(f"R({xf_r:.1f},{zf_r:.1f})", (xf_r, zf_r),
                 textcoords="offset points", xytext=(5,5))
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.legend()
    ax1.grid(True)

    # Velocity
    ax2 = axs[1]
    ax2.plot(t_e, vx_e, '--', label="Vx Euler")
    ax2.plot(t_r, vx_r,  '-', label="Vx RK4")
    ax2.plot(t_e, vz_e, '--', label="Vz Euler")
    ax2.plot(t_r, vz_r,  '-', label="Vz RK4")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("v (m/s)")
    ax2.legend()
    ax2.grid(True)

    results.append({
        "Cenário": i+1,
        "dt": dt,
        "vx0": initial_args[2],
        "vz0": initial_args[3],
        "Euler x_final": x_e[-1],
        "Euler z_final": z_e[-1],
        "RK4 x_final": x_r[-1],
        "RK4 z_final": z_r[-1],
        "Euler vx_final": vx_e[-1],
        "Euler vz_final": vz_e[-1],
        "RK4 vx_final": vx_r[-1],
        "RK4 vz_final": vz_r[-1],
    })
# print a table with results
df = pd.DataFrame(results)
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

plt.tight_layout()
plt.show()


