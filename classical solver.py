import numpy as np
import matplotlib.pyplot as plt

def advection_solver(c, a, d, nx, nt, dt):
    """
    Solve the advection equation with periodic boundary conditions.

    Parameters:
        c: speed
        a: parameter in the initial condition. It has to be large
           to satisfy the periodic boundary conditions. In the paper
           it's set to 20
           
        nx: number of spatial grid points
        nt: time steps
        dt: time step size
    """
    dx = d / nx
    x = np.linspace(0, d, nx, endpoint=False)

    # Initial condition
    u = np.exp(-a * (x - d / 3)**2)

    # Solution array
    u_new = np.zeros_like(u)

    # Storage for solutions at intervals
    solutions = [u.copy()]
    intervals = nt/10

    # Main time-stepping loop
    for n in range(nt):
        # Periodic boundary conditions
        u_new[0] = u[0] - c * dt / dx * (u[0] - u[-1])
        for i in range(1, nx):
            u_new[i] = u[i] - c * dt / dx * (u[i] - u[i - 1])
        u[:] = u_new

        # Store solution at intervals
        if (n + 1) % intervals == 0:
            solutions.append(u.copy())

    return x, solutions

# Parameters
c = 1.0   # Wave speed
a = 100.0 # Parameter in the initial condition
d = 4.0   # Domain size
nx = 2**6 # Number of spatial points
nt = 200  # Number of time steps
dt = 0.01 # Time step size
dx = d/nx
print('dx/dt = ', dx/dt)

# Solve the equation
x, solutions = advection_solver(c, a, d, nx, nt, dt)

# Plot the result
plt.figure(figsize=(10, 6))
for i, u in enumerate(solutions):
    plt.plot(x, u, label=f"t = {i * (nt // 10) * dt:.2f}")
plt.title("Advection Equation Solution")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.grid()
plt.show()
