import functions as f
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
mu = 398600  # km^3/s^2
h = 53335.2  # km^2/s
ecc = 0      # None
raan = 0     # radians
inc = np.radians(98.43)  # radians
omega = 0    # radians
ta = 0       # radians (True anomaly)

# Semi-major axis and orbital period
a = h**2 / mu / (1 - ecc**2)
orbital_period = 2 * np.pi * np.sqrt(a**3 / mu)

# Torque free scenario (Given)
t = np.array([0, 0, 0])


# Get initial orbital position and velocity in ECI frame
r_ECI_0= np.array([7136.586951931760, 0, 0])
v_ECI_0 = np.array([0, -1.09562080449860,7.39274268014112])

# Initial F_LVLH basis vectors in F_ECI components
z_LVLH = -r_ECI_0 / np.linalg.norm(r_ECI_0)
y_LVLH = -np.cross(r_ECI_0, v_ECI_0) / np.linalg.norm(np.cross(r_ECI_0, v_ECI_0))
x_LVLH = np.cross(y_LVLH, z_LVLH)

f_LVLH = np.vstack([x_LVLH, y_LVLH, z_LVLH])

# Initial Euler angles relating F_body and F_LVLH
phi_0 = 0
theta_0 = 0
psi_0 = 0
E_b_LVLH_0 = np.array([phi_0, theta_0, psi_0])

# Initial Quaternion relating F_body and F_LVLH
q_b_LVLH_0 = np.array([0, 0, 0, 1])

# Compute initial rotation matrices
C_LVLH_ECI_0 = f_LVLH
C_b_LVLH_0 = np.eye(3)
C_b_ECI_0 = f_LVLH


# Initial Euler angles relating body to ECI
E_b_ECI_0 = f.C2EulerAngles(C_b_ECI_0)

# Initial quaternion relating body to ECI
q_b_ECI_0 = f.Quaternion.C2quat(C_b_ECI_0)

# Initial body rates of spacecraft (Given)
w_b_ECI_0 = np.array([0.001, -0.001, 0.002])

# Simulation time span
tspan = orbital_period

def simulate(t_span, y0, w_b_ECI):
    # Use solve_ivp to integrate the quaternion dynamics over time
    result = solve_ivp(fun=f.Quaternion.rates, t_span=t_span, y0=y0, args=(w_b_ECI,), dense_output=True)
    return result

# sim 

# Outputs of simulation (placeholders for actual outputs)
print("Omegas (Angular velocities):", omegas)
print("Euler angles:", eulers)
print("Quaternions:", quats)
