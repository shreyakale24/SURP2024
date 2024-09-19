import functions as f
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#-------------------------------------------------------------

# Part 1: DETUMBLE
# Total mass 
mtot = 640  # kg 
s = 2       # m 

# J
Jtum = mtot * s**2 / 6 * np.eye(3)

print('Detumble:')
print(f'Total mass = {mtot:.0f} kg')
print('Center of mass: [0, 0, 0] m')
print('Inertia matrix: [kg*m^2]')
print(Jtum)

# NORMAL OPERATION 

# Total mass (kg)
mbus = 500
msens = 100
mpan = 20

zdist_sens = 1.5  # m 

# Centroid 
zcent = zdist_sens * msens / mtot

print('Normal Operations:')
print(f'Total mass = {mtot:.0f} kg')
print(f'Center of mass: [0, 0, {zcent:.3f}] m')

# Inertia matrix 
busx = 2
buxy = 2
buxz = 2

sensx = 0.25
sensy = 0.25
sensz = 1

panx = 2
pany = 3
panz = 0.05

J1bus = mbus * busx**2 / 6 * np.eye(3)
r21bus = np.array([0, 0, zcent])
J2bus = J1bus - mbus * f.aCross(r21bus) @ f.aCross(r21bus)

J1pan = mpan / 12 * np.diag([(pany**2 + panz**2), (panx**2 + panz**2), (panx**2 + pany**2)])
r21pan1 = np.array([0, -2.5, zcent])
r21pan2 = np.array([0, 2.5, zcent])
J2pan1 = J1pan - mpan * f.aCross(r21pan1) @ f.aCross(r21pan1)
J2pan2 = J1pan - mpan * f.aCross(r21pan2) @ f.aCross(r21pan2)

J1sens = msens / 12 * np.diag([sensy**2 + sensz**2, sensx**2 + sensz**2, sensx**2 + sensy**2])
r21sens = np.array([0, 0, -1.5 + zcent])
J2sens = J1sens - msens * f.aCross(r21sens) @ f.aCross(r21sens)

j = J2bus + J2pan1 + J2pan2 + J2sens

print('Inertia matrix: [kg*m^2]')
print(j)

#-----------------------------------------------------------------------

# Constants
mu = 398600  # km^3/s^2
h = 53335.2  # km^2/s
ecc = 0
raan = 0 * np.pi / 180
inc = 98.43 * np.pi / 180
omega = 0 * np.pi / 180
nu = 0 * np.pi / 180
ta = nu

# Orbital parameters
a = h**2 / (mu * (1 - ecc**2))
orbital_period = 2 * np.pi * np.sqrt(a**3 / mu)

# Initial torque (torque-free scenario)
t = np.array([0, 0, 0])

# Compute initial position and velocity
r_ECI_0= np.array([7136.586951931760, 0, 0])
v_ECI_0 = np.array([0, -1.09562080449860,7.39274268014112])

# Compute initial F_LVLH basis vectors
z_LVLH = -r_ECI_0 / np.linalg.norm(r_ECI_0)
y_LVLH = -np.cross(r_ECI_0, v_ECI_0) / np.linalg.norm(np.cross(r_ECI_0, v_ECI_0))
x_LVLH = np.cross(y_LVLH, z_LVLH)

f_LVLH = np.array([x_LVLH, y_LVLH, z_LVLH])

# Initial conditions
phi_0 = 0
theta_0 = 0
psi_0 = 0
E_b_LVLH_0 = np.array([phi_0, theta_0, psi_0])

q_b_LVLH_0 = np.array([0, 0, 0, 1])

C_LVLH_eci_0 = f_LVLH
C_b_LVLH_0 = np.eye(3)
C_b_ECI_0 = f_LVLH

E_b_ECI_0 = f.C2EulerAngles(C_b_ECI_0)
q_b_ECI_0 = f.Quaternion.C2quat(C_b_ECI_0)
w_b_ECI_0 = np.array([0.001, -0.001, 0.002])

def dynamics(t, j, omega):
   # omega = y[:3]
   # dq = np.array(f.Quaternion.rates((4))) # Placeholder for quaternion rates
  #  dE = np.zeros(3)  # Placeholder for Euler angle rates
  #  domega = np.linalg.solve(j, t - f.aCross(omega) @ (j @ omega))
   # return np.concatenate((domega, dq, dE))
    domega = np.linalg.inv(j) @ (t - f.aCross(omega) @ j @ omega)
    return domega


def euler_rates(E, omega):
    phi, theta, psi = E
    p, q, r = omega
    
    # Assuming a specific order (3-2-1 or ZYX rotation)
    dE = np.array([
        p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r,
        np.cos(phi) * q - np.sin(phi) * r,
        np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r
    ])
    
    return dE
def kinematics(omega, q, E):
# Compute the derivative of the quaternion components
    dq = f.Quaternion.rates(q, omega)
    
    # Compute the derivative of the Euler Angle components
    dE = euler_rates(E, omega)
    
    return dq, dE
# Setup simulation
y0 = np.concatenate((w_b_ECI_0, q_b_ECI_0, E_b_ECI_0))
tspan = [0, orbital_period]

sol = solve_ivp(dynamics, tspan, y0, t_eval=np.linspace(tspan[0], tspan[1], 500))

# Extract results
omega = sol.y[:3, :]
quaternions = sol.y[3:7, :]
eulers = sol.y[7:10, :]

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(sol.t, omega.T)
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.title('Angular Velocities')
plt.legend(['$\omega_x$', '$\omega_y$', '$\omega_z$'])
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(sol.t, quaternions.T)
plt.xlabel('Time [s]')
plt.ylabel('Quaternions')
plt.title('Quaternion Components')
plt.legend(['$\epsilon_x$', '$\epsilon_y$', '$\epsilon_z$', '$\eta$'])
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(sol.t, eulers.T)
plt.xlabel('Time [s]')
plt.ylabel('Euler Angles [rad]')
plt.title('Euler Angles')
plt.legend(['$\phi$', '$\theta$', '$\psi$'])
plt.grid(True)

plt.tight_layout()
plt.show()
