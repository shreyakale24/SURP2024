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

#-------------------------------------------------------------

#Part 2
# Spacecraft Orbit Properties (given)
mu = 398600  # km^3/s^2
h = 53335.2  # km^2/s
ecc = 0  # none
raan = 0 * np.pi / 180  # radians
inc = 98.43 * np.pi / 180  # radians
omega = 0 * np.pi / 180  # radians
nu = 0 * np.pi / 180  # radians (true anomaly)
ta = nu  # radians

# Semi-major axis
a = h**2 / mu / (1 - ecc**2)

# Orbital period
orbital_period = 2 * np.pi * np.sqrt(a**3 / mu)

# Torque-free scenario (Given)
t = np.array([0, 0, 0])

# Set/Compute initial conditions
# Initial orbital position and velocity
r_ECI_0= np.array([7136.586951931760, 0, 0])
v_ECI_0 = np.array([0, -1.09562080449860,7.39274268014112])

# Compute initial F_LVLH basis vectors in F_ECI components
z_LVLH = -r_ECI_0 / np.linalg.norm(r_ECI_0)
y_LVLH = -np.cross(r_ECI_0, v_ECI_0) / np.linalg.norm(np.cross(r_ECI_0, v_ECI_0))
x_LVLH = np.cross(y_LVLH, z_LVLH)

f_LVLH = np.array([x_LVLH, y_LVLH, z_LVLH])

# Initial quaternion relating F_body and F_LVLH (given)
q_b_LVLH_0 = np.array([0, 0, 0, 1])

phi_0 = 0
theta_0 = 0
psi_0 = 0
E_b_LVLH_0 = np.array([phi_0, theta_0, psi_0])

# Initial Quaternion relating F_body and F_LVLH (given)
q_b_LVLH_0 = np.array([0, 0, 0, 1])

# Compute initial C_LVLH_ECI_0, C_b_LHVL_0, and C_b_ECI_0 rotation matrices
c_LVLH_ECI_0 = f_LVLH
C_b_LVLH_0 = np.eye(3)
C_b_ECI_0 = f_LVLH

# Initial Euler angles relating body to ECI
E_b_ECI_0 = f.C2EulerAngles(C_b_ECI_0)

# Initial quaternion relating body to E
q_b_ECI_0 = f.Quaternion.C2quat(C_b_ECI_0)

# Initial body rates of spacecraft (given)
w_b_ECI_0 = np.array([0.001, -0.001, 0.002]);

def dynamics(t, y):
    omega = y[:3]
    dq = np.zeros(4)  
    dE = np.zeros(3)  #
    domega = np.linalg.solve(j, t - f.aCross(omega) @ (J @ omega))
    return np.concatenate((domega, dq, dE))

# Function to compute quaternion derivative
def quat_derivative(q, w):
    q0, q1, q2, q3 = q
    wx, wy, wz = w
    dqdt = 0.5 * np.array([
        -q1 * wx - q2 * wy - q3 * wz,
         q0 * wx + q2 * wz - q3 * wy,
         q0 * wy - q1 * wz + q3 * wx,
         q0 * wz + q1 * wy - q2 * wx
    ])
    return dqdt

# Euler's rotational equations of motion
def rotational_dynamics(t, y, J, T):
    w = y[0:3]     # Angular velocities
    q = y[3:7]     # Quaternions
    
    # Inverse of inertia matrix (assuming it's diagonal)
    J_inv = np.linalg.inv(J)
    
    # Angular velocity derivative (Euler's equation)
    w_dot = J_inv @ (T - np.cross(w, J @ w))
    
    # Quaternion derivative
    q_dot = quat_derivative(q, w)
    
    return np.hstack([w_dot, q_dot])

# Initial conditions
w_b_ECI_0 = np.array([0.001, -0.001, 0.002])  # Angular velocity
q_b_ECI_0 = np.array([0, 0, 0, 1])  # Initial quaternion (no initial rotation)

# Combine angular velocity and quaternion into initial state vector
y0 = np.hstack([w_b_ECI_0, q_b_ECI_0])

# Time span for simulation
t_span = (0, orbital_period)
t_eval = np.linspace(0, orbital_period, 100)  # Time points for evaluation

# Solve the rotational dynamics using solve_ivp
sol = solve_ivp(rotational_dynamics, t_span, y0, t_eval=t_eval, args=(j, t))

# Extract angular velocities and quaternions from the solution
omegas = sol.y[0:3].T  # Angular velocities (wx, wy, wz)
quats = sol.y[3:7].T   # Quaternions (q0, q1, q2, q3)

#something is wrong! 
#get euler funtion
eulers = np.zeros((len(t_eval), 3)) 

# Plotting

plt.figure(1)
plt.suptitle('Part 2: Body to ECI Kinematics and Dynamics')

# Angular Velocity Plot
plt.subplot(3, 1, 1)
plt.plot(t_eval, omegas[:, 0], label='ω_x', linewidth=1.5)
plt.plot(t_eval, omegas[:, 1], label='ω_y', linewidth=1.5)
plt.plot(t_eval, omegas[:, 2], label='ω_z', linewidth=1.5)
plt.title('Angular Velocity')
plt.xlabel('time [s]')
plt.ylabel('ω [rad/s]')
plt.grid(True)
plt.legend()

# Quaternion Plot
plt.subplot(3, 1, 2)
plt.plot(t_eval, quats[:, 0], label='ε₁', linewidth=1.5)
plt.plot(t_eval, quats[:, 1], label='ε₂', linewidth=1.5)
plt.plot(t_eval, quats[:, 2], label='ε₃', linewidth=1.5)
plt.plot(t_eval, quats[:, 3], label='η', linewidth=1.5)
plt.xlabel('time [s]')
plt.ylabel('parameters')
plt.title('Quaternions')
plt.grid(True)
plt.legend()

# Euler Angles Plot (Placeholder: Implement quaternion to Euler angle conversion)
plt.subplot(3, 1, 3)
plt.plot(t_eval, eulers[:, 0], label='φ', linewidth=1.5)
plt.plot(t_eval, eulers[:, 1], label='θ', linewidth=1.5)
plt.plot(t_eval, eulers[:, 2], label='ψ', linewidth=1.5)
plt.xlabel('time [s]')
plt.ylabel('Angle [rad]')
plt.title('Euler Angles')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#-------------------------------------------------------------
#FORGOT PART 3 WHOOPS 
a = 4
#-------------------------------------------------------------
# Part 4: Defining new initial conditions

j = j  # Inertia matrix
T_c = np.array([0, 0, 0])  # Commanded torque
T_d = T_c  # Disturbance torque

# Convert rotation matrix to Euler angles and quaternions
E_b_LVLH_0 = f.C2EulerAngles(C_b_LVLH_0)
q_b_LVLH_0 = f.C2quat(C_b_LVLH_0)

#FIGURE OUT SIM :(((((

# Simulated data ( will replace with actual simulation outputs)
part4out_tout = np.linspace(0, 100, 100)  # time steps
w_b_ECI = np.random.rand(100, 3)  # Angular velocities in ECI frame (replace with real data)
E_b_ECI = np.random.rand(100, 3)  # Euler angles in ECI frame (replace with real data)
q_b_ECI = np.random.rand(100, 4)  # Quaternions in ECI frame (replace with real data)

w_b_LVLH = np.random.rand(100, 3)  # Angular velocities in LVLH frame (replace with real data)
E_b_LVLH = np.random.rand(100, 3)  # Euler angles in LVLH frame (replace with real data)
q_b_LVLH = np.random.rand(100, 4)  # Quaternions in LVLH frame (replace with real data)

# Plotting results
# Plot Angular Velocities, Euler Angles, and Quaternions (Body to ECI)
plt.figure(41)

plt.subplot(3, 1, 1)
plt.plot(part4out_tout, w_b_ECI, linewidth=1.5)
plt.xlabel('time [s]')
plt.ylabel('ω [rad/sec]')
plt.title('Angular Velocities')
plt.grid(True)
plt.legend(['ω_x', 'ω_y', 'ω_z'])

plt.subplot(3, 1, 2)
plt.plot(part4out_tout, E_b_ECI, linewidth=1.5)
plt.xlabel('time [s]')
plt.ylabel('Euler Angles [rad]')
plt.title('Euler Angles')
plt.grid(True)
plt.legend(['φ', 'θ', 'ψ'])

plt.subplot(3, 1, 3)
plt.plot(part4out_tout, q_b_ECI, linewidth=1.5)
plt.xlabel('time [s]')
plt.ylabel('Quaternions')
plt.title('Quaternions')
plt.grid(True)
plt.legend(['ε_x', 'ε_y', 'ε_z', 'η'])
plt.suptitle('Body to ECI Kinematics')

# Plot Angular Velocities, Euler Angles, and Quaternions (Body to LVLH)
plt.figure(42)

plt.subplot(3, 1, 1)
plt.plot(part4out_tout, w_b_LVLH, linewidth=1.5)
plt.xlabel('time [s]')
plt.ylabel('ω [rad/sec]')
plt.title('Angular Velocities')
plt.grid(True)
plt.legend(['ω_x', 'ω_y', 'ω_z'])

plt.subplot(3, 1, 2)
plt.plot(part4out_tout, E_b_LVLH, linewidth=1.5)
plt.xlabel('time [s]')
plt.ylabel('Euler Angles [rad]')
plt.title('Euler Angles')
plt.grid(True)
plt.legend(['φ', 'θ', 'ψ'])

plt.subplot(3, 1, 3)
plt.plot(part4out_tout, q_b_LVLH, linewidth=1.5)
plt.xlabel('time [s]')
plt.ylabel('Quaternions')
plt.title('Quaternions')
plt.grid(True)
plt.legend(['ε_x', 'ε_y', 'ε_z', 'η'])
plt.suptitle('Body to LVLH Kinematics')

plt.show()

import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------

# Part 5: Initial conditions and simulation setup

# Gain (experimental)
k = -0.2  

# Initial conditions
w_b_ECI_tumb = np.array([-0.05, 0.03, 0.2])  # Angular velocity in ECI frame
# Time span based on number of orbits
n_revs = 5
tspan = n_revs * orbital_period

#FIGURE OUT SIM :(((((

# Simulated data (will replace with actual simulation outputs)
part5out_tout = np.linspace(0, tspan, 100)  # time steps
w_b_ECI = np.random.rand(100, 3)  # Angular velocities in ECI frame (replace with real data)
q_b_ECI = np.random.rand(100, 4)  # Quaternions in ECI frame (replace with real data)
E_b_ECI = np.random.rand(100, 3)  # Euler angles in ECI frame (replace with real data)

# Compute control torque
T_cout = k * w_b_ECI  # Control torque using gain and angular velocity

# Plot Angular Velocities, Quaternions, and Euler Angles

plt.figure(51)

# Angular velocities
plt.subplot(3, 1, 1)
plt.plot(part5out_tout, w_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('ω [rad/s]')
plt.title("Angular Velocity")
plt.grid(True)
plt.legend(["ω_x", 'ω_y', 'ω_z'])

# Quaternion components
plt.subplot(3, 1, 2)
plt.plot(part5out_tout, q_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Quaternion')
plt.title("Quaternion Components")
plt.grid(True)
plt.legend(['ε_x', 'ε_y', 'ε_z', 'η'])

# Euler angles
plt.subplot(3, 1, 3)
plt.plot(part5out_tout, E_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Euler Angles [rad]')
plt.title("Euler Angles")
plt.grid(True)
plt.legend(['φ', 'θ', 'ψ'])

plt.tight_layout()
plt.show()

# Plot Control Torque Components

plt.figure(52)
plt.plot(part5out_tout, T_cout, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Control Torque [N*m]')
plt.title("Control Torque Components")
plt.grid(True)
plt.legend(["T_x", 'T_y', 'T_z'])
plt.show()

#-------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Constants
Is_unit = 1.2  # kg/m^2
It_unit = 0.6  # kg/m^2
m_wheel = 1  # kg

# Orbital period (example value, replace with actual period)
orbital_period = 5400  # seconds, e.g., for a low Earth orbit

# New total inertia matrix considering the wheels
Jnew = j + (2*It_unit + Is_unit + 2*m_wheel) * np.eye(3)

# Inertia of wheels in wheel space
Is = np.diag([1.2, 1.2, 1.2])

# Damping ratio and settling time
zeta = 0.65
ts = 100  # seconds

# Natural frequency (omegan) calculation
omegan = np.log(0.02*np.sqrt(1 - zeta**2)) / (-ts * zeta * np.sqrt(1 - zeta**2))

# Gains Kd and Kp
Kd = 2 * zeta * omegan * Jnew
Kp = 2 * omegan**2 * Jnew

# Initial conditions for wheel speeds and quaternion command
Omega_0 = np.array([0, 0, 0])  # Wheel speeds in rad/s
q_c = np.array([0, 0, 0, 1])  # Commanded quaternion (LVLH to Body)

# Time span for the first simulation (300 seconds)
tspan = 300  # seconds

#FIGURE OUT SIM :(((((

part6out_tout = np.linspace(0, tspan, 100)  # time steps


'''E_b_ECI = 
q_b_ECI =
w_b_ECI = 
Mc = 
Omega = ''' 

# Body to ECI plots (First 300 seconds)

plt.figure(61)
# Euler angles plot
plt.subplot(3, 1, 1)
plt.plot(part6out_tout, E_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Euler Angles [rad]')
plt.grid(True)
plt.legend(['φ', 'θ', 'ψ'])

# Quaternion plot
plt.subplot(3, 1, 2)
plt.plot(part6out_tout, q_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Quaternions')
plt.grid(True)
plt.legend(['ε_x', 'ε_y', 'ε_z', 'η'])

# Angular velocity plot
plt.subplot(3, 1, 3)
plt.plot(part6out_tout, w_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('ω [rad/s]')
plt.grid(True)
plt.legend(['ω_x', 'ω_y', 'ω_z'])

plt.suptitle('ECI to Body (First 300 Seconds)')
plt.tight_layout()
plt.show()

# LVLH to Body plots (First 300 seconds)

plt.figure(62)
# Euler angles plot
plt.subplot(3, 1, 1)
plt.plot(part6out_tout, E_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Euler Angles [rad]')
plt.grid(True)
plt.legend(['φ', 'θ', 'ψ'])

# Quaternion plot
plt.subplot(3, 1, 2)
plt.plot(part6out_tout, q_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Quaternions')
plt.grid(True)
plt.legend(['ε_x', 'ε_y', 'ε_z', 'η'])

# Angular velocity plot
plt.subplot(3, 1, 3)
plt.plot(part6out_tout, w_b_ECI, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('ω [rad/s]')
plt.grid(True)
plt.legend(['ω_x', 'ω_y', 'ω_z'])

plt.suptitle('LVLH to Body (First 300 Seconds)')
plt.tight_layout()
plt.show()

# Control Moment plot (First 300 seconds)

plt.figure(63)
plt.plot(part6out_tout, Mc, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('M_c [N*m]')
plt.grid(True)
plt.title('Command Moment (First 300 Seconds)')
plt.legend(['M_x', 'M_y', 'M_z'])
plt.show()

# Wheel speeds plot (First 300 seconds)

plt.figure(64)
plt.plot(part6out_tout, OMEGA, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Wheel Speed [rad/s]')
plt.grid(True)
plt.title('Wheel Speeds (First 300 Seconds)')
plt.legend(['ω_{w1}', 'ω_{w2}', 'ω_{w3}'])
plt.show()

# Extended simulation for 5 orbits
tspan2 = 5 * orbital_period  # time span for five orbits

# Simulated data for five orbits 
part6out2_tout = np.linspace(0, tspan2, 500)  # time steps
'''E_b_ECI_2 = 
q_b_ECI_2 = 
w_b_ECI_2 = 
Mc_2 =
Omega_2 =
'''
# Body to ECI plots (Five orbits)

plt.figure(65)
# Euler angles plot
plt.subplot(3, 1, 1)
plt.plot(part6out2_tout, E_b_ECI_2, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Euler Angles [rad]')
plt.grid(True)
plt.legend(['φ', 'θ', 'ψ'])

# Quaternion plot
plt.subplot(3, 1, 2)
plt.plot(part6out2_tout, q_b_ECI_2, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Quaternions')
plt.grid(True)
plt.legend(['ε_x', 'ε_y', 'ε_z', 'η'])

# Angular velocity plot
plt.subplot(3, 1, 3)
plt.plot(part6out2_tout, w_b_ECI_2, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('ω [rad/s]')
plt.grid(True)
plt.legend(['ω_x', 'ω_y', 'ω_z'])

plt.suptitle('ECI to Body (Five Orbits)')
plt.tight_layout()
plt.show()

# LVLH to Body plots (Five orbits)

plt.figure(66)
# Euler angles plot
plt.subplot(3, 1, 1)
plt.plot(part6out2_tout, E_b_ECI_2, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Euler Angles [rad]')
plt.grid(True)
plt.legend(['φ', 'θ', 'ψ'])

# Quaternion plot
plt.subplot(3, 1, 2)
plt.plot(part6out2_tout, q_b_ECI_2, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Quaternions')
plt.grid(True)
plt.legend(['ε_x', 'ε_y', 'ε_z', 'η'])

# Angular velocity plot
plt.subplot(3, 1, 3)
plt.plot(part6out2_tout, w_b_ECI_2, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('ω [rad/s]')
plt.grid(True)
plt.legend(['ω_x', 'ω_y', 'ω_z'])

plt.suptitle('LVLH to Body (Five Orbits)')
plt.tight_layout()
plt.show()

# Control Moment plot (Five orbits)

plt.figure(67)
plt.plot(part6out2_tout, Mc_2, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('M_c [N*m]')
plt.grid(True)
plt.title('Command Moment (Five Orbits)')
plt.legend(['M_x', 'M_y', 'M_z'])
plt.show()

# Wheel speeds plot (Five orbits)

plt.figure(68)
plt.plot(part6out2_tout, OMEGA_2, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Wheel Speed')
plt.grid(True)
plt.title('Wheel Speed [rad/s]')
plt.legend(['ω_{w1}', 'ω_{w2}', 'ω_{w3}'])
plt.show()

#----------------------------------------------------------

