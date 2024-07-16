import numpy as np

## cross vector
def aCross(a):
    crossVector = np.array([[0, -a[2], a[1]],
                            [a[2], 0, -a[0]],
                            [-a[1], a[0], 0]])
    return crossVector

## x rotation 
def rotCx(theta):
    Cx = np.array([[1, 0, 0],
                   [0, np.cos(theta), np.sin(theta)],
                   [0, -np.sin(theta), np.cos(theta)]])
    return Cx

## y rotation
def rotCy(theta):
    Cy = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    return Cy

## z rotation
def rotCz(theta):
    Cz = np.array([[np.cos(theta), np.sin(theta), 0],
                   [-np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    return Cz

## principle axis rotation
def principalAxis(C21):
    phi = np.arccos((np.trace(C21) - 1) / 2)

    a1 = (C21[1, 2] - C21[2, 1]) / (2 * np.sin(phi))
    a2 = (C21[2, 0] - C21[0, 2]) / (2 * np.sin(phi))
    a3 = (C21[0, 1] - C21[1, 0]) / (2 * np.sin(phi))

    a = np.array([a1, a2, a3])

    return a, phi

## zyx principle axis rotation
def zyx_principle(phi, theta, psi):
    C21 = rotCx(phi) @ rotCy(theta) @ rotCz(psi)
    a, phiPrinicple = principalAxis(C21)
    
    return a, phiPrinicple

## rotation maatrix to euler angle

def C2EulerAngles(C21):
    phi = np.arctan2(C21[1, 2], C21[2, 2])
    theta = -np.arcsin(C21[0, 2])
    psi = np.arctan2(C21[0, 1], C21[0, 0])

    return phi, theta, psi

## rotation matrix to quaternion
def C2quat(C21):
    n = np.sqrt(np.trace(C21) + 1) / 2
    e1 = (C21[1, 2] - C21[2, 1]) / (4 * n)
    e2 = (C21[2, 0] - C21[0, 2]) / (4 * n)
    e3 = (C21[0, 1] - C21[1, 0]) / (4 * n)
    
    E = np.array([e1, e2, e3, n])
    
    return E

## quaternion to rotation matrix
def quat2C(quat):
    e = quat[:3]
    n = quat[3]
    
    C21 = (2 * n**2 - 1) * np.eye(3) + (2 * np.outer(e, e)) - (2 * n * aCross(e))
    
    return C21

## quaternion rates
def quatRates(quat, omega):
    d_quat = np.zeros(4)
    
    e = quat[:3]
    n = quat[3]
    
    d_quat[:3] = 0.5 * (n * np.eye(3) + aCross(e)) @ omega
    d_quat[3] = -0.5 * e @ omega
    
    return d_quat

## quaternion multuplication
def quatMult(p, q):
    eS = p[3] * q[:3] + q[3] * p[:3] + aCross(p[:3]) @ q[:3]
    nS = p[3] * q[3] - np.dot(p[:3], q[:3])
    
    s = np.concatenate((eS, [nS]))
    
    return s

## quaternion conjugate
def quatConjugate(q):
    qStarE = -1 * q[:3]
    qStarN = q[3]
    qStar = np.concatenate((qStarE, [qStarN]))
    
    return qStar

## COE from R and V
import numpy as np

def COE(r, v, mu):
    rMag = np.linalg.norm(r)
    vMag = np.linalg.norm(v)

    # velocity
    vr = np.dot(r, v) / rMag

    # momentum
    hVec = np.cross(r, v)
    h = np.linalg.norm(hVec)

    # inclination
    inc = np.arccos(hVec[2] / h)
    inc = np.degrees(inc)

    # Node lines
    N = np.cross([0, 0, 1], hVec)
    NMag = np.linalg.norm(N)

    # RAAN
    if N[1] < 0:
        RAAN = 2 * np.pi - np.arccos(N[0] / NMag)
        RAAN = np.degrees(RAAN)
    else:
        RAAN = np.arccos(N[0] / NMag)
        RAAN = np.degrees(RAAN)

    # eccentricity
    eVec = (1 / mu) * (np.cross(v, hVec) - (mu * r / rMag))
    ecc = np.linalg.norm(eVec)

    # argument of perigee
    if eVec[2] < 0:
        omega = 2 * np.pi - np.arccos(np.dot(N, eVec) / (NMag * ecc))
        omega = np.degrees(omega)
    else:
        omega = np.arccos(np.dot(N, eVec) / (NMag * ecc))
        omega = np.degrees(omega)

    # true anomaly
    if vr >= 0:
        theta = np.arccos(np.dot(eVec, r) / (ecc * rMag))
        theta = np.degrees(theta)
    else:
        theta = 2 * np.pi - np.arccos(np.dot(eVec, r) / (ecc * rMag))
        theta = np.degrees(theta)

    # energy
    energy = (vMag**2 / 2) - (mu / rMag)

    # semi-major axis
    a = (h**2 / mu) * (1 / (1 - ecc**2))

    # ra and rp
    ra = (h**2 / mu) * (1 / (1 - ecc))
    rp = (h**2 / mu) * (1 / (1 + ecc))

    # period
    P = (2 * np.pi / np.sqrt(mu)) * a**(3 / 2)

    # mean anomaly
    Me = theta - ecc * np.sin(np.radians(theta))

    # eccentricity vector
    eccVec = eVec

    # time since periapse (tau)
    tau = (2 * np.pi / P) * (Me / (2 * np.pi))

    return h, inc, ecc, RAAN, omega, theta, a, tau, energy, Me, eccVec, ra, rp, P

## COEs to R and V
def coes2rv(a, ecc, inc, omega, OMEGA, TA, mu):
    # angular momentum (h)
    h = np.sqrt(mu * a * (1 - ecc**2))
    
    # Position in perifocal coordinates
    r_peri = (h**2 / mu) * (1 / (1 + ecc * np.cos(TA))) * np.array([np.cos(TA), np.sin(TA), 0])
    
    # Velocity in perifocal coordinates
    v_peri = (mu / h) * np.array([-np.sin(TA), ecc + np.cos(TA), 0])
    
    # perifocal to inertial matrix
    R3_RAAN = rotCz(np.radians(OMEGA))
    R1_inc = rotCx(np.radians(inc))
    R3_omega = rotCz(np.radians(omega))
    Ceci2peri = np.dot(np.dot(R3_omega, R1_inc), R3_RAAN)
    
    # position and velocity to inertial coordinates
    r = np.dot(Ceci2peri.T, r_peri)
    v = np.dot(Ceci2peri.T, v_peri)
    
    return r, v


