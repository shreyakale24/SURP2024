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

class Quaternion:
    def __init__(self, e1=0, e2=0, e3=0, n=1):
        self.e = np.array([e1, e2, e3])
        self.n = n

    @staticmethod
    def from_rotation_matrix(C21):
        n = np.sqrt(np.trace(C21) + 1) / 2
        e1 = (C21[1, 2] - C21[2, 1]) / (4 * n)
        e2 = (C21[2, 0] - C21[0, 2]) / (4 * n)
        e3 = (C21[0, 1] - C21[1, 0]) / (4 * n)
        return Quaternion(e1, e2, e3, n)

    def to_rotation_matrix(self):
        e = self.e
        n = self.n
        C21 = (2 * n**2 - 1) * np.eye(3) + 2 * np.outer(e, e) - 2 * n * aCross(e)
        return C21

    def rates(self, omega):
        d_quat = Quaternion()
        d_quat.e = 0.5 * (self.n * np.eye(3) + aCross(self.e)) @ omega
        d_quat.n = -0.5 * self.e @ omega
        return d_quat

    def multiply(self, other):
        eS = self.n * other.e + other.n * self.e + aCross(self.e) @ other.e
        nS = self.n * other.n - np.dot(self.e, other.e)
        return Quaternion(eS[0], eS[1], eS[2], nS)

    def conjugate(self):
        return Quaternion(-self.e[0], -self.e[1], -self.e[2], self.n)

    def as_array(self):
        return np.concatenate((self.e, [self.n]))

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


