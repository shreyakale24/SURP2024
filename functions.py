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

## for part 2 of simulation
def domega(T, J, omega):
    """Compute the derivative of the angular velocity."""
    # Compute the angular acceleration
    omega = np.array(omega)
    J_inv = np.linalg.inv(J)
    cross_term = aCross(omega) @ J @ omega
    domega = J_inv @ (T - cross_term)
    return domega

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
    def C2quat(C21):
        n = np.sqrt(np.trace(C21) + 1) / 2
        e1 = (C21[1, 2] - C21[2, 1]) / (4 * n)
        e2 = (C21[2, 0] - C21[0, 2]) / (4 * n)
        e3 = (C21[0, 1] - C21[1, 0]) / (4 * n)
        sdf = 4
        return Quaternion(e1, e2, e3, n)

    def quat2C(self):
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
    n = np.array(np.cross([0, 0, 1], hVec))
    NMag = np.linalg.norm(n)

    # RAAN
    if n[1] < 0: #should not be a constant
        raan = 2 * np.pi - np.arccos(n[0] / NMag)
        raan = np.degrees(raan)
    else:
        raan = np.arccos(n[0] / NMag)
        raan = np.degrees(raan)

    # eccentricity
    eVec = (1 / mu) * (np.cross(v, hVec) - (mu * r / rMag))
    ecc = np.linalg.norm(eVec)

    # argument of perigee
    if eVec[2] < 0:
        omega = 2 * np.pi - np.arccos(np.dot(n, eVec) / (NMag * ecc))
        omega = np.degrees(omega)
    else:
        omega = np.arccos(np.dot(n, eVec) / (NMag * ecc))
        omega = np.degrees(omega)

    # true anomaly
    if vr >= 0:
        theta = np.arccos(np.dot(eVec, r) / (ecc * rMag))
        theta = np.degrees(theta)
    else:
        theta = 2 * np.pi - np.arccos(np.dot(eVec, r) / (ecc * rMag))
        theta = np.degrees(theta)

    return h, inc, ecc, raan, omega, theta

## COEs to R and V
def coes2rv(a, ecc, inc, omega, raan, TA, mu):
    # angular momentum (h)
    h = np.sqrt(mu * a * (1 - ecc**2))
    #print(h)
    # Position in perifocal coordinates
    r_peri = (h**2 / mu) * (1 / (1 + ecc * np.cos(TA))) * np.array([np.cos(TA), np.sin(TA), 0])
   # print(r_peri)
    # Velocity in perifocal coordinates
    v_peri = (mu / h) * np.array([-np.sin(TA), ecc + np.cos(TA), 0])
   # print(v_peri)
    # perifocal to inertial matrix
    R3_raan = rotCz(np.radians(raan))
    R1_inc = rotCx(np.radians(inc))
    R3_omega = rotCz(np.radians(omega))
    Ceci2peri = np.dot(np.dot(R3_omega, R1_inc), R3_raan)
    
    # position and velocity to inertial coordinates
    r = np.dot(Ceci2peri.T, r_peri)
    v = np.dot(Ceci2peri.T, v_peri)
    
    return r, v

#lin alg ode solver in numpy

#idp solver in numpy 