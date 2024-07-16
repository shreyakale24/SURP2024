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

## euler angles FROM rotation maatrix 

## quaternion FROM rotation matrix

## rotation matrix FROM quaternion 


