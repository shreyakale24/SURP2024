import functions as f
import numpy as np
import unittest 

class TestFunctions:
    def test_rotCx(self):
        assert np.array_equal(f.rotCx(0),np.identity(3))
        ##if f.rotCx(0)== np.eye
    def test_rotCy(self): 
        assert np.array_equal(f.rotCy(0),np.identity(3))
    def test_rotCz(self): 
        assert np.array_equal(f.rotCy(0),np.identity(3))
    def test_aCross(self):
        assert np.array_equal(f.aCross([1,2,3]), np.array([[0, -3, 2],
                            [3, 0, -1],
                            [-2, 1, 0]]))
    def test_principalAxis(self):
        assert np.array_equal(f.principalAxis([[0.3, 0.4,-0.1], [0.7, 0.8, 0.4], [-0.9, 0.3, 0.6]]), 
                              np.array([ 0.0534, -0.4270, -0.1601]) )
    def test_zyx_principle(self):
        phi, theta, psi = 0.1, 0.2, 0.3

        expected_a = 0.1886
        expected_phiPrinciple = 0.0275

        a, phiPrinciple = f.zyx_principle(phi, theta, psi)

        assert a == expected_a
        assert phiPrinciple== expected_phiPrinciple
 
        