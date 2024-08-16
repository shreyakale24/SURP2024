import functions as f
import numpy as np
import unittest 

class TestRotations(unittest.TestCase):
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

class TestPrincipalAxis(unittest.TestCase):
    def test_principalAxis(self):
        C21 = np.array([[0.9027, 0.1830, 0.3894], 
                        [-0.1596, 0.9829, 0.0920], 
                        [0.3996, -0.0209, 0.9165]])
        
        expected_a = np.array([0.1302, 0.0118, 0.3950])
        expected_phi =  0.4486
        
        result_a, result_phi = f.principalAxis(C21)
        
        np.testing.assert_array_almost_equal(result_a, expected_a, decimal=4)
        self.assertAlmostEqual(result_phi, expected_phi, places=4)

    def test_zyx_principle(self):
        phi, theta, psi = 0.1, 0.4, 0.2

        expected_a = np.array([0.1300, 0.0117,0.3949])
        expected_phiPrinciple = 0.4487

        a, phiPrinciple = f.zyx_principle(phi, theta, psi)

        np.testing.assert_array_almost_equal(a, expected_a, decimal=4)
        self.assertAlmostEqual(phiPrinciple, expected_phiPrinciple, places=4)

    def test_C2EulerAngles(self):
        # Define a test rotation matrix C21
        C21 = np.array([[0.9027, 0.1830, 0.3894], 
                        [-0.1596, 0.9829, 0.0920], 
                        [0.3996, -0.0209, 0.9165]])

        expected_phi = 0.1
        expected_theta = -0.4
        expected_psi = 0.2

        # Call the function
        phi, theta, psi = f.C2EulerAngles(C21)
        
        # Assert the results
        self.assertAlmostEqual(phi, expected_phi, places=4)
        self.assertAlmostEqual(theta, expected_theta, places=4)
        self.assertAlmostEqual(psi, expected_psi, places=4)

class TestQuaternions:
    def test_quat(self):
        pass


if __name__ == '__main__':
    unittest.main()       