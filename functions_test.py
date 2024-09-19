import functions as f
import numpy as np
import unittest 

class TestRotations(unittest.TestCase):
    def test_rotCx(self):
        assert np.array_equal(f.rotCx(0),np.identity(3))
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

class TestQuaternions(unittest.TestCase):
    def test_from_rotation_matrix(self):
        C21 = np.array([[0, -1, 0],
                        [1,  0, 0],
                        [0,  0, 1]])
        q = f.Quaternion.from_rotation_matrix(C21)
        expected_quaternion = f.Quaternion(0, 0, -0.7071, 0.7071)  # Example values
        self.assertTrue(np.allclose(q.e, expected_quaternion.e, atol=1e-4))
        self.assertAlmostEqual(q.n, expected_quaternion.n, places=4)

    def test_to_rotation_matrix(self):
        q = f.Quaternion(0, 0, -0.7071, 0.7071)
        C21 = q.to_rotation_matrix()
        expected_C21 = np.array([[0, -1, 0],
                                 [1,  0, 0],
                                 [0,  0, 1]])  # Example values
        self.assertTrue(np.allclose(C21, expected_C21, atol=1e-4))

    def test_rates(self):
        q = f.Quaternion(1, 0, 0, 0)
        omega = np.array([0, 0, 1])
        d_quat = q.rates(omega)
        expected_d_quat = f.Quaternion(0, -0.5, 0, 0)  # Example values
        self.assertTrue(np.allclose(d_quat.e, expected_d_quat.e, atol=1e-4))
        self.assertAlmostEqual(d_quat.n, expected_d_quat.n, places=4)

    def test_multiply(self):
        q1 = f.Quaternion(1, 0, 0, 0)
        q2 = f.Quaternion(0, 1, 0, 0)
        result = q1.multiply(q2)
        expected_result = f.Quaternion(0, 0, 1, 0)  # Example values
        self.assertTrue(np.array_equal(result.e, expected_result.e))
        self.assertEqual(result.n, expected_result.n)

    def test_conjugate(self):
        q = f.Quaternion(1, 2, 3, 4)
        conjugate = q.conjugate()
        expected_conjugate = f.Quaternion(-1, -2, -3, 4)
        self.assertTrue(np.array_equal(conjugate.e, expected_conjugate.e))
        self.assertEqual(conjugate.n, expected_conjugate.n)

    def test_as_array(self):
        q = f.Quaternion(1, 2, 3, 4)
        arr = q.as_array()
        expected_arr = np.array([1, 2, 3, 4])
        self.assertTrue(np.array_equal(arr, expected_arr))

class TestOrbits(unittest.TestCase):
    def test_COE(self):
        r= np.array([9031.5,-5316.9,-1647.2])
        v= np.array([-2.8640, 5.1112, -5.0805])
        mu= 398600

        e_h, e_inc, e_ecc, e_RAAN, e_omega, e_theta= 69086.198957, 63.3998, 0.7411, 145.0002, 270.0005, 279.9993
        h, inc, ecc, raan, omega, theta= f.COE(r, v, mu)

        self.assertAlmostEqual(h, e_h, places=6)
        self.assertAlmostEqual(e_inc, inc, places=4)
        self.assertAlmostEqual(e_ecc, ecc, places=4)
        self.assertAlmostEqual(e_RAAN, raan, places=4) #in the function the node line calc comes out as a constant which is wack af
        self.assertAlmostEqual(e_theta, theta, places=4)
        self.assertAlmostEqual(e_omega, omega, places=4)

    def test_coes2rv(self):
        a, ecc, inc, omega, raan, theta, mu= 7027, 0.0002733, 0.5639, 6.2762, 3.6502, 0.0081, 398600
        e_r= np.array([-6132.5154, -3426.9490, 4.2698])
        e_v= np.array([3.1081, -5.5570, 4.0266])

        r, v= f.coes2rv(a, ecc, inc, omega, raan, theta, mu)
        print(r)
        print(v)

        self.assertTrue(np.allclose(e_r, r, atol=1e-4))
        self.assertAlmostEqual(np.all(e_v, v, places=4))

if __name__ == '__main__':
    unittest.main()       