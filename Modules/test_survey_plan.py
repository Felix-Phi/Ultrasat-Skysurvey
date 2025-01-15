import unittest
import numpy as np
from scipy.interpolate import interp1d
from survey_plan import maglim_func
import os

class TestMaglimFunc(unittest.TestCase):

    def setUp(self):
        # Create mock data for Rdeg.dat and LimMag.dat
        self.rdeg_data = np.linspace(0, 10, 25)
        self.lim_mag_data = np.linspace(20, 25, 25)
        
        with open('Data/Rdeg.dat', 'w') as f:
            for value in self.rdeg_data:
                f.write(f"{value}\n")
        
        with open('Data/LimMag.dat', 'w') as f:
            for _ in range(40):  # Write 40 lines of dummy data
                f.write(",".join(map(str, np.random.rand(25))) + "\n")
            f.write(",".join(map(str, self.lim_mag_data)) + "\n")  # 41st line with actual data

    def tearDown(self):
        # Clean up the mock data files
        os.remove('Data/Rdeg.dat')
        os.remove('Data/LimMag.dat')

    def test_maglim_func(self):
        x_values = np.array([0, 2.5, 5, 7.5, 10])
        expected_results = interp1d(self.rdeg_data, self.lim_mag_data, kind="linear", bounds_error=False, fill_value="extrapolate")(x_values)
        
        results = maglim_func(x_values, source_number=41)
        
        np.testing.assert_array_almost_equal(results, expected_results, decimal=5)

if __name__ == '__main__':
    unittest.main()