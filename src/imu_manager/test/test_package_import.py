import unittest


class ImuManagerImportTest(unittest.TestCase):

    def test_filter_module_imports(self):
        from imu_manager import imu_filter

        self.assertTrue(hasattr(imu_filter, "ImuFilter"))


if __name__ == "__main__":
    unittest.main()
