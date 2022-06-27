""" Test scan image operations """
import unittest
import nibabel as nib
import os


class TestImageOperations(unittest.TestCase):
    """ Test scan image operations """

    def test_image_operations(self):
        self.assertTrue(True)

    def test_load_scan(self):
        """ Test loading of a scan """
        scan = nib.load("data/sample_data/sample_scan.nii")
        self.assertTrue(scan.shape == (160, 240, 256, 1))


def main():
    """ Main """
    unittest.main()


if __name__ == "__main__":
    main()
