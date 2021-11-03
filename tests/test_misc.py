"""
Tests for utils/misc.py

I will practice with the unittest module here. I don't know that it's strictly
necessary for this project, but I think it's a decent opportunity to try it!

Created: November 2, 2021
"""
__author__ = "Ramsey Karim"


import unittest

from astropy import coordinates

from pacs_calibrate.utils import misc


def setup_wcs_inputs(good=True, pixel_scale=1):
    """
    Test utility function for setting up WCS object / getting coordinates
    """
    if good:
        coord_str = "50:00:00 +10:00:00"
    else:
        coord_str = "0:00:00 +0:00:00"
    coord = coordinates.SkyCoord(coord_str, frame='galactic',
        unit=(misc.u.deg, misc.u.deg))
    shape = (10, 10)
    return coord.fk5, shape, pixel_scale


class TestWCS(unittest.TestCase):

    def test_inputs(self):
        # Test the inputs (briefly)
        with self.assertRaises(RuntimeError):
            misc.make_wcs()


    def test_wcs(self):
        # Make a WCS object and test it
        # Check that make_wcs runs and contains its reference pixel (when default)
        coord, shape, pixel_scale = setup_wcs_inputs()
        wcs = misc.make_wcs(ref_coord=coord, grid_shape=shape,
            pixel_scale=pixel_scale)
        self.assertTrue(wcs.footprint_contains(coord))
