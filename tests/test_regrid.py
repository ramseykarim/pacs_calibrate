"""
Tests for regrid.py

The classes/functions that are used in calc_offset.py need to be tested to make sure
their behavior doesn't change under a new regrid implementation.
Those classes/functions are:
    HEALPix2FITS
        healpix_to_intermediate
            takes open_healpix() as argument
        intermediate_to_target
        pop_intermediate
        project
            takes open_healpix() as argument
            is just healpix_to_intermediate + intermediate_to_target +
            pop_intermediate wrapper
    open_healpix
        return val is argument to healpix_to_intermediate and project
        this will be hard to test without just opening a file, but
        it shouldn't be a complicated function
    prepare_convolution
    convolve_properly
        (the two convolution functions should eventually be moved to
        utils/misc.py)
    gaussian
        (gaussian should be replaced with astropy.modeling.models.Gaussian1D)

Created: November 2, 2021
"""
__author__ = "Ramsey Karim"


import unittest

import numpy as np

from astropy.io import fits

from pacs_calibrate.utils import misc
from pacs_calibrate.utils import regrid

from .test_misc import setup_wcs_inputs

def setup_target_inputs(**kwargs):
    """
    Test utility for setting up fake "PACS" data
    """
    coord, shape, pixel_scale = setup_wcs_inputs(**kwargs)
    wcs = misc.make_wcs(ref_coord=coord, grid_shape=shape,
        pixel_scale=pixel_scale)
    data = np.ones(shape)
    return data, wcs.to_header()


def setup_fake_healpix(nside):
    """
    Create HEALPix map object, whatever that may be
    For healpy, this is a 1D array. If I switch regrid implementation,
    that could change.

    This should return the same type of object as open_healpix()
    """
    return np.arange(nside)


def print_array_like_image(arr):
    """
    Print an array out as if it were origin='lower'
    """
    print(arr[::-1, :])


class TestBasicProject(unittest.TestCase):

    def test_HEALPix2FITS_init(self):
        target_data, target_header = setup_target_inputs()
        self.assertIsNotNone(regrid.HEALPix2FITS(target_data, target_header))


    def test_healpix_to_intermediate_basic(self):
        projector = regrid.HEALPix2FITS(*setup_target_inputs())
        projector.healpix_to_intermediate(setup_fake_healpix(12))
        result = projector.pop_intermediate()
        # print(result)
        self.assertIsNotNone(result)


    def test_healpix_to_intermediate_intermediate(self):
        # the different npix should cause different values in the result map
        projector = regrid.HEALPix2FITS(*setup_target_inputs())
        values = []
        for i in [1, 2, 4]:
            npix = 12 * i**2
            projector.healpix_to_intermediate(setup_fake_healpix(npix))
            result = projector.pop_intermediate()
            # print(result)
            values.append(result[0, 0])
            self.assertIsNotNone(result, msg=f"failed on npix = {npix}")
        self.assertNotEqual(values[0], values[1])
        self.assertNotEqual(values[0], values[2])
        self.assertNotEqual(values[1], values[2])


    def test_healpix_to_intermediate_badnpix(self):
        projector = regrid.HEALPix2FITS(*setup_target_inputs())
        with self.assertRaises(Exception):
            # invalid npix
            npix = 12345
            projector.healpix_to_intermediate(np.arange(npix))


    def test_healpix_to_intermediate_advanced(self):
        # Check that a larger target field demands a larger intermediate map
        array_sizes = []
        # pixel scales of 1 arcminute and 10 arcminutes
        for ps in [1, 10]:
            projector = regrid.HEALPix2FITS(*setup_target_inputs(pixel_scale=ps))
            projector.healpix_to_intermediate(setup_fake_healpix(12))
            # print(projector._intermediate.shape)
            array_sizes.append(projector.pop_intermediate().size)
        self.assertLess(array_sizes[0], array_sizes[1], msg=f"{array_sizes[0]} !< {array_sizes[1]}")


    def test_project(self):
        # Check that the final projection comes back in the same shape
        data, hdr = setup_target_inputs()
        projector = regrid.HEALPix2FITS(data, hdr)
        result = projector.project(setup_fake_healpix(12))
        self.assertEqual(result.shape, data.shape)


class TestGalacticLongitude(unittest.TestCase):

    def test_project_across_0_lat_lon(self):
        # use l,b = 0,0 and check for NaNs
        data, hdr = setup_target_inputs(good=False)
        projector = regrid.HEALPix2FITS(data, hdr)
        result = projector.project(setup_fake_healpix(12 * 4**2))
        print_array_like_image(result)
        self.assertTrue(not np.any(np.isnan(result)))
