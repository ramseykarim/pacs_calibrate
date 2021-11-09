"""
Regridding utility functions.

Created: July 26, 2019 (approximately)
"""
__author__ = "Ramsey Karim"


import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5, Angle
from astropy import units as u

# try:
#     import healpy as hp
# except ModuleNotFoundError:
#     print("HEALPY MODULE NOT FOUND; continuing code anyway")
#     hp = None
# from scipy.interpolate import interpn, griddata

import reproject
from reproject import mosaicking


"""
Functions for the outside world.
"""


def open_healpix(filename, **kwargs):
    """
    Shortcut to opening a HEALPix map
    :param filename: string filename (with valid path) to HEALPix FITS file
    :param kwargs: not used; for backwards compatibility
    :return: HDU object
    """
    with fits.open(filename) as hdul:
        hdu = hdul[1] # HEALPix default is 1 (see `hdu_in` parameter description here: https://reproject.readthedocs.io/en/stable/api/reproject.reproject_from_healpix.html#reproject.reproject_from_healpix)
    return hdu


"""
HEALPix2FITS object for best-effort projecting HEALPix to FITS standard.
"""


class HEALPix2FITS:

    def __init__(self, target_data, target_header, pixel_scale_arcsec=75):
        """
        Create a HEALPix2FITS object.
        This object carries functionality to reliably regrid from a HEALPix
        map to a predefined standard target FITS grid with WCS information.
        This object can be reused with different source HEALPix maps.
        :param target_data: data array from target FITS
        :param target_header: header from target FITS
        :param pixel_scale: intermediate pixel scale in arcseconds.
            This will not be the final pixel scale; that is defined by target.
            This should reflect the approximate pixel scale of the HEALPix
            data. For example, Planck HFI maps are saved at a 75 arcsecond
            pixel scale.
            This can be any Quantity equivalent to arcseconds, or an int/float
            representing arcseconds
        """
        self.target_data = target_data
        self.target_head = target_header
        if hasattr(pixel_scale_arcsec, 'unit'):
            self.pixel_scale = pixel_scale_arcsec
        else:
            self.pixel_scale = pixel_scale_arcsec * u.arcsec

        # Intermediate grid descriptors (int. grid is at self.pixel_scale)
        self._int_wcs = None
        self._int_shape = None

        # Sets the above variables; used by the actual projection methods
        # TODO: See if I actually need a setup function?
        self.prepare_projection()

        # Stores the intermediate image, and should be None when not in use
        self._intermediate = None

    def prepare_projection(self):
        """
        SETUP FUNCTION, can rename
        """
        self._int_wcs, self._int_shape = mosaicking.find_optimal_celestial_wcs(
            [(self.target_data, self.target_head),], resolution=self.pixel_scale
        )

    def healpix_to_intermediate(self, source_hp, **kwargs):
        """
        Project intermediate image at self.pixel_scale from healpix source.

        In self.intermediate_to_target, the intermediate map is mapped to
        the target grid.

        This map should fully encompass the target map.
        Sets self.intermediate to a numpy array, the image at self.pixel_scale
        :param source_hp: HDU containing HEALPix data
        :param kwargs: not used; backwards compatibility
        """
        # Get first return value only, second one is footprint
        self._intermediate = reproject.reproject_from_healpix(source_hp,
            self._int_wcs, shape_out=self._int_shape)[0]

    def pop_intermediate(self):
        intermediate = self._intermediate
        self._intermediate = None
        return intermediate

    def intermediate_to_target(self, intermediate=None, method='nearest'):
        """
        Interpolate the self.intermediate image to the target FITS image grid.
        Assumes the intermediate image is on the CartesianProj grid;
            it should have been created by self.healpix_to_intermediate
        This routine interpolates via scipy.interpolate.interpn
            interpn assumes a uniform grid source; the output of healpy's projection
            fulfills this requirement.
            interpn does not need to interpolate to a gridded target; this is good
            for our unstructured target (we don't want to assume RA/DEC are uniform)
            (though there is evidence that RA/DEC are uniform...)
        :param intermediate: overrides "self.intermediate"
        :param method: type of interpolation used by scipy.interpolate.interpn
            Default is 'nearest', since that is the most honest way to project
            poor-resolution/pixel scale data onto a finer grid, in my opinion.
            Options are 'nearest', 'linear', and 'splinef2d'.
        :return: The data from the HEALPix source interpolated
            onto the target FITS grid.
        """
        if intermediate is None:
            intermediate = self._intermediate
        result = reproject.reproject_interp((intermediate, self._int_wcs),
            self.target_head, shape_out=self.target_data.shape, return_footprint=False)
        return result

    def project(self, source_hp, nest=False, method='nearest'):
        """
        Wrapper for the two steps in projecting HEALPix to FITS target.
        :param source_hp: source HEALPix, already opened with healpy
        :param nest: nest parameter for healpy. See open_healpy for notes
        :param method: interpolation method to pass to interpn
        :return: final interpolated image on target FITS grid
        """
        self.healpix_to_intermediate(source_hp, nest=nest)
        result = self.intermediate_to_target(method=method)
        # Clear intermediate image to avoid confusion
        self._intermediate = None
        return result


# noinspection SpellCheckingInspection
def prepare_convolution(w, beam, data_shape):
    """
    Given a WCS object and beam FWHMs in arcminutes,
        returns the Gaussian needed to convolve image by this kernel
    Gaussian is returned in smaller array that includes contributions out to 5sigma
    :param w: WCS object for image
    :param beam: float beam FWHM in arcuminutes
    :param data_shape: 2-element tuple describing (row, col) shape of image array
    :return: Gaussian convolution kernel with FWHM of beam. 2d array of data_shape
    """
    # Find pixel scale, in arcminutes
    dtheta_dpix_i = w.array_index_to_world(0, 0).separation(w.array_index_to_world(1, 0)).to('arcmin').to_value()
    dtheta_dpix_j = w.array_index_to_world(0, 0).separation(w.array_index_to_world(0, 1)).to('arcmin').to_value()
    dthetas = [dtheta_dpix_i, dtheta_dpix_j]
    # FWHM to standard deviation
    sigma_arcmin = beam / 2.35
    ij_arrays = [None, None]
    for x in range(2):
        x_array = np.arange(data_shape[x], dtype=float) - data_shape[x]//2
        x_array *= dthetas[x]
        y_array = np.exp(-x_array * x_array / (2 * sigma_arcmin * sigma_arcmin))
        y_array = y_array / np.trapz(y_array)
        ij_arrays[x] = y_array
    i, j = ij_arrays
    convolution_beam = i[:, np.newaxis] * j[np.newaxis, :]
    return convolution_beam


def convolve_helper(image, kernel):
    ft = np.fft.fft2(image) * np.fft.fft2(kernel)
    result = np.fft.ifft2(ft)
    return np.real(np.fft.fftshift(result))


def convolve_properly(image, kernel):
    """
    Convolve image with kernel
    Preserve NaNs
    Also mitigate edge effects / normalization from NaN correction
    Convolves using convolve helper (check that implementation for details)
    :param image: 2d array image
    :param kernel: 2d array kernel, must be same shape as image
    :return: 2d array convolved result matching shape of image
    """
    image = image.copy()
    nan_mask = np.isnan(image)
    image[nan_mask] = 0.
    result = convolve_helper(image, kernel)
    # now account for edge effects / normalization
    image[~nan_mask] = 1.
    norm = convolve_helper(image, kernel)
    image[:] = 1.
    norm /= convolve_helper(image, kernel)
    result /= norm
    result[nan_mask] = np.nan
    return result

def gaussian(x, mu, sigma, amplitude):
    """
    Exactly what it looks like. Good for curve fitting or convolution.
    :param x: independent variable x array
    :param mu: mean of gaussian
    :param sigma: standard deviation of gaussian
    :param amplitude: amplitude coefficient of gaussian
    :return: gaussian curve with array shape of x argument
    """
    coefficient = amplitude / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mu) ** 2 / (2 * sigma * sigma))
    return coefficient * np.exp(exponent)
