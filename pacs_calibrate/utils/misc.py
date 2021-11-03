"""
Utility functions that don't fit into another category.

Created: November 2, 2021
"""
__author__ = "Ramsey Karim"

from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits


def make_wcs(ref_coord=None, ref_pixel=None, grid_shape=None, pixel_scale=None,
    return_header=False, **extra_header_kws):
    """
    Make a fresh, simple WCS object based on a few parameters. Does not need
        any existing data or WCS info.
    This will only work for 2-dimensional image data. No cubes or spectral axes.
    <HISTORY>
        This code is from g0_stars.py. I will probably need to edit this.
        Update: I edited it to accept a pixel scale matrix, but I probably won't
            even use that functionality. I am not sure why VLT and HST use matrices
            with off-diagonal elements, but IRAC uses diagonals only and I trust
            IR astronomers more than optical astronomers.
        This code was copied on Nov 2 2021 from mosaic_vlt.py in the feedback
            repository. I should eventually have just one version of this,
            since it's clear I've used it multiple times in different projects.
            But for now, it's here.
    </HISTORY>
    :param ref_coord: SkyCoord object that will match ref_pixel. This is a
        required argument.
    :param ref_pixel: 2-element integer sequence indicating reference pixel,
        which will match up with the ref_coord.
        The values should be Numpy array indices (0-indexed, ij)
        If not specified, will default to the approximate center of the grid.
    :param grid_shape: 2-element integer sequence indicating the shape of
        the data array.
        If grid shape is (10, 10) i.e. (0..9, 0..9) and you want pixel (4, 4)
        i.e. the fifth i,j pixels to be the center, specify (4, 4).
        This function will pass (4+1, 4+1) to WCS to ensure that the fifth
        pixels are chosen in this case.
        This is a required argument.
    :param pixel_scale: some indication of the pixel scale of the WCS object.
        This can be a 2x2 matrix or a scalar, in which case it can be an
        astropy Quantity. If it's a scalar but not a Quantity, it's assumed
        to be in units of arcminutes.
        This is a required argument.
        The exact keywords added to the header will depend on the form of this
        argument. (Aug 11, 2020)
    :param return_header: return the Header instead of the WCS object made
        from the Header
    :param extra_header_kws: any additional FITS Header keywords and their
        values that you would like to add. If they're not used by WCS, they will
        be lost.
    :returns: simple astropy WCS object described by the arguments.
    """
    # Check arguments
    if any(x is None for x in (ref_coord, grid_shape, pixel_scale)):
        raise RuntimeError("You are missing required arguments.")
    if ref_pixel is None:
        ref_pixel = tuple(int(x/2) for x in grid_shape)
    # Figure out pixel scale and ultimately get a matrix
    if hasattr(pixel_scale, 'shape') and pixel_scale.shape == (2, 2):
        pixel_scale_kwargs = {
            'CD1_1': (pixel_scale[0, 0], "Transformation matrix"),
            'CD1_2': (pixel_scale[0, 1], ""),
            'CD2_1': (pixel_scale[1, 0], ""),
            'CD2_2': (pixel_scale[1, 1], ""),
        }
    else:
        if not isinstance(pixel_scale, u.quantity.Quantity):
            pixel_scale *= u.arcmin
        pixel_scale_kwargs = {
            'CDELT1': -1 * pixel_scale.to(u.deg).to_value(),
            'CDELT2': pixel_scale.to(u.deg).to_value(),  # RA increasing to the left side
        }
    # Lay out the keywords in a dictionary
    kws = {
        'NAXIS': (2, "Number of axes"),
        'NAXIS1': (grid_shape[1], "X/j axis length"),
        'NAXIS2': (grid_shape[0], "Y/i axis length"),
        'RADESYS': (ref_coord.frame.name.upper(), ""),
        'CRVAL1': (ref_coord.ra.deg, "[deg] RA of reference point"),
        'CRVAL2': (ref_coord.dec.deg, "[deg] DEC of reference point"),
        'CRPIX1': (ref_pixel[1] + 1, "[pix] Image reference point"),
        'CRPIX2': (ref_pixel[0] + 1, "[pix] Image reference point"),
        'CTYPE1': ('RA---TAN', "RA projection type"),
        'CTYPE2': ('DEC--TAN', "DEC projection type"),
        'PA': (0., "[deg] Position angle of axis 2 (E of N)"),
        'EQUINOX': (2000., "[yr] Equatorial coordinates definition"),
    }
    kws.update(pixel_scale_kwargs)
    kws.update(extra_header_kws)
    header = fits.Header()
    # Two lines to avoid some weird bug about reading dictionaries in the constructor
    header.update(kws)
    if return_header:
        # Return the Header object
        return header
    else:
        # Return the WCS object
        return WCS(header)
