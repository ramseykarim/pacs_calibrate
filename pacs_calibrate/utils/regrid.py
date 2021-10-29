import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5, Angle
from astropy import units

try:
    import healpy as hp
except ModuleNotFoundError:
    print("HEALPY MODULE NOT FOUND; continuing code anyway")
    hp = None
from scipy.interpolate import interpn, griddata

"""
Functions for the outside world.
"""


def healpix2fits(source_hp, target_fits_grid, target_fits_header,
                 nest=False, pixel_scale_arcsec=75, method='scipy',
                 interp_method='nearest'):
    """
    Wrapper for both _healpy and _scipy methods of projecting HEALPix
    to FITS standard.
    See documentation for following for more details:
        regrid_healpix_to_fits_scipy (method='scipy')
        regrid_healpix_to_fits_healpy (method='healpy')
        scipy method is currently more trustworthy.
    :param source_hp: HEALPix map, already opened in healpy (numpy array)
    :param target_fits_grid: data from target grid.
        Pixels with NaN values in this grid will not be interpolated to.
    :param target_fits_header: header object from target grid. Needs WCS info.
    :param nest: "nest" argument to be passed to healpy.
        See open_healpy documentation for notes
    :param pixel_scale_arcsec: intermediate pixel scale in arcseconds.
        75 arcseconds for Planck HFI
    :param method: HEALPix to FITS interpolation routine; 'scipy' or 'healpy',
        see above.
    :param interp_method: interpolation method to pass to interpn (method='scipy' only)
        Options are: 'nearest', 'linear', and 'splinef2d'. Default is 'nearest'.
        See HEALPix2FITS.intermediate_to_target documentation for some info.
        See scipy.interpolate.interpn for more detail.
    :return: array of target shape with values interpolated from source
    """
    if method == 'scipy':
        return regrid_healpix_to_fits_scipy(source_hp, target_fits_grid, target_fits_header,
                                            nest=nest, pixel_scale_arcsec=pixel_scale_arcsec,
                                            method=interp_method)
    elif method == 'healpy':
        return regrid_healpix_to_fits_healpy(source_hp, target_fits_grid, target_fits_header,
                                             nest=nest)
    else:
        raise TypeError("Invalid method: {}".format(method))


def open_healpix(filename, nest=False):
    """
    Shortcut to opening a HEALPix map, so healpy need not be imported for a single call
    :param filename: string filename (with valid path) to HEALPix FITS file
    :param nest: HEALPix nest parameter. True implies "NESTED" data ordering,
        False implies "RING". Usually RING for the newer Planck Archive HEALPix maps.
        If you open up the map with this function, try
            the_map = this_module.open_healpix(filename, nest=False)
            this_module.hp.mollview(the_map)
            plt.show()  # with matplotlib.pyplot as plt
        it will become very clear whether you have the right ordering.
        The wrong ordering will be composed entirely of streaks/artifacts.
    :return: healpy map object
    """
    return hp.read_map(filename, nest=nest)


"""
Convenience functions for projection methods
"""


def wcs2galactic(wcs_obj, ij_list):
    """
    Calculates galactic coordinates from a list of pixel coordinate pairs
    Pixel coordinates should be (row, col) and zero-indexed.
        This is the default in numpy.
    :param wcs_obj: WCS object, the result of WCS(fits_header)
    :param ij_list: numpy array of shape (n x m, 2) for original image of
        dimension (n, m); list of pixel coord pairs
    :return: numpy array of shape (n x m, 2) containing galactic
        l, b coordinates corresponding to the input ij_list
    """
    # Get SkyCoords from ij array indices
    coord_list = wcs_obj.array_index_to_world(ij_list[:, 0], ij_list[:, 1])
    # Access the galactic l, b components of SkyCoords
    l, b = coord_list.galactic.l.degree, coord_list.galactic.b.degree

    return l, b


def make_ij_list(array_shape, mask=None):
    """
    Given the array shape, create a list of pixel coordinate pairs
    Resulting array is of shape (n x m, 2) for a 2D array of shape (n, m)
    This does NOT reverse coordinate order; it remains (row, col)
        (an earlier version did do the reversal)
    For the record, I have written this as 1 line, but it's way easier to read
        broken up into several lines
    :param array_shape: tuple of int dimension lengths (n, m);
        result of array.shape
    :param mask: boolean array of the same shape;
        True where you want coordinates
    :return: numpy array of shape (n x m, 2) containing all i, j
        pixel coordinate pairs
    """
    if mask is None:
        mask = np.full(array_shape, True)
    # Get 0...n, 0...m arrays into a tuple
    pixel_arrays = tuple(np.arange(x) for x in array_shape)
    # Run meshgrid, so get tuple of i and j 2D grids
    ij_grids = np.meshgrid(*pixel_arrays, indexing='ij')
    # Flatten the grids and the mask
    ijm_arrays = (x.ravel() for x in (*ij_grids, mask))
    # Zip these together to generate a triplet of (i, j, flag)
    # Add the (i, j) pair to the running list if the flag is True
    return np.array(tuple((i, j) for i, j, m in zip(*ijm_arrays) if m))


def assign_to_pixels(interp_value_list, pixel_list, target_shape):
    """
    Generate a final image based on a list of interpolated values,
        a list of pixel coordinates, and the desired shape of the final image.
    Final image has NaNs for pixels not assigned interpolated values.
    :param interp_value_list: sequence of length (n x m), values to assign
    :param pixel_list: array of shape (n x m, 2) with pixel coordinate pairs.
        These pairs should be in the same order as the values to assign at
        these coordinates.
    :param target_shape: tuple(int) shape of final image. Should be (n, m)
    :return: Returns an array of target_shape with values from interp_value_lst
    """
    interpolated_data = np.full(target_shape, np.nan)
    for pair_index in range(pixel_list.shape[0]):
        i, j = pixel_list[pair_index, :]
        interpolated_data[i, j] = interp_value_list[pair_index]
    return interpolated_data


def calc_galactic_limits(target_fits_grid, target_fits_header):
    """
    Calculates the high and low galactic coordinate limits in the image.
    Does not assume that the image represents a true "grid" or is oriented
        with l and b
    Edited Sept 15, 2020 to handle 360-wrap edge case.
        astropy_healpix might be a smoother solution.
    :param target_fits_grid: fits data array
    :param target_fits_header: fits header; needs proper WCS info
    :return: tuple( (min_l, max_l), (min_b, max_b) )
    """
    l, b = wcs2galactic(WCS(target_fits_header),
                        make_ij_list(target_fits_grid.shape))
    b_min, b_max = np.min(b), np.max(b)
    l_min, l_max = np.min(l), np.max(l)
    """
    Handle edge case of wrapping at 360

    The problem:
    If the longitude values wrap around 360, those > 360 are represented as
    positive values < 10 or so (since these regions only extend like 5 deg).
    My min and max are then essentially 0 and 360, the entire l span.

    My solution:
    Calculate min and max l first. Check if min is < 90 and max > 270.
    If this is the case, it's highly likely that we wrapped and split the
    distribution. Our real values probably only extend ~5 degrees, so if we've
    wrapped, it's vastly unlikely that anything will extend nearly 180 degrees.
    Subtract 360 from everything > 180, which should safely gather the ~ 355
    values at ~ -5 and leave the ~ +5 values where they are.
        ** Ideally this works, subtracting 360 from >180. However, it triggers
        a bug somewhere else, so I add 360 to <180 instead, and that's just as
        good for this purpose and avoids the bug elsewhere.

    The caveat:
    If we're at high galactic latitude, this could fail, because we might still
    only span ~5 degrees in true separation, but we might span a significant
    longitude due to the cosine factor.

    The solution to the caveat:
    Therefore, I impose a check on the latitude limits. I require b_min to be
    larger than something near -80 and b_max to be smaller than something near
    +80. This means that the most polar latitude must stay around 10 degrees
    from the pole. If the most polar latitude is closer than that to the pole,
    and the longitude wraps at 360, then we will simply have to work with
    the entire longitude span. At ~ |80| latitude, this should be around 6
    times better (for a thin latitude layer) than the worst case (0 latitude).

    I picked 85 degrees because it ensures that the problem, if it still
    occurs, will still be ~ 5-10 times better than the worst unhandled case.
    It's probably fine to get closer to 90, but I'd rather stay 5 degrees away.
    At any rate, I don't think we have any polar sources, since that's not where
    we should find lots of dust.
    """
    if (l_min < 90) and (l_max > 270):
        printlims = lambda : f"{l_min:.2f}, {l_max:.2f}"
        wrn = f"Gal. lon limits ({printlims()}) span < 90 to > 270."
        if (b_min > -82) and (b_max < 82):
            print(f"FIXED: {wrn}", end=" ")
            # This has probably wrapped around 360
            # Add 360 to everything less than 180 and recalculate
            l[l < 180] += 360
            l_min, l_max = np.min(l), np.max(l)
            print(f"Adjusted limits are {printlims()}")
        else:
            print("+"*45)
            print(f"WARNING: {wrn}")
            print("Source is close to polar, so I am *not* addressing this.")
            print("+"*45)
    return (l_min, l_max), (b_min, b_max)


def calc_pixel_count(coord_limits_galactic, pixel_scale_arcsec):
    """
    Calculate necessary number of pixels of size pixel_scale_arcsec to
        span the given galactic coordinate limits
    :param coord_limits_galactic: (min, max l), (min, max b)
    :param pixel_scale_arcsec: size of each pixel in arcseconds
    :return: (pixels for l direction, pixels for b direction)
    """

    # 3600 arcseconds per degree
    def to_number(lim):
        return int(np.ceil(3600 * (lim[1] - lim[0]) / pixel_scale_arcsec))

    return tuple(to_number(lim) for lim in coord_limits_galactic)


def make_lb_grid_arrays(projection):
    """
    Get the defining galactic l and b arrays (like meshgrid inputs)
        for the given healpy projection.
    Note from my previous code:
        The projection appears to work like meshgrid, so this is fine.
        I compared using meshgrid. This is correct.
    BIG NOTE: I had to hack through a healpy bug.
        This function works around a bug in healpy. healpy does something
        funny with galactic l > 180 degrees. healpy will somehow report
        negative l, which can be seen in CartesianProj.get_extent()
        But if you use CartesianProj.get_center(), that works on
        l=160 degrees, but not on l=270 degrees. There, I got the
        complement of 270, 90. Weird! I think this fools healpy into
        thinking it's generating invalid coordinates when queried with i,j.
        It will mask the coordinates and you'll have an empty array.
        My hacky fix is to just unmask the arrays even though healpy thinks
        they're junk. This may cause problems somewhere else, but it works
        for now.

        UPDATE Septempber 15, 2020: this caused problems elsewhere (I think)
            I fixed a 360-wrapping issue, but it created a bug in the
            interpolation routine. This is the only place I can think that
            might be manifesting

            I think it's something to do with having both positive and negative
            l values. After fixing that first bug, I was using values centered
            around 0. Now I'm using values centered aronud 360, and this works.
    :param projection: healpy CartesianProj object
    :return: meshgrid-input-like arrays of l, b
    """
    # Get number of pixels in the l and b directions
    n_pix_l = projection.arrayinfo['xsize']
    n_pix_b = projection.arrayinfo['ysize']
    # Use healpy ij2xy to get the l, b coordinates given pixel i, j (row, col)
    # These ordinarily return masked arrays, but "data" gets the
    #   underlying data, masked or not. Hacked!
    l_range = projection.ij2xy(i=np.full(n_pix_l, n_pix_b // 2),
                               j=np.arange(n_pix_l))[0].data
    b_range = projection.ij2xy(i=np.arange(n_pix_b),
                               j=np.full(n_pix_b, n_pix_l // 2))[1].data
    # Flip sign on galactic l, because healpy reports negative l
    l_range = -l_range
    return l_range, b_range


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
        :param pixel_scale_arcsec: intermediate pixel scale in arcseconds.
            This will not be the final pixel scale; that is defined by target.
            This should reflect the approximate pixel scale of the HEALPix
            data. For example, Planck HFI maps are saved at a 75 arcsecond
            pixel scale.
        """
        self.target_data = target_data
        self.target_head = target_header
        self.pixel_scale = pixel_scale_arcsec
        # Declare some instance variables to be set in prepare_projection
        self._projection = None
        self._b_src, self._l_src = None, None
        self._bl_target_pairs, self._pixel_list = None, None
        # Sets the above variables; used by the actual projection methods
        self.prepare_projection()
        # Stores the intermediate image, and should be None when not in use
        self._intermediate = None

    def prepare_projection(self):
        """
        Set up a healpy.projector.CartesianProj object with the appropriate
        galactic l, b limits.
        This object doesn't have any data, it's just a glorified (and terrible)
        WCS-like object.
        :return: healpy.projector.CartesianProj object, primed for target grid
        """
        box_lim_l, box_lim_b = calc_galactic_limits(self.target_data,
                                                    self.target_head)
        n_pix_l, n_pix_b = calc_pixel_count((box_lim_l, box_lim_b),
                                            self.pixel_scale)
        self._projection = hp.projector.CartesianProj(xsize=n_pix_l, ysize=n_pix_b,
                                                      lonra=np.array(box_lim_l),
                                                      latra=np.array(box_lim_b))
        self._projection.set_flip('astro')

        # Get the l and b grid arrays from which we will interpolate
        # The intermediate image is a regular grid in l, b
        self._l_src, self._b_src = make_lb_grid_arrays(self._projection)
        # Get the target image pixel coordinates
        self._pixel_list = make_ij_list(self.target_data.shape,
                                        mask=(~np.isnan(self.target_data)))
        # Convert image coordinates (pixels: row, col) to galactic coordinates
        # Do not assume these are a structured grid in l, b
        l_target, b_target = wcs2galactic(WCS(self.target_head), self._pixel_list)
        # Note inversion of l, b: l is the 'x' axis, which takes the 'j' index
        self._bl_target_pairs = np.stack([b_target, l_target], axis=1)
        # These are probably quite large, so delete them
        del l_target, b_target

    def healpix_to_intermediate(self, source_hp, nest=False):
        """
        Project intermediate image at self.pixel_scale from healpix source.
        Since healpy doesn't let you (easily) fully define a grid and then
        interpolate to it, we have to play its little game and step through
        this intermediate map.
        In self.intermediate_to_target, the intermediate map is mapped to
        the target grid.
        This map should fully encompass the target map.
        Sets self.intermediate to a numpy array, the image at self.pixel_scale
        :param source_hp: HEALPix map, already opened in healpy
        :param nest: nest parameter for healpy. See open_healpy for notes
        """
        # Get healpy "n_side" parameter
        n_side = hp.get_nside(source_hp)

        # Get vec2pix function to pass to CartesianProj.projmap
        # Fun fact! The healpy documentation incorrectly defines
        #   the type of vec2pix function needed by projmap! Terrible!
        def vec2pix_func(*xyz):
            return hp.pixelfunc.vec2pix(n_side, *xyz, nest=nest)

        # Now project the HEALPix map using the CartesianProj
        self._intermediate = self._projection.projmap(source_hp, vec2pix_func)

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
        # Interpolate. Invert the longitude order to be strictly increasing.
        interp_values_list = interpn((self._b_src, self._l_src[::-1]),
                                     intermediate[:, ::-1],
                                     self._bl_target_pairs, method=method,
                                     bounds_error=False, fill_value=np.nan)
        interpolated_data = assign_to_pixels(interp_values_list,
                                             self._pixel_list,
                                             self.target_data.shape)
        return interpolated_data

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


def regrid_healpix_to_fits_scipy(source_hp, target_fits_grid, target_fits_header,
                                 nest=False, pixel_scale_arcsec=75, method='nearest'):
    """
    Wrapper for the HEALPix2FITS object and project method.
    Regrid a HEALPix map onto a target grid.
    This is the preferred way to project a HEALPix image onto a predefined
        FITS grid. regrid_healpix_to_fits_healpy is an alternative using pure healpy.
    This method relies on the healpy CartesianProj object to bring HEALPix to
        grid, and the scipy.interpolate.interpn routine to interpolate the resulting
        "intermediate" grid to the final target grid.
        The intermediate grid should be close to the native pixel scale of the data.
        For Planck HFI, this defaults to 75 arcseconds.
    The data in the target grid doesn't matter; this routine will
        just not locations of NaNs and use the grid shape.
    Note that "target" does NOT imply destination. This target array/header
        will NOT be modified or overwritten in any way.
    Designed for Planck Archive HEALPix to Herschel PACS/SPIRE grids.
    :param source_hp: HEALPix map, already opened
    :param target_fits_grid: data from target grid.
        Pixels with NaN values in this grid will not be interpolated to.
    :param target_fits_header: header object from target grid. Needs WCS info.
    :param nest: "nest" argument to be passed to healpy.
        See open_healpy for notes
    :param pixel_scale_arcsec: intermediate pixel scale in arcseconds.
    :param method: interpolation method to pass to interpn
    :return: array of target shape with values interpolated from source
    """
    # See? Now you don't have to write that!
    # Note that the call signature is the same as the alternative function!
    return HEALPix2FITS(target_fits_grid, target_fits_header,
                        pixel_scale_arcsec=pixel_scale_arcsec).project(
        source_hp, nest=nest, method=method
    )


"""
Alternate to HEALPix2FITS.project: regrid_healpix_to_fits_healpy
Uses pure healpy to project.
Given the number of bugs I've found in healpy, I trust the HEALPix2FITS method.
"""


def regrid_healpix_to_fits_healpy(source_hp, target_fits_grid, target_fits_header,
                                  nest=False):
    """
    ALTERNATE TO HEALPix2FITS
    <IMPORTANT NOTE>
    This routine uses the healpy built-in get_interp_val routine. See
        their documentation for more info.
    (Ramsey, July 25, 2019) This function uses what healpy offers to regrid.
        HOWEVER, I think the scipy option using interpn is safer,
        re: treating the wcs projection properly.
        Don't use this one! This is included here for completeness.
        If, someday, you decide that healpy is the way to go, here's your
        routine.
    </IMPORTANT NOTE>
    Regrid a HEALPix map onto a target grid.
    The data in the target grid doesn't matter; this routine will
        just not locations of NaNs and use the grid shape.
    Note that "target" does NOT imply destination. This target array/header
        will NOT be modified or overwritten in any way.
    Designed for Planck Archive HEALPix to Herschel PACS/SPIRE grids.
    :param source_hp: HEALPix map, already opened
    :param target_fits_grid: data from target grid.
        Pixels with NaN values in this grid will not be interpolated to.
    :param target_fits_header: header object from target grid. Needs WCS info.
    :param nest: "nest" argument to be passed to healpy.
        See open_healpy for notes
    :return: array of target shape with values interpolated from source
    """
    # Get pixel coordinate pair list
    pixel_list = make_ij_list(target_fits_grid.shape,
                              mask=(~np.isnan(target_fits_grid)))
    # Convert this pixel coordinate list to galactic l, b coordinates
    l, b = wcs2galactic(WCS(target_fits_header), pixel_list)
    # Interpolate from HEALPix map to these galactic coordinates
    interp_values_list = hp.get_interp_val(source_hp, l, b,
                                           lonlat=True, nest=nest)
    # Generate the final array (same shape as PACS/SPIRE) and populate
    interpolated_data = assign_to_pixels(interp_values_list,
                                         pixel_list,
                                         target_fits_grid.shape)
    return interpolated_data


"""
General-use FITS regrid
"""


def regrid_fits(source_array, source_head, target_array, target_head,
                method='linear'):
    """
    General regrid of FITS data.
    Target is not "destination". Target is not modified/overwritten.
    This routine does not assume that the image grid is a grid in WCS coordinates.
    The routine interpolates using scipy.interpolate.griddata, which assumes
        unstructured grid input
    :param source_array: numpy array, the data from which to interpolate
    :param source_head: FITS header containing relevant WCS info
    :param target_array: numpy array, the grid (data doesn't matter) to interpolate to
    :param target_head: FITS header containing relevant WCS info
    :param method: interpolation method, passed to scipy.griddata
    :return: numpy array result of interpolation. Grid matches target_grid
    """
    # Get pixel lists for both target and source
    t_pixel_list, s_pixel_list = (make_ij_list(a.shape, mask=~np.isnan(a))
                                  for a in (target_array, source_array))
    # Pixel lists are i,j (row, col), so need to reverse into X,Y for pix2world
    t_coord_list, s_coord_list = (WCS(h).array_index_to_world(p[:, 0], p[:, 1])
                                  for h, p in ((target_head, t_pixel_list),
                                               (source_head, s_pixel_list)))
    # The source data values, flattened and in the pixel & coordinate list order
    s_value_list = source_array[s_pixel_list[:, 0], s_pixel_list[:, 1]]
    # Interpolate via scipy griddata and grid the result
    interp_values_list = griddata(s_coord_list, s_value_list, t_coord_list,
                                  method=method)
    interpolated_data = assign_to_pixels(interp_values_list, t_pixel_list,
                                         target_array.shape)
    return interpolated_data


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
