import numpy as np
import matplotlib.pyplot as plt
import os

from astropy.io.fits import getdata as fits_getdata
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

from .utils import regrid as rgu
from . import path_config as cfg
from .utils import planck as plu


class GNILCModel:
    band_dictionary_template = {'F545': None, 'F857': None}

    def __init__(self, *target_args, target_bandpass='PACS160um', extension=0,
        pixel_scale_arcsec=75, mask_constraint=(0.9, 1.1),
        spire250_filename=None, spire500_filename=None,
        save_beta_only=False):
        """
        Object representing the GNILC dust model and packaging its capability
        to predict flux in various instrument bands.
        This object also contains all the statistical tools needed to calculate
        the target's most common offset from the GNILC-predicted flux, under a
        mask where the GNILC model can be trusted. This  mask is calculated here
        as well by predicting Planck HFI fluxes and comparing back to the observed
        HFI flux used to generate the GNILC model.

        EXAMPLE USE (basic):
        >>> model = GNILCModel(pacs_filename)
        >>> model.get_offset()
        this will print the offset and show the histogram diagnostic plot

        :param target_args: either a filename (str) or a tuple(array, header)
            describing the target whose offset is sought.
            If filename, note the FITS extension argument (default 0)
        :param target_bandpass: stub describing target filter profile.
            Must be one of the (Herschel) filters supported by this code
        :param extension: FITS extension of the data; relevant only if
            target_args gives a string file path
        :param pixel_scale_arcsec: pixel scale (arcseconds) at which to
            grid the HEALPix maps for the intermediate step. Defaults to
            75 arcseconds, the Planck HFI pixel scale, and should be used
            at that value. But, it can be increased to something like 300
            for a faster test/debug run.
        :param mask_constraint: the (low, high) limits on the required
            fractional agreement between the predicted and observed
            HFI fluxes. These constraints are used to make masks for the
            offset derivation.
        :param save_beta_only: intercept the whole bandpass calculation thingy
            and just save the mask and the beta map gridded to the target.
        """
        if len(target_args) == 1 and type(target_args[0]) == str:
            # String file path
            target_filename = target_args[0]
            target_data, target_head = fits_getdata(target_filename, ext=extension, header=True)
        elif len(target_args) == 2:
            # Data, header array
            target_data, target_head = target_args
        else:
            msg = f"Unsupported argument: {target_args}."
            msg += " Accepted formats are: str(filename) or tuple(data, header)."
            raise TypeError(msg)
        # """
        # DEBUG
        # """
        # self.target_data = target_data
        # self.mask = self.accumulate_spire_masks(spire250_filename, spire500_filename)
        # self.diagnostic_mask()
        # plt.show()
        # """
        # /DEBUG
        # """
        self.target_data = target_data
        self.target_wcs = WCS(target_head)
        self.target_bandpass_stub = target_bandpass
        self.projector = rgu.HEALPix2FITS(target_data, target_head,
            pixel_scale_arcsec=pixel_scale_arcsec)
        self.T = self.load_component('Temperature')
        self.beta = self.load_component('Spectral-Index')
        self.tau = self.load_component('Opacity')
        # Declare variables to be set within functions
        self.masks = GNILCModel.band_dictionary_template.copy()
        self.predicted_target_flux = None
        self.mask = None
        self.mask_constraint = mask_constraint
        self.difference = None
        self.stats = {
            # If value is string or Exception, indicates/describes error
            "peak_mode": None,  # location of maximum value in data
            "peak_val": None,  # maximum value in data
            "hist_vals_edges": None,  # tuple(histogram values, bin edges)
            "bin_centers": None,  # histogram bin centers calculated from edges
            "hist_xy": None,  # tuple(xarray, yarray) plot ready histogram
            "spline_roots": None,  # tuple(r1, r2) roots from spline fit
            "gauss_fit": None,  # tuple() Gaussian fit parameters
            "mode": None,  # final mode determination, by some method
        }

        # Basic operations with reusable results
        self.accumulate_planck_masks()
        self.accumulate_spire_masks(spire250_filename, spire500_filename)
        if save_beta_only:
            self.intercept_and_save_beta_and_mask()
        else:
            self.difference_to_target()
            self.offset_statistics()

    @plu.shape_of_component
    def load_component(self, stub):
        """
        Load a GNILC model component and project it to intermediate grid.
        Decorator adjusts the shape of the array.
        :param stub: GNILC component stub
        :return: GNILC component map at intermediate pixel scale (pixel_scale_arcsec)
        """
        component_filename = cfg.PlanckConfig.component_filename(stub)
        self.projector.healpix_to_intermediate(rgu.open_healpix(component_filename))
        return self.projector.pop_intermediate()

    @plu.shape_of_frequency
    def load_bandpass(self, stub):
        """
        Load the frequency and transmission arrays for a filter profile.
        Decorator adjusts the shape of the array.
        :param stub: Planck HFI or Herschel band stub
        :return: tuple(frequency (Hz) array, transmission array)
        """
        return cfg.get_bandpass_data(stub)

    def predict_flux_in_band(self, band_stub):
        """
        Predict the flux in the specified band.
        Integrate flux at intermediate pixel scale for efficiency.
        Return map gridded to target resolution.
        :param band_stub: Planck HFI or Herschel band stub
        :return: flux map (2D array) in MJy/sr matching target grid
        """
        frequency, transmission = self.load_bandpass(band_stub)
        band_center = cfg.get_bandpass_center(band_stub)
        image = plu.calculate_gnilc_flux(band_center, frequency, transmission,
                                         self.T, self.tau, self.beta)
        return self.projector.intermediate_to_target(intermediate=image)

    def generate_planck_ratio_mask(self, planck_band_stub):
        """
        For a single Planck HFI band, load and remap the observed flux,
        and use the GNILC model to predict the flux in that band.
        Take the ratio of observed to predicted.
        Generate a mask based on constraining the observed-to-predicted
        ratio to within 10% (0.9 < ratio < 1.1).
        Save this mask.
        The logic here is that wherever the Planck GNILC model agrees to
        within 10% with the measurements used to  make this model, it must
        be an appropriate model for predicting flux. Thus, in these regions,
        it should also predict the target (probably PACS 160 micron) flux
        fairly well.
        :param planck_band_stub: HFI band stub
        :param mask_constraint: (low, high) limits for mask creation
        """
        planck_map_fn = cfg.PlanckConfig.light_map_filename(planck_band_stub)
        observed_planck_flux = self.projector.project(rgu.open_healpix(planck_map_fn))
        observed_planck_flux = cfg.PlanckConfig.unit_conversion(planck_band_stub,
                                                                observed_planck_flux)
        predicted_planck_flux = self.predict_flux_in_band(planck_band_stub)
        ratio = (observed_planck_flux / predicted_planck_flux)
        # Constrain ratio to be within 10% (and not Nan)
        lo, hi = self.mask_constraint
        self.masks[planck_band_stub] = np.isfinite(ratio) & (ratio > lo) & (ratio < hi)

    def accumulate_planck_masks(self):
        """
        Generate the Planck observed-to-predicted ratio masks for all
        bands listed in the band_dictionary_template.
        Take a bitwise AND of the masks to create the final mask.
        This final mask is used with the difference image to generate
        the offset statistics.
        """
        for band_stub in self.masks:
            self.generate_planck_ratio_mask(band_stub)
        planck_mask = np.all(np.array(list(self.masks.values())), axis=0)
        if self.mask is None:
            self.mask = planck_mask
        else:
            self.mask &= planck_mask

    def accumulate_spire_masks(self, spire250_filename, spire500_filename):
        """
        Create additional masks based on the (160 or 250um) and 500um flux.
        The 500um mask (masking out high fluxes) should aid in eliminating
        high column density regions. The 160/250um mask (masking out low flux)
        should aid in eliminating very low emission regions.
        Uses quantiles rather than fixed flux levels to mask.
        This function does not return anything, it just ANDs these masks with
        self.mask.
        :param spire250_filename: complete filename of the SPIRE 250um flux
            FITS file. The grid must be the same as the PACS image being
            corrected. The resolution does not necessarily need to match.
            This function does not check if WCS matches.
        :param spire500_filename: complete filename of the SPIRE 500um flux
            FITS file. Same restrictions as for 250um.

        At present, this uses THIS BAND instead of SPIRE 250um. The reason
        for this is that this mask gives roughly the same information
        but matches this image's footprint better, since the SPIRE images are
        generally spatially offset.

        At present, 5-quantiles are used for both masks.
        """
        # Load the SPIRE 500 data (so we can do the finite mask first)
        spire_data = fits_getdata(spire500_filename)
        # Make the finite mask from both images
        finite_mask = np.isfinite(spire_data) & np.isfinite(self.target_data)
        # Mask out low PACS emission in this band
        pacs_lo, pacs_hi = flquantiles(self.target_data[finite_mask].flatten(), 5)
        pacs_mask = (self.target_data > pacs_lo)
        debug_mask_1 = (self.target_data < pacs_lo)
        """
        # I commented this stuff out on March 9, 2021 because I don't think I'll
        #   use any of it. It's all here for reference though.

        # First, mask out low 250um flux
        spire_data = fits_getdata(spire250_filename)
        finite_mask = np.isfinite(spire_data)
        spire_lo, spire_hi = flquantiles(spire_data[finite_mask].flatten(), 5)
        # Enforce a "SPIRE exists here" implied mask
        # The reason for this is that SPIRE250 > something is catching low
        # emission regions. These are likely to be at the edge of the FOV.
        # We don't want a case where PACS extends further than SPIRE, so a
        # strip *near* the edge is masked out but the *actual* edge is included.
        mask250 = (spire_data > spire_lo)
        """
        # Now mask out high 500um flux
        spire_lo, spire_hi = flquantiles(spire_data[finite_mask].flatten(), 5)
        mask500 = (spire_data < spire_hi)
        debug_mask_2 = (spire_data > spire_hi)

        if self.mask is None:
            self.mask = (pacs_mask & mask500)
        else:
            self.mask &= (pacs_mask & mask500)

        # # DEBUG: just plotting some stuff to learn about the mask
        fig = plt.figure("Flux and NaN Masks", figsize=(18, 9))
        N_finite = np.sum(finite_mask)
        print("Finite: ", N_finite)
        N_this = np.sum(pacs_mask[finite_mask])
        print("Lower than 20%: ", (N_this/N_finite))
        N_500 = np.sum(mask500[finite_mask])
        print("Higher than 20% ", (N_500/N_finite))
        ax1 = plt.subplot(231)
        ax1.imshow(debug_mask_1, origin='lower')
        ax1.set_title(f"Bottom 20% of {self.target_bandpass_stub}")

        ax3 = plt.subplot(232)
        ax3.imshow(debug_mask_1.astype(int) + debug_mask_2.astype(int), origin='lower')
        ax3.set_title(f"Mask1 OR mask2")

        ax2 = plt.subplot(233)
        ax2.imshow(debug_mask_2, origin='lower')
        ax2.set_title("Top 20% of 500um")

        ax = plt.subplot(234)
        # True is +1, False is 0, NaN is -1
        debug_mask_1_int = debug_mask_1.astype(int)
        debug_mask_1_int[~finite_mask] -= 2
        ax.imshow(debug_mask_1_int, origin='lower')
        ax.set_title(f"Same as above with NaNs highlighted")

        ax = plt.subplot(235)
        # Same rules as above
        debug_mask_1_int = debug_mask_1.astype(int)
        debug_mask_2_int = debug_mask_2.astype(int)
        debug_mask_1_int[debug_mask_1_int == 0] = debug_mask_2_int[debug_mask_1_int == 0]
        debug_mask_1_int[~finite_mask] -= 2
        ax.imshow(debug_mask_1_int, origin='lower')
        ax.set_title(f"Both masks, NaNs highlighted")

        ax = plt.subplot(236)
        # Same rules as above
        debug_mask_2_int[~finite_mask] -= 2
        ax.imshow(debug_mask_2_int, origin='lower')
        ax.set_title(f"500um mask, NaNs highlighted")

        fig.savefig(fig.canvas.get_window_title().replace(' ', '_') + "_" + self.target_bandpass_stub + '.png')
        fig.clf()



    def difference_to_target(self):
        """
        Predict the flux in the target band using the GNILC model.
        Save that map. Convolve the observed target data up to the GNILC
        resolution and create & save a difference image.
        """
        # Assume this map to be at GNILC resolution
        self.predicted_target_flux = self.predict_flux_in_band(self.target_bandpass_stub)
        # Convolve Herschel map to Planck GNILC resolution
        h_beam_size = cfg.HerschelConfig.beam_size(self.target_bandpass_stub)
        p_beam_size = cfg.gnilc_resolution
        convolution_beam_size = np.sqrt(p_beam_size ** 2 - h_beam_size ** 2)
        convolution_beam = rgu.prepare_convolution(self.target_wcs,
                                                   convolution_beam_size,
                                                   self.target_data.shape)
        target_data_convolved = rgu.convolve_properly(self.target_data, convolution_beam)
        # [predicted] - [observed] difference map
        self.difference = self.predicted_target_flux - target_data_convolved

    def offset_statistics(self, n_bins=128, x_range=None):
        """
        Make masked difference image histogram and use it to calculate
        a few basic statistics. Save all this to the self.stats dictionary.
        :param n_bins: number of histogram bins. Default=128
        :param x_range: histogram limits. Default is first/last 25-quantile.
            (updated by rkarim Oct 8, 2019)
            The default x_range is calculated for each region.
            You can rerun this function with a handpicked range for fine-tuned
            results.
            HISTORICAL COMMENT:
            This range (25, 80) is acceptable for the Perseus test region,
            but may need to be tweaked for future regions.
        """
        # Get flattened difference array
        diff_array_flat = self.difference[self.mask].ravel()
        # Check if x_range is None, and if so, calculate appropriate range
        if x_range is None:
            # Use first and last something-quantile to define range
            sorted_diff = np.sort(diff_array_flat)
            x_range = flquantiles(sorted_diff, 25, presorted=True)
            self.stats['trimmed_hist_lims'] = x_range
            print("Calculating histogram bins...")
            print("  diffs (min, max): {:.2f}, {:.2f}".format(
                    sorted_diff[0], sorted_diff[-1]))
            print("  histogram lims: {:.2f}, {:.2f}".format(*x_range))
        # Use Freedman-Diaconis rule (sort of) for bin number
        iqr = flquantiles(sorted_diff, 4, presorted=True)
        bin_width = (2*(iqr[1] - iqr[0]) / (len(sorted_diff) ** (1./3)))
        n_bins = 3 * int((x_range[1] - x_range[0]) / bin_width)
        print("  using {:d} histogram bins".format(n_bins))
        # Run numpy histogram function with default bins and limits
        # Rerun this with new bins/limits if these don't work
        d_hist, d_edges = np.histogram(diff_array_flat,
                                       bins=n_bins, range=x_range)
        # Set up plot-friendly arrays
        hist_x, hist_y = map(lambda x: np.array([x[0], x[1]]).T.flatten(),
                             ((d_edges[:-1], d_edges[1:]), (d_hist, d_hist)))
        # Get centers of bins using edges
        bin_centers = (d_edges[:-1] + d_edges[1:]) / 2
        # Find maximum value (number of times mode occurred)
        peak_val = np.max(d_hist)
        # Get crude mode using this peak value
        mode = bin_centers[d_hist == peak_val][0]
        # Save all these findings to a dictionary for later use
        self.stats.update(dict(hist_vals_edges=(d_hist, d_edges),
                               hist_xy=(hist_x, hist_y),
                               bin_centers=bin_centers,
                               peak_val=peak_val, peak_mode=mode))

    def calculate_offset(self):
        """
        Fit a Gaussian to the difference image histogram to refine the
        mode calculation and return that mode.
        Also saves the fitted parameters to the self.stats dictionary.
        :return: the fit Gaussian mean, representing the mode of the histogram
        """
        # Set up the Gaussian fit to the histogram
        peak_location, peak_val = self.stats['peak_mode'], self.stats['peak_val']
        # Default the standard deviation to 10; should be of that order
        p0 = [peak_location, 10, peak_val]
        x_to_fit = self.stats['bin_centers']
        y_to_fit = self.stats['hist_vals_edges'][0]
        # We will try to fit above half-max to avoid bias from a noisy/uneven floor
        # Find full-width half-max locations
        try:
            # Use spline to find roots of function sunk by half-max
            spline = UnivariateSpline(x_to_fit, y_to_fit - peak_val/2, s=0)
            r1, r2 = spline.roots()
            self.stats['spline_roots'] = (r1, r2)
            mask_to_fit = (x_to_fit > r1) & (x_to_fit < r2)
            # If too few data points are included in this root mask,
            #  fall through the "except" statement and use half-max mask
            if np.sum(mask_to_fit.astype(int)) < 5:
                raise ValueError("Spline roots are too close together")
        except ValueError as e:
            # Approximate roots by limiting function to above half max
            # Depending on function shape, this may not be ideal. Assumes Gaussian.
            print("spline fit estimate of histogram FWHM failed; approximating")
            self.stats['spline_roots'] = e
            mask_to_fit = y_to_fit > peak_val/2
        # Try to fit a Gaussian to the values above half-max
        # If this fails, centroid the middle half of the data
        try:
            # Gaussian fit to distribution
            # noinspection SpellCheckingInspection,PyTypeChecker
            popt, pcov = curve_fit(rgu.gaussian, x_to_fit[mask_to_fit],
                                   y_to_fit[mask_to_fit], p0=p0)
            # Save fitted Gaussian parameters to the stats dictionary
            self.stats['gauss_fit'] = popt
            # Return the mean of the Gaussian (more accurate mode of the distribution)
            mode = popt[0]
        except Exception as e:
            print("Gaussian fit esimate of mean failed; approximating")
            self.stats['gauss_fit'] = e
            mask_to_cntrd = slice(len(x_to_fit) // 4, len(x_to_fit)*3 // 4)
            # Take mode to be the centroid between the first & third quartiles
            mode = np.sum(x_to_fit[mask_to_cntrd] * y_to_fit[mask_to_cntrd])
            mode /= np.sum(y_to_fit[mask_to_cntrd])
        self.stats['mode'] = mode
        return mode

    def diagnostic_difference_histogram(self):
        """
        Plot the difference image histogram with overlaid Gaussian fit.
        """
        fig = plt.figure("Difference Histogram", figsize=(9, 5.5))
        # Plot the histogram itself
        plt.plot(*self.stats['hist_xy'], '-', color=(.1, .5, .1),
                 linewidth=3, label="$F_{GNILC} - F_{obs}$")
        curve_x = self.stats['hist_xy'][0]
        # Check if Gaussian was valid
        if not isinstance(self.stats['gauss_fit'], Exception):
            mode, sigma, amplitude = self.stats['gauss_fit']
            # Trim Gaussian fit x extent to 3 sigma to make it look better
            trim = 3 * sigma
            lo_trim, hi_trim = mode - trim, mode + trim
            # Plot the Gaussian fit
            curve_x = curve_x[(curve_x > lo_trim) & (curve_x < hi_trim)]
            curve_y = rgu.gaussian(curve_x, mode, sigma, amplitude)
            eq_string = r"$Ae^{-(x - \mu)^{2}/2\sigma^{2}}$"
            plt.plot(curve_x, curve_y, '-', color='k', alpha=.7,
                     label="{:s}->({:s}: {:.1f}, {:s}: {:.1f}, A: {:.1f})".format(
                         eq_string, r"$\mu$", mode,
                         r"$\sigma$", sigma,
                         amplitude))
        # Plot spline roots if calculated
        if not isinstance(self.stats['spline_roots'], Exception):
            for i, r in enumerate(self.stats['spline_roots']):
                peak_val = self.stats['peak_val']
                plt.plot([r, r], [peak_val*0.25, peak_val*0.75],
                    color='g', linewidth=0.5,
                    label="Half Max" if i else "_nolabel_")
        # Plot mode (however calculated)
        plt.plot([self.stats['mode']], [self.stats['peak_val']], 'x', markersize=12,
                 color='k', alpha=0.7,
                 label="Offset: {:.2f} MJy/sr".format(self.stats['mode']))
        plt.title("Difference histogram: Predicted $-$ Observed")
        plt.xlabel("Pixel difference between predicted/observed (MJy/sr)")
        plt.ylabel("Histogram count")
        plt.legend()
        return fig

    def diagnostic_flux_map(self):
        """
        Plot the predicted and observed flux.
        The observed flux is shown at native, not GNILC, resolution.
        """
        fig = plt.figure("Predicted and Observed Flux", figsize=(16, 6.5))
        # First plot predicted data
        ax = plt.subplot(131)
        # Find appropriate color limits
        fq, tq = visual_min_max(self.predicted_target_flux)
        last_plot = plt.imshow(self.predicted_target_flux, origin='lower',
                               vmin=fq, vmax=tq)
        fig.colorbar(last_plot, ax=ax)
        plt.title("GNILC Model Predicted Flux (GNILC resolution)")

        # Plot the difference image
        ax = plt.subplot(132)
        # The color limits will be taken from the histogram trim
        fq, tq = self.stats['trimmed_hist_lims']
        last_plot = plt.imshow(self.difference, origin='lower', vmin=fq, vmax=tq)
        fig.colorbar(last_plot, ax=ax)
        plt.title("Predicted $-$ Observed Difference Map")

        # Now repeat the first plot for observed data
        ax = plt.subplot(133)
        fq, tq = visual_min_max(self.target_data)
        last_plot = plt.imshow(self.target_data, origin='lower',
                               vmin=fq, vmax=tq)
        fig.colorbar(last_plot, ax=ax)
        plt.title("Observed {:s} flux (native resolution)".format(
            self.target_bandpass_stub))

        plt.subplots_adjust(top=0.88, bottom=0.11, left=0.05, right=0.975, hspace=0.2, wspace=0.115)
        return fig

    def diagnostic_mask(self):
        """
        Plot the mask used to calculate difference statistics
        """
        fig = plt.figure("GNILC Trust Mask")
        plt.imshow(self.mask.astype(int), origin='lower',
                   vmin=0, vmax=1, cmap='Greys_r')
        plt.colorbar()
        plt.title("Mask for Difference Image; Included if True (1)")
        return fig

    def get_offset(self, no_diagnostic=False, full_diagnostic=False, savedir=None):
        """
        Calculate and return the offset.
        Plot the histogram diagnostic unless no_diagnostic is True.
        Plot the flux and mask diagnostics if full_diagnostic is True
            (and no_diagnostic is False)
        Returns the offset after the plots are closed. Raises an error
            if the offset is either NaN or unprecedentedly large.
        :param no_diagnostic: True if you do NOT want ANYTHING to plot
        :param full_diagnostic: True if you want THREE plots
        :return: offset calibration needed by target, in MJy/sr
        """
        offset = self.calculate_offset()
        print("="*25)
        print("="*25)
        print("= OFFSET: {:6.2f} MJy/sr =".format(offset))
        print("="*25)
        print("="*25)
        if not no_diagnostic:
            figs = []
            figs.append(self.diagnostic_difference_histogram())
            if full_diagnostic:
                figs.append(self.diagnostic_flux_map())
                figs.append(self.diagnostic_mask())
            if savedir is not None:
                for fig in figs:
                    savename = os.path.join(savedir, fig.canvas.get_window_title().replace(' ', '_') + "_" + self.target_bandpass_stub + '.png')
                    fig.savefig(savename)
                    fig.clf()
            else:
                plt.show()
        if np.isnan(offset) or np.abs(offset) > 1e6:
            msg = "Apparent issue with the derived offset. "
            msg += "Check diagnostic plots with the full_diagnostic=True argument to this function."
            raise RuntimeError(msg)
        return offset

    def intercept_and_save_beta_and_mask(self):
        """
        May 26, 2021
        Regrid the beta map to the target grid and save that and the (full) mask
        to .fits files
        """
        beta_regrid = self.projector.intermediate_to_target(intermediate=self.beta[0, :, :])
        beta_hdr = self.target_wcs.to_header()
        beta_hdr['COMMENT'] = "Planck GNILC spectral index"
        beta_hdu = fits.PrimaryHDU(data=beta_regrid, header=beta_hdr)
        beta_hdu.writeto("planck-beta.fits")
        mask_hdr = self.target_wcs.to_header()
        mask_hdr['COMMENT'] = "Mask from calc_offset.py"
        mask_hdu = fits.PrimaryHDU(data=self.mask.astype(float), header=mask_hdr)
        mask_hdu.writeto("calib-mask.fits")
        print("Wrote 2 fits files.")

"""
Helper functions
"""

def flquantiles(x, q, presorted=False):
    """
    Get values of first and last q-quantiles of x values.
    If x is multi-D, only works if first axis (0) is sample value axis.
    :param x: sample values
    :param q: number of quantiles. Should be >2
    :param presorted: True if x array is already sorted. Could save
        time if running this function multiple time on large x array.
    :return: tuple(first, last) where elements have dtype of x[i]
    """
    sorted_x = np.sort(x, axis=0) if not presorted else x
    first_quant = sorted_x[len(x) // q]
    last_quant = sorted_x[(q-1)*len(x) // q]
    return (first_quant, last_quant)


def visual_min_max(array):
    """
    Get locations of some visually appropriate min/max for an n-D array.
    Uses locations of 10th and 90th data percentiles
    :param array: n-dimensional array of values
    :return: tuple of the values near (10%, 90%) in sorted order
    """
    # Clean, flatten, and sort array
    array_1d = array[np.isfinite(array)].flatten()
    # Use above function for quartiles. Values won't be exact,
    # but will be close enough if the array is large.
    return flquantiles(array_1d, 10)
