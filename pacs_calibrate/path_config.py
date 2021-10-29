from scipy.constants import c
import numpy as np
from astropy.io.fits import open as fits_open

"""
Configuration module for file paths and related items, some file I/O

The items in this file should be changed by the user (probably Lee) when appropriate.
The items you will most likely change (at top of this file):

gnilc_directory
planck_directory
herschel_bandpass_directory
"""

"""
***************************************************************************
***************************************************************************
GNILC COMPONENT MAPS

This is the path to the GNILC component maps.
Place them all in the same directory & put the path to the directory here.
***************************************************************************
***************************************************************************
"""
# *************************************************************************
gnilc_directory = "/n/sgraraid/filaments/data/filterInfo_PlanckHerschel/"
# *************************************************************************

"""
***************************************************************************
***************************************************************************
PLANCK HFI RIMO

Name/path to the Planck HFI Reduced Instrument Model (RIMO).
Put the 353, 545, and 857 GHz maps in this directory too.
List the full path to the file here
***************************************************************************
***************************************************************************
"""
# *************************************************************************
planck_directory = "/n/sgraraid/filaments/data/filterInfo_PlanckHerschel/"
# *************************************************************************

"""
***************************************************************************
***************************************************************************
HERSCHEL BANDPASS PROFILES

These are names/paths to the Herschel filter profiles.
At present time, these are the same ones Kevin is using;
    I copied them out of the manticore source code.
Place them all in the same directory & put the path to the directory here.
***************************************************************************
***************************************************************************
"""
# *************************************************************************
herschel_bandpass_directory = "/n/sgraraid/filaments/data/filterInfo_PlanckHerschel/"
# *************************************************************************

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Should not be any need to change anything below this (but who knows)
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# Bandpass stubs for each telescope (Planck HFI, Herschel PACS/SPIRE)
# These are the only filters this code supports
hfi_bandpass_stubs = ("F100", "F143", "F217", "F353", "F545", "F857")
herschel_bandpass_stubs = ("PACS70um", "PACS100um", "PACS160um", "SPIRE250um", "SPIRE350um", "SPIRE500um")
# Herschel_beam_sizes in arcseconds
# NOTE (Dec 5, 2019) these are ever so slightly off from what T. Huard uses in
# ConvolveHerschelImages.pro
# PACS: depends on scan speed, but usually the geometric mean of maj/min FWHMs
# SPIRE: (18.1, 24.9, 36.4)
# NEED TO ADDRESS THESE BEAM SIZE DIFFERENCES; I "guessed" on 70 and 100
# PACS Beam sizes are available in Table 3.1 OF PACS Observer's Manual (Version 2.5.1; 09-July-2013)
# SPIRE Beam sizes are available:
# BETWEEN FIGURE 5.8 AND TABLE 5.3 IN SPIRE HANDBOOK (Version 2.5; March 24, 2014), CONCERNING BEAM SIZES ON MAPS WITH (6,10,14)" PIXELS IN (250,350,500)um SPIRE IMAGES
herschel_beam_sizes = (5.9, 7.0, 11.8, 18.2, 24.9, 36.3)
# HFI beam sizes in arcminutes
hfi_beam_sizes = (9.66, 7.27, 5.01, 4.86, 4.84, 4.63,)
# GNILC effective resolution in arcminutes
gnilc_resolution = 5
# FITS extension for each HFI band in the HFI RIMO
hfi_bandpass_indices = (3, 4, 5, 6, 7, 8)
# HFI observation filenames (valid only for 353-857)
hfi_maps = (
    None, None, None, "HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits",
    "HFI_SkyMap_545-field-Int_2048_R3.00_full.fits",
    "HFI_SkyMap_857-field-Int_2048_R3.00_full.fits",
)
# Unit conversions for HFI bands (some reference CMB)
hfi_unit_conversions = (244.1, 371.74, 483.690, 287.450, 1, 1)
# Bandpass centers for Herschel (wavelength, Angstroms)
herschel_bandpass_centers = (689247.4, 979036.1, 1539451.3, 2471245.1, 3467180.4, 4961067.7)
# GNILC component names
gnilc_components = ('Temperature', 'Spectral-Index', 'Opacity')


def if_contained_in(valid_stub_list):
    """
    This is honestly just me having fun with decorators.
    :param valid_stub_list: List of valid bandpass "stub" names.
        Or any list that the input must be from.
    :return: Decorator that limits its function to arguments within
        this bandpass_stub_list and throws an error if not.
    """

    def if_contained_decorator(func_to_decorate):
        # Nested decorator so the original decorator can take arguments!
        def decorated_function(*args):
            # Some function of a given bandpass, with the stub as the first arg
            arg_stub = args[0]
            if arg_stub in valid_stub_list:
                return func_to_decorate(*args)
            else:
                basic_message = "is not a valid stub."
                extra_info = f"Supported options are: {valid_stub_list}"
                raise RuntimeError(f"{arg_stub} {basic_message} {extra_info}")

        return decorated_function

    return if_contained_decorator


def angstroms_to_hz(x_angstroms):
    """
    This is exactly what it looks like
    :param x_angstroms: wavelength in Angstroms (in vacuum)
    :return: frequency in Hz
    """
    return c / (x_angstroms * 1e-10)


def limit_planck_frequency(f_arr):
    """
    Helper function for bandpass_profile functions
    Returns a mask for trimming a frequency-related array
        based on the frequency array of the same shape
    Limits are generous and therefore hardcoded
    :param f_arr: the frequency array
    :return: the trimmed array
    """
    return (f_arr > 1e10) & (f_arr < 2e12)


class PlanckConfig:
    """
    Class to hold "static variables" describing locations of Planck-related files
    Doesn't need to be instanced; everything should be a class variable
    """

    @staticmethod
    @if_contained_in(gnilc_components)
    def component_filename(component_stub):
        """
        Get one of the Planck GNILC dust model component maps
        :param component_stub:
        :return:
        """
        return "{}COM_CompMap_Dust-GNILC-Model-{}_2048_R2.00.fits".format(
            gnilc_directory, component_stub
        )

    hfi_rimo = planck_directory + "HFI_RIMO_R3.00.fits"

    @staticmethod
    @if_contained_in(hfi_bandpass_stubs)
    def bandpass_profile(bandpass_stub):
        """
        Load bandpass profile from the HFI RIMO
        :param bandpass_stub: HFI band stub
        :return: arrays: frequency (Hz), weight (arbitrary)
        """
        with fits_open(PlanckConfig.hfi_rimo) as hdul:
            # Use extension index i to reference RIMO info
            i = hfi_bandpass_indices[hfi_bandpass_stubs.index(bandpass_stub)]
            # Sanity check!
            try:
                assert bandpass_stub == hdul[i].header['EXTNAME'][-4:]
            except AssertionError:
                msg = "Filter name mismatch. "
                msg += "Failed assertion between: "
                msg += f"{bandpass_stub} // {hdul[i].header['EXTNAME'][-4:]}"
                raise RuntimeError(msg)
            # Wavenumber delivered in cm-1; convert to frequency in Hz
            frequency_hz = hdul[i].data['WAVENUMBER'] * c * 1e2
            weight = hdul[i].data['TRANSMISSION']
        # Trim high and low frequencies
        f_mask = limit_planck_frequency(frequency_hz)
        weight = weight[f_mask]
        frequency_hz = frequency_hz[f_mask]
        return frequency_hz, weight

    @staticmethod
    @if_contained_in(hfi_bandpass_stubs)
    def bandpass_center(bandpass_stub):
        """
        Calculate bandpass center for HFI bands
        :param bandpass_stub: HFI band stub
        :return: band center (Hz)
        """
        return float(bandpass_stub[1:]) * 1e9

    @staticmethod
    @if_contained_in(hfi_bandpass_stubs)
    def beam_size(bandpass_stub):
        """
        Get beam size for HFI band
        :param bandpass_stub: HFI band stub
        :return: beam size in arcminutes
        """
        return hfi_beam_sizes[hfi_bandpass_stubs.index(bandpass_stub)]

    @staticmethod
    @if_contained_in(hfi_bandpass_stubs)
    def unit_conversion(bandpass_stub, array):
        """
        Some HFI bands use CMB reference units.
        This function converts to MJy/sr
        :param bandpass_stub: HFI band stub
        :param array: map to be converted to MJy/sr
        :return: map in MJy/sr
        """
        return array * hfi_unit_conversions[hfi_bandpass_stubs.index(bandpass_stub)]

    @staticmethod
    @if_contained_in(tuple(b for b in hfi_bandpass_stubs if b))
    def light_map_filename(bandpass_stub):
        """
        Filename for Planck HFI observed light maps
        :param bandpass_stub: HFI band stub
        :return: file path to map
        """
        return f"{planck_directory}{hfi_maps[hfi_bandpass_stubs.index(bandpass_stub)]}"


class HerschelConfig:
    """
    Class to hold "static variables" describing locations of Herschel-related files
    Doesn't need to be instanced; everything should be a class variable
    """

    @staticmethod
    @if_contained_in(herschel_bandpass_stubs)
    def bandpass_profile(bandpass_stub):
        """
        Load bandpass profile from saved versions of the Herschel profiles
        :param bandpass_stub: Herschel band stub
        :return: frequency (Hz), weight (arbitrary)
        """
        bandpass_filename = f"{herschel_bandpass_directory}{bandpass_stub}_fromManticore.dat"
        bandpass_data = np.loadtxt(bandpass_filename)
        frequency_hz, weight = bandpass_data[:, 0], bandpass_data[:, 1]
        return frequency_hz, weight

    @staticmethod
    @if_contained_in(herschel_bandpass_stubs)
    def bandpass_center(bandpass_stub):
        """
        Calculate bandpass center for Herschel bands
        :param bandpass_stub: Herschel band stub
        :return: band center (Hz)
        """
        return angstroms_to_hz(herschel_bandpass_centers[herschel_bandpass_stubs.index(bandpass_stub)])

    @staticmethod
    @if_contained_in(herschel_bandpass_stubs)
    def beam_size(bandpass_stub):
        """
        Get beam size for Herschel band
        :param bandpass_stub: Herschel band stub
        :return: beam size in arcminutes
        """
        return herschel_beam_sizes[herschel_bandpass_stubs.index(bandpass_stub)] / 60


"""
Convenience functions exposed to the outside world.
"""


def get_bandpass_data(stub):
    """
    Get photometry weighting function for either Herschel or Planck HFI.
    Returns tuple(frequency array in Hz, weight array).
    Relies on this config module to have accurate filenames
        to all Herschel filter profiles as well as the Planck HFI RIMO.
    :param: stub: short string name indicating the bandpass filter
    :returns: tuple(array, array) of frequencies (Hz) and filter transmission.
        Normalization of the transmission curve is arbitrary
    """
    # Check if Planck ('F') or Herschel
    instrument = PlanckConfig if stub[0] == 'F' else HerschelConfig
    return instrument.bandpass_profile(stub)


def get_bandpass_center(stub):
    """
    Get effective band centers in Hz for either Herschel or Planck HFI
    Returns frequency in Hz
    Relies on this config module to have accurate filenames
        to all Herschel filter profiles as well as the Planck HFI RIMO
    :param stub: short string name indicating the bandpass filter
    :return: float frequency in Hz
    """
    # Check if Planck ('F') or Herschel
    instrument = PlanckConfig if stub[0] == 'F' else HerschelConfig
    return instrument.bandpass_center(stub)
