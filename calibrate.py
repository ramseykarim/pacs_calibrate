#!/usr/bin/env python
"""
Wrapper script for the PACS calibration code in scripts/
More automatic version of example.py; uses command line arguments.

To use this script from any directory, run this shell command from the
directory containing this file:
~$ PATH=$(pwd):$PATH
And then you can leave the directory and run commands like
~$ calibrate.py ./processed

Created: July 9, 2020
"""
__author__ = "Ramsey Karim"

import sys
import os
import argparse

from scripts import calc_offset, modify_fits
# calc_offset, modify_fits = None, None

bands = {70: "PACS70um", 160: "PACS160um", 250: "SPIRE250um", 350: "SPIRE350um", 500: "SPIRE500um"}
uncertainties = {"PACS70um": 6., "PACS160um": 8., "SPIRE250um": 5.5, "SPIRE350um": 5.5, "SPIRE500um": 5.5}


def get_data_path():
    """
    Pull the data directory from the first command line argument
    :returns: absolute path to the PACS data folder for reading and writing,
        and the ArgumentParser object with extra keywords in it
    """
    parser = argparse.ArgumentParser(description="Command line tool to zero-point calibrate the PACS data and add systematic uncertainties the error maps.")
    parser.add_argument('directory', type=str, nargs='?', default='./', help="directory containing the Herschel PACS/SPIRE data (default: <current directory> ).")
    parser.add_argument('--test', action='store_true', help="print out the actions that would be taken, but do not execute any of them. No I/O at all.")
    parser.add_argument('--band', type=int, nargs='*', default=[160], help="select the PACS bands to be calibrated. Use integer wavelengths in microns.")
    parser.add_argument('--calc', action='store_true', help="skip the assignment stage. No assignment dialog will be produced. Figures (if produced) will be saved to the formerly specified directory.")
    parser.add_argument('--assign', action='store_true', help="skip straight to the offset assignment. No calculations or diagnostic figures will be produced.")
    parser.add_argument('--beta', action='store_true', help="don't calibrate, just save the beta image and mask as fits files and quit.")
    args = parser.parse_args()
    data_path = args.directory

    if not os.path.isabs(data_path):
        data_path = os.path.abspath(data_path)
    if not os.path.isdir(data_path):
        raise RuntimeError(f"Invalid directory: {data_path}")

    return data_path, args


def is_int_or_float(x, float_round=None):
    """
    Check if string x can be casted to an int without losing some fractional
    bit, and then check if it can be casted to a float.
    If int or float is ok, return the number, and if neither, return False

    Does not accept exponential notation or fringe floats (NaN, inf, imaginary)

    :param x: string, the thing to check
    :param float_round: int or None, if int, passed to "round" function to
        round a float result. If None, no rounding is performed.
    :returns: int, if possible, then float if possible, then False
    """
    # Helper function to do rounding
    def float_and_round(y):
        # Cast to float and round if necessary
        if float_round is None:
            return float(y)
        else:
            return round(float(y), float_round)

    if not x.replace('.', '', 1).isdigit():
        # Can't be valid int or float
        return False
    int_part = x.split('.')[0]
    if not int_part:
        return float_and_round(x)
    if int(int_part) == float(x):
        return int(int_part)
    else:
        return float_and_round(x)


def safe_cast(cls, x):
    """
    Check if x can be casted to cls, if so, return that casted version.
    :param cls: a class (constructor) to apply to x
    :param x: the thing to check
    :returns: False if cannot cast to cls, or the result of the successful cast
    """
    try:
        return cls(x)
    except:
        return False


def get_assigned_offset(band_stub, derived_offset):
    """
    Manage input loop and return an integer offset from user
    :param band_stub: the string bandpass name to reference in the prompt
    :param derived_offset: the float offset derived by the automatic procedure
    :returns: the int assigned offset. Or, None if the file writing should
        be skipped.
    """
    first_msg = f"Derived {band_stub} offset is {derived_offset:.2f}. Assign: "
    next_msg = lambda s: f"Having trouble interpreting your response ({s}) as an integer. Enter 'q' to quit. Assign: "
    quit_commands = ['exit', '', 'q', 'x', 'quit']
    response = input(first_msg)
    while (response.lower() not in quit_commands) and (safe_cast(float, response) is False):
        # This response is NEITHER a quit command NOR a number. Try again
        response = input(next_msg(response))
    # Passed through the while loop under one of the conditions.
    # Find out which one
    if response.lower() in quit_commands:
        # Return None, indicating that we should not assign anything
        return None
    else:
        # Must be a number
        return is_int_or_float(response, float_round=1)


if __name__ == "__main__":
    STOP = False
    data_path, other_args = get_data_path()
    # Can list 70 and 160 here if you wanted
    band_stubs = [bands[k] for k in other_args.band]
    modified_flux_files = {}
    for band_stub in band_stubs:
        # Line up all the filenames
        # Use the NATIVE RESOLUTION (but regridded) PACS map
        pacs_flux_filename = f"{data_path}/{band_stub}-image-remapped.fits"
        # SPIRE flux map filenames for the masks in calc_offset
        spire_filenames = {
            "spire250_filename": f"{data_path}/SPIRE250um-image-remapped.fits",
            "spire500_filename": f"{data_path}/SPIRE500um-image-remapped.fits",
        }
        print(f"Working on {pacs_flux_filename}")
        # Handle the possibility of debug runs that don't need calculation
        if other_args.test:
            print("--- model calculation ---")
            derived_offset = -99.99
        elif other_args.assign:
            print("Skipping to assignment. No calculations will be made.")
            derived_offset = -99.99
        else:
            model = calc_offset.GNILCModel(pacs_flux_filename, target_bandpass=band_stub, **spire_filenames, save_beta_only=other_args.beta)
            if not other_args.beta:
                derived_offset = model.get_offset(full_diagnostic=True, savedir=(data_path if other_args.calc else None))

        # Handle the possibility that we want to save the images (done) and quit
        if other_args.calc or other_args.beta:
            # Do other bands, if present, but don't ask for offset or
            # write errors
            STOP = True
            continue

        # Use the input-handling function to get the assigned offset
        assigned_offset = get_assigned_offset(band_stub, derived_offset)
        if assigned_offset is None:
            # Do other bands, if present, but do not write errors
            STOP = True
            continue
        # Write out the CONVOLVED flux map plus assigned offset
        # The offset should NOT have a resolution dependence
        pacs_flux_filename = f"{data_path}/{band_stub}-image-remapped-conv.fits"
        calibrated_pacs_flux_filename = modify_fits.add_offset(assigned_offset, pacs_flux_filename, savename=data_path, test=other_args.test)
        modified_flux_files[band_stub] = calibrated_pacs_flux_filename
        print("written")

    if STOP:
        sys.exit()

    for band_stub in uncertainties:
        # Consistency between image-conv and error-conv
        if band_stub in modified_flux_files:
            flux_filename = modified_flux_files[band_stub]
        else:
            flux_filename = f"{data_path}/{band_stub}-image-remapped-conv.fits"
        error_filename = f"{data_path}/{band_stub}-error-remapped-conv.fits"
        if not os.path.isfile(flux_filename):
            print(f"{band_stub} not found, skipping")
            continue
        """
        SPIRE from https://www.cosmos.esa.int/documents/12133/1035800/QUICK-START+GUIDE+TO+HERSCHEL-SPIRE
        PACS from https://www.cosmos.esa.int/documents/12133/996891/PACS+Photometer+Quick+Start+Guide
        """
        if not other_args.test:
            modify_fits.add_systematic_error(uncertainties[band_stub]*0.01, error_filename, flux_filename,
                savename=data_path)
        else:
            print(f"--- writing {uncertainties[band_stub]} % error to {error_filename} based on {flux_filename} ---")
