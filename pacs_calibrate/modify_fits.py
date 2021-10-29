import os
from astropy.io import fits


def add_offset(offset, filename, extension=0, savename=None, test=False):
    """
    Add a fixed offset to a FITS image. Will only save a single extension.
    Does not overwrite the original (unless savename==filename)
    Will save to the same directory as filename.
    :param offset: int or float additive offset
    :param filename: file path to FITS to modify
    :param extension: FITS extension of data to read; default is 0
    :param savename: file path to which to write the modified data
        Default is to add "-plus######" right before ".fits"
        Will overwrite anything already saved as savename
        If savename ends in '/', it is assumed to specify a directory,
        not a filename.
    :param test: do NOT edit a file. Immediately return the proposed savename
    :return: full name and path of newly saved file
    """
    # Make tidy offset string based on float vs int
    offset_str = "{:06d}".format(offset) if type(offset) == int else "{:06.1f}".format(offset)
    # Add "-plus######.fits" for a default
    savename = process_savename(savename, filename, offset_str)
    if test:
        print(f"--- would have written {savename}, but this is a TEST ---")
        return savename
    with fits.open(filename) as hdul:
        data = hdul[extension].data
        head = hdul[extension].header
    data += offset
    head['COMMENT'] = "Added {:s} MJy/sr offset".format(offset_str)
    # Overwrite if previous version exists, but inform the user
    try:
        fits.writeto(savename, data, header=head)
        print(f"Wrote to {savename} with {offset_str} offset")
    except OSError:
        fits.writeto(savename, data, header=head, overwrite=True)
        print(f"Wrote to {savename} with {offset_str} offset (overwriting existing)")
    return savename


def multiply_flux(flux_coefficient, flux_filename, flux_extension=0,
    source_folder=None, destination_folder=None):
    """
    Opens the flux map referenced by flux_filename in the source_folder,
    multiplies it by the flux_coefficient, and saves the new image to
    the destination_folder with the same flux_filename.
    Example:
    >>> multiply_flux(1.2, "PACS160um-image.fits", source_folder="./",
            destination_folder="../MOD1.2/")
    The above call will save the original image multiplied by 1.2 to
    the full path ../MOD1.2/PACS160um-image.fits
    :param flux_coefficient: float multiplier for new image.
        New image = old image * flux_coefficient
    :param flux_filename: the string filename (not path) of the file to
        read. Will ALSO be the write name.
    :param flux_extension: FITS extension at which flux is found.
    :param source_folder: string path to folder where a flux map already
        exists under the name flux_filename.
    :param destination_folder: string path to folder in which to save new
        map. If this matches source folder, this program will raise an error.
    """
    problem_exists = False
    if source_folder is None:
        problem_exists = True
        print("Please set source_folder keyword argument.")
    if destination_folder is None:
        problem_exists = True
        print("Please set destination_folder keyword argument.")
    if problem_exists:
        raise RuntimeError("Fix the above issue(s)")
    if source_folder == destination_folder:
        raise RuntimeError("Source and destination folders should not be the same.")
    if source_folder[-1] != '/':
        source_folder = source_folder + '/'
    if destination_folder[-1] != '/':
        destination_folder = destination_folder + '/'
    data, header = fits.getdata(source_folder+flux_filename, flux_extension, header=True)
    fits.writeto(destination_folder+flux_filename, data*flux_coefficient, header=header)
    print("Saved {:s}".format(destination_folder+flux_filename))



def add_systematic_error(flux_fraction, error_filename, flux_filename,
                         error_extension=0, flux_extension=0, savename=None):
    """
    Adds the specified fraction of the flux map to the error map as
        uncorrelated systematic uncertainty.
    Specify the flux fraction as a decimal fraction (NOT percent)
    Does not overwrite the existing error map unless you specify
        the original error map's name as the savename here.
    Will save to the same directory as filename.
    :param flux_fraction: decimal fraction of flux to be added to error
        NOT PERCENTAGE. 1.5% should be input here as 0.015
    :param error_filename: file path to error map to modify.
        Does not overwrite this error map (unless savename==error_filename)
    :param flux_filename: file path to flux map referenced by flux_fraction
        Flux and error need to be on grids of the same shape
        They should also be aligned in WCS, but this function will not check
    :param error_extension: FITS extension where error map is found. default=0
    :param flux_extension: FITS extension where flux map is found. default=0
    :param savename: file path to which to write the modified error map
        Default is to add "-plus#.#pct" right before ".fits"
        Will overwrite anything already saved as savename
        If savename ends in '/', it is assumed to specify a directory,
        not a filename.
    :return: full name and path of newly saved file
    """
    # Make flux percentage string
    pct_string = "{:03.1f}pct".format(flux_fraction*100)
    # Add "-plus###pct.fits" for a default
    savename = process_savename(savename, error_filename, pct_string)
    with fits.open(error_filename) as hdul:
        error = hdul[error_extension].data
        head = hdul[error_extension].header
    with fits.open(flux_filename) as hdul:
        flux = hdul[flux_extension].data
    error += flux * flux_fraction
    head['COMMENT'] = "Added {:s} % flux as uncorrelated systematic uncertainty".format(pct_string)
    try:
        fits.writeto(savename, error, header=head)
        print(f"Wrote to {savename} with {pct_string} offset")
    except OSError:
        fits.writeto(savename, error, header=head, overwrite=True)
        print(f"Wrote to {savename} with {pct_string} offset (overwriting existing)")
    return savename


def process_savename(savename, original_filename, append_string):
    """
    Helper function that interprets the savename keyword argument for
        add_systematic_error and add_offset
    :param savename: string (or None) argument set to the "savename" keyword
        in add_systematic_error or add_offset.
        If savename ends in '/', it is assumed to specify a directory,
        not a filename.
    :param original_filename: original filename; only used if savename
        is a directory or None
    :param append_string: string to append to the filename,
        right before ".fits"
    :return: string, full save name + path
    """
    if savename is None:
        # Default filename, saved to same directory as original file
        # Get filename before ".fits" and insert append_string there
        filename_first, fits_stub = original_filename[:-5], original_filename[-5:]
        assert fits_stub == ".fits"
        savename = f"{filename_first}-plus{append_string}{fits_stub}"
    elif os.path.isdir(savename):
        # Default filename, but saved to savename directory
        # Isolate original filename without path
        filename_without_path = original_filename.split('/')[-1]
        # Get name before ".fits" and insert append_string there
        filename_first, fits_stub = filename_without_path[:-5], filename_without_path[-5:]
        assert fits_stub == ".fits"
        # Put a slash in the savename directory if it's not there
        if not savename[-1] == '/':
            savename = savename + '/'
        # Rebuild full filename with savename as path
        savename = f"{savename}{filename_first}-plus{append_string}{fits_stub}"
    # In any other case, savename will be assumed to be the full desired name+path
    return savename
