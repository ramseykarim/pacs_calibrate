import numpy as np
from scipy.constants import k, h, c


def sizeof_fmt(num, suffix='B'):
    """
    Convert number of bytes to string expression with proper binary prefix.
    Taken directly from StackOverflow
    :param num: size of some object in number of bytes
    :param suffix: symbol for basic unit. Default='B' for Bytes
    :return: formatted, human-readable string describing the size
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# (For PyCharm IDE)
# noinspection PyPep8Naming
def B(T, nu):
    """
    Planck blackbody function in SI units (frequency).
    Simple enough for numpy broadcasting to work well.
    :param T: temperature in Kelvin
    :param nu: frequency in Hz
    :return: spectral radiance in W sr-1 m-2 Hz-1
    """
    coefficient = 2 * h * (nu ** 3) / (c * c)
    exponent = h * nu / (k * T)
    return coefficient / (np.exp(exponent) - 1)


def shape_of_component(func_to_decorate):
    """
    Decorator for functions returning component arrays.
    Adds an empty dimension in a convenient place.
    :param func_to_decorate: a function that returns a component map
    :return: the component map with an extra dimension
    """

    def decorated_function(*args):
        return func_to_decorate(*args)[np.newaxis, :]

    return decorated_function


def shape_of_frequency(func_to_decorate):
    """
    Decorator for functions returning frequency-related arrays.
    Adds an empty dimension in a convenient place.
    :param func_to_decorate: a function that returns (frequency, weight)
    :return: (frequency, weight) but with extra dimensions
    """

    def decorated_function(*args):
        return tuple(map(lambda array: array[:, np.newaxis, np.newaxis], func_to_decorate(*args)))

    return decorated_function


def flux_helper(n_start, n_end, frequency, weight, normalization, temperature, tau, beta):
    """
    Helper function for calculating flux in blocks
    :param n_start: start index
    :param n_end: end index
    :param frequency: frequency in Hz (shaped correctly)
    :param weight: transmission curve for some filter (shaped correctly)
    :param normalization: normalized weight curve (shaped correctly)
    :param temperature: temperature in Kelvin (shaped correctly)
    :param tau: opacity at 353 GHz (shaped correctly)
    :param beta: spectral index (shaped correctly)
    :return: flux for the given filter profile (weight) between axis=1 n_start:n_end
    """
    result = np.empty((frequency.shape[0], n_end - n_start, temperature.shape[2]))
    result[:] = B(temperature[:, n_start:n_end, :], frequency)
    result *= tau[:, n_start:n_end, :]
    # 353 GHz is the frequency referenced by the opacity map
    result *= (frequency / (353 * 1e9)) ** beta[:, n_start:n_end, :]
    result *= weight
    result = np.sum(result, axis=0) / normalization
    # 10^20 to convert from SI to MJy/sr
    return result * 1e20


def calculate_gnilc_flux(band_center, frequency, weight, temperature, tau, beta):
    """
    Calculate Planck GNILC dust model-predicted flux for a given filter profile.
    Because this can be extremely memory intensive but highly parallel, the
        calculation may be done in 128 MiB blocks. This will happen if the
        frequency x row x col array is larger than 512 MiB.
    :param band_center: float effective central frequency in Hz
    :param frequency: frequency array in Hz (shaped correctly)
    :param weight: transmission curve for some filter (shaped correctly)
    :param temperature: temperature in Kelvin (shaped correctly)
    :param tau: opacity at 353 GHz (shaped correctly)
    :param beta: spectral index (shaped correctly)
    :return: flux for the given filter profile (weight)
    """
    # Calculate filter curve normalization
    normalization = np.sum(weight * band_center / frequency)
    # 64 bit float == 8 byte float
    # noinspection PyPep8Naming
    MiB = 1024 * 1024
    # noinspection PyPep8Naming
    cube_size = frequency.size * temperature.size * 8 / MiB
    is_large = cube_size > 512
    if is_large:
        row_step = 128 * MiB // (frequency.size * temperature.shape[2] * 8)
        row_msg = None
        if row_step < 5:
            # Added this on July 28, 2020 in response to a bug
            # Should cover A) very long waits and B) row_step == 0, which
            # triggers a ZeroDivisionError when we divide a few lines down
            row_msg = f"Row step {row_step} requested, but floored at 5."
            row_step = 5
        n_rows, n_cols = temperature.shape[1], temperature.shape[2]
        n_steps = n_rows // row_step
        n_rows_leftover = n_rows % row_step
        print("CALCULATING FLUX IN BLOCKS")
        if row_msg is not None:
            print(f"Extremely large data. {row_msg} Be warned of long calculation time.")
        print("ROWS: %d. LEFTOVER: %d.\nNeed %d steps." % (n_rows, n_rows_leftover, n_steps))
        print("Shape: Frequency x rows x columns x sizeof(float)")
        print("Step size: %d x [[%d]] x %d x 64 bits = %s" % (frequency.shape[0], row_step, n_cols, sizeof_fmt(frequency.shape[0] * row_step * n_cols * 8)))
        print("Total size: %d x [[%d]] x %d x 64 bits = %s" % (frequency.shape[0], n_rows, n_cols, sizeof_fmt(frequency.shape[0] * n_rows * n_cols * 8)))
        result = np.zeros((n_rows, n_cols))
        for i in range(n_steps):
            n_start, n_end = i * row_step, (i + 1) * row_step
            result[n_start:n_end, :] = flux_helper(n_start, n_end,
                                                   frequency, weight,
                                                   normalization,
                                                   temperature, tau, beta)
            print("-> array[0:%d, [[%d:%d]], 0:%d]\r" % (frequency.shape[0], n_start, n_end, n_cols), end="")
        n_start, n_end = n_steps * row_step, n_steps * row_step + n_rows_leftover
        print("-> array[0:%d, [[%d:%d]], 0:%d]" % (frequency.shape[0], n_start, n_end, n_cols))
        result[n_start:n_end, :] = flux_helper(n_start, n_end,
                                               frequency, weight,
                                               normalization,
                                               temperature, tau, beta)
    else:
        result = flux_helper(0, temperature.shape[1], frequency, weight,
                             normalization, temperature, tau, beta)
    return result
