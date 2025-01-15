import numpy as np
import sncosmo

def register_ultrasat_bands(transmission_file, wavelength_file, rdeg_file):
    """
    Register the ULTRASAT bands in SNCosmo using provided data files.

    Args:
        transmission_file (str): Path to the ULTRASAT transmission matrix file.
        wavelength_file (str): Path to the wavelength file.
        rdeg_file (str): Path to the file containing radial position (Rdeg) values.
    """
    # Load data files
    transmission_matrix = np.loadtxt(transmission_file, delimiter=",")
    wavelengths = np.loadtxt(wavelength_file)
    rdeg_values = np.loadtxt(rdeg_file)

    # Trim the data to relevant ranges
    transmission_matrix = transmission_matrix[230:1000, :]
    wavelengths = wavelengths[230:1000]

    # Validate the shapes of the input data
    assert transmission_matrix.shape == (770, 25), "Unexpected shape for the transmission matrix."
    assert len(wavelengths) == 770, "Unexpected number of wavelengths."
    assert len(rdeg_values) == 25, "Unexpected number of radial positions (Rdeg)."

    # Register each band in sncosmo
    for i, rdeg in enumerate(rdeg_values):
        band_name = f"ultrasat_band_{rdeg:.2f}"  # Unique name for each radial position
        transmission = transmission_matrix[:, i]

        # Create and register the band
        band = sncosmo.Bandpass(wavelengths, transmission, name=band_name)
        sncosmo.registry.register(band, force=True)

    print(f"All Ultrasat bands successfully registered.")