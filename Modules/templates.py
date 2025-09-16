import os
import sncosmo

def create_salt3_template(model_dir):
    """
    Creates and registers a custom SALT3 model in SNCosmo.

    Args:
        model_dir (str): Path to the directory containing SALT3 files.

    Returns:
        sncosmo.Model: The registered SNCosmo model based on SALT3.
    """
    # Paths to SALT3 files
    m0file = os.path.join(model_dir, "salt3_template_0.dat")
    m1file = os.path.join(model_dir, "salt3_template_1.dat")
    clfile = os.path.join(model_dir, "salt3_color_correction.dat")
    cdfile = os.path.join(model_dir, "salt3_color_dispersion.dat")
    lcrv00file = os.path.join(model_dir, "salt3_lc_variance_0.dat")
    lcrv11file = os.path.join(model_dir, "salt3_lc_variance_1.dat")
    lcrv01file = os.path.join(model_dir, "salt3_lc_covariance_01.dat")

    # Check if all required files exist
    required_files = [m0file, m1file, clfile, cdfile, lcrv00file, lcrv11file, lcrv01file]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' not found.")

    # Create SALT3 model source
    salt3_source = sncosmo.SALT3Source(
        m0file=m0file,
        m1file=m1file,
        clfile=clfile,
        cdfile=cdfile,
        lcrv00file=lcrv00file,
        lcrv11file=lcrv11file,
        lcrv01file=lcrv01file,
    )

    #Add the dust effect as parameters
    sncosmo.Model(source=salt3_source,effects=[sncosmo.CCM89Dust()],effect_names=["mw"],effect_frames=["rest"])


    # Register the source in SNCosmo (force=True in case it is already registered)
    sncosmo.registry.register(salt3_source, name="QinanSalt3", force=True)

    # Return the SNCosmo model using the custom source
    return sncosmo.Model(source="QinanSalt3")

def create_salt3_template_dust(model_dir):
    """
    Creates and registers a custom SALT3 model in SNCosmo and add the dust parameters RV and EBV.

    Args:
        model_dir (str): Path to the directory containing SALT3 files.

    Returns:
        sncosmo.Model: The registered SNCosmo model based on SALT3.
    """
    # Paths to SALT3 files
    m0file = os.path.join(model_dir, "salt3_template_0.dat")
    m1file = os.path.join(model_dir, "salt3_template_1.dat")
    clfile = os.path.join(model_dir, "salt3_color_correction.dat")
    cdfile = os.path.join(model_dir, "salt3_color_dispersion.dat")
    lcrv00file = os.path.join(model_dir, "salt3_lc_variance_0.dat")
    lcrv11file = os.path.join(model_dir, "salt3_lc_variance_1.dat")
    lcrv01file = os.path.join(model_dir, "salt3_lc_covariance_01.dat")

    # Check if all required files exist
    required_files = [m0file, m1file, clfile, cdfile, lcrv00file, lcrv11file, lcrv01file]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' not found.")

    # Create SALT3 model source
    salt3_source = sncosmo.SALT3Source(
        m0file=m0file,
        m1file=m1file,
        clfile=clfile,
        cdfile=cdfile,
        lcrv00file=lcrv00file,
        lcrv11file=lcrv11file,
        lcrv01file=lcrv01file,
    )

    #Add the dust effect as parameters
    sncosmo.Model(source=salt3_source,effects=[sncosmo.CCM89Dust()],effect_names=["host"],effect_frames=["rest"])


    # Register the source in SNCosmo (force=True in case it is already registered)
    sncosmo.registry.register(salt3_source, name="QinanSalt3", force=True)

    # Return the SNCosmo model using the custom source
    return sncosmo.Model(source=salt3_source,effects=[sncosmo.CCM89Dust()],effect_names=["host"],effect_frames=["rest"])


def create_salt3_template_dust_noerrors(model_dir):
    """
    Creates and registers a custom SALT3 model in SNCosmo and add the dust parameters RV and EBV.

    Args:
        model_dir (str): Path to the directory containing SALT3 files.

    Returns:
        sncosmo.Model: The registered SNCosmo model based on SALT3.
    """
    # Paths to SALT3 files
    m0file = os.path.join(model_dir, "salt3_template_0.dat")
    m1file = os.path.join(model_dir, "salt3_template_1.dat")
    clfile = os.path.join(model_dir, "salt3_color_correction.dat")
    cdfile = os.path.join(model_dir, "salt3_color_dispersion.dat")
    lcrv00file = os.path.join(model_dir, "salt3_lc_variance_0.dat")
    lcrv11file = os.path.join(model_dir, "salt3_lc_variance_1.dat")
    lcrv01file = os.path.join(model_dir, "salt3_lc_covariance_01.dat")

    # Check if all required files exist
    required_files = [m0file, m1file, clfile, cdfile, lcrv00file, lcrv11file, lcrv01file]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' not found.")

    # Create SALT3 model source
    salt3_source = sncosmo.SALT3Source(
        m0file=m0file,
        m1file=m1file,
        clfile=clfile,
    )

    #Add the dust effect as parameters
    sncosmo.Model(source=salt3_source,effects=[sncosmo.CCM89Dust()],effect_names=["host"],effect_frames=["rest"])


    # Register the source in SNCosmo (force=True in case it is already registered)
    sncosmo.registry.register(salt3_source, name="QinanSalt3", force=True)

    # Return the SNCosmo model using the custom source
    return sncosmo.Model(source=salt3_source,effects=[sncosmo.CCM89Dust()],effect_names=["host"],effect_frames=["rest"])