"""
swift_visualization.py
========================

This script provides two analyses of the Swift Type Ia supernova sample:

1.  A histogram of redshift values for all SNe Ia listed in the
    classification table.  The redshift column is read from the CSV file
    provided by the Swift Optical/Ultraviolet Supernova Archive (SOUSA)
    team (e.g., ``Swift_Ia_light_curves_list.csv``).  Any non‑numeric
    values are ignored.  The histogram is written to ``swift_redshift_hist.png``.

2.  Absolute‑magnitude light curves for each filter.  The script
    searches a directory of downloaded ``.dat`` light‑curve files
    (default ``/home/qinanwang/data/swift``) for files whose names begin
    with the supernova name.  For each supernova a distance modulus is
    computed using a simple flat ΛCDM cosmology (H₀ = 70 km/s/Mpc,
    Ωₘ = 0.3, ΩΛ = 0.7) via numerical integration of
    ``1/sqrt(Ωₘ(1+z)^3 + ΩΛ)``.  Apparent magnitudes are converted to
    absolute magnitudes by subtracting the distance modulus.  Data with
    ``NULL`` magnitudes or errors are skipped.  One PNG plot per
    filter is written with all light‑curve points overlaid; the
    y‑axis is inverted to follow astronomical convention.

Usage
-----

Run the script with a Python interpreter.  Adjust the paths to the
classification file and data directory as needed.

.. code-block:: bash

   python swift_visualization.py \
       --info-csv Swift_Ia_light_curves_list.csv \
       --data-dir /home/qinanwang/data/swift \
       --output-dir output_plots

The script will create ``output_plots`` if it does not exist and
populate it with PNG files for the histogram and each filter.
"""

import argparse
import os
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure non‑interactive backend
import matplotlib.pyplot as plt

try:
    # Prefer astropy if available.  Astropy provides a robust cosmology framework
    # including a distance modulus calculation.  If astropy is installed in
    # the user's environment, this will be used automatically.  Otherwise the
    # code falls back to a manual ΛCDM computation defined below.
    from astropy.cosmology import FlatLambdaCDM  # type: ignore
    _USE_ASTROPY = True
    # Define a global cosmology instance for H0=70 km/s/Mpc, Ωm=0.3.
    _ASTRO_COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
except ImportError:
    _USE_ASTROPY = False
    _ASTRO_COSMO = None


def lumdist(z: float, n: int = 1000, H0: float = 70.0,
            Omega_m: float = 0.3, Omega_L: float = 0.7) -> float:
    """Return the luminosity distance at redshift `z` [Mpc].

    A simple numerical integration of the flat ΛCDM luminosity distance
    integral:

    d_L(z) = (c/H₀) * (1 + z) * ∫₀ᶻ dz' / √(Ωₘ (1+z')³ + ΩΛ).

    Parameters
    ----------
    z : float
        Redshift.
    n : int, optional
        Number of integration steps for trapezoidal rule.
    H0 : float, optional
        Hubble constant in km/s/Mpc.
    Omega_m : float, optional
        Matter density parameter.
    Omega_L : float, optional
        Cosmological constant density parameter.

    Returns
    -------
    float
        Luminosity distance in Mpc.
    """
    c = 299792.458  # speed of light [km/s]
    # set up the integration grid
    z_vals = np.linspace(0.0, z, n + 1)
    E = np.sqrt(Omega_m * (1.0 + z_vals) ** 3 + Omega_L)
    # trapezoidal rule integration of 1/E(z)
    integral = np.trapz(1.0 / E, z_vals)
    d_c = (c / H0) * integral  # comoving distance
    return d_c * (1.0 + z)


def distance_modulus(z: float) -> float:
    """Return the distance modulus μ for redshift ``z``.

    By default, if the optional dependency **astropy** is available, this
    function uses ``astropy.cosmology.FlatLambdaCDM.distmod`` with
    H₀ = 70 km/s/Mpc and Ωₘ = 0.3.  Otherwise it falls back to the
    manual integration implemented in :func:`lumdist`.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        Distance modulus μ.
    """
    if _USE_ASTROPY:
        # astropy returns a Quantity; `.value` extracts the numeric value
        # in magnitudes (unitless).  See astropy.cosmology documentation.
        return float(_ASTRO_COSMO.distmod(z).value)
    # Manual fallback: compute luminosity distance using trapezoidal rule
    d_l_mpc = lumdist(z)
    d_pc = d_l_mpc * 1e6
    return 5.0 * math.log10(d_pc) - 5.0


def load_info(info_csv: str) -> pd.DataFrame:
    """Load the classification/metadata CSV and return as DataFrame.

    The function expects at least columns ``SNname`` and ``redshift``.

    Parameters
    ----------
    info_csv : str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        DataFrame with SNname and redshift columns.
    """
    info = pd.read_csv(info_csv)
    return info[['SNname', 'redshift']].copy()


def histogram_redshift(info: pd.DataFrame, out_path: Path) -> None:
    """Plot histogram of redshift values and save as PNG."""
    redshifts = pd.to_numeric(info['redshift'], errors='coerce').dropna()
    plt.figure(figsize=(6, 4))
    plt.hist(redshifts, bins=20, edgecolor='black')
    plt.xlabel('Redshift')
    plt.ylabel('Number of SNe Ia')
    plt.title('Redshift distribution of Swift SNe Ia')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_dat_file(path: Path, mu: float) -> dict:
    """Parse a Swift light‐curve .dat file.

    Returns a mapping of filter names to arrays of (mjd, abs_mag, err).
    Apparent magnitudes labelled ``NULL`` are skipped.  Absolute magnitudes
    are computed by subtracting the supplied distance modulus.

    Parameters
    ----------
    path : Path
        Path to the .dat file.
    mu : float
        Distance modulus for the supernova.

    Returns
    -------
    dict
        Mapping from filter name to list of (mjd, absolute magnitude, error).
    """
    curves = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # expecting at least: filter, mjd, mag, magErr
            filt = parts[0]
            try:
                mjd = float(parts[1])
            except (IndexError, ValueError):
                continue
            mag_str = parts[2] if len(parts) > 2 else ''
            err_str = parts[3] if len(parts) > 3 else ''
            # skip if magnitude is null or missing
            if mag_str.upper() == 'NULL' or err_str.upper() == 'NULL':
                continue
            try:
                mag = float(mag_str)
                err = float(err_str)
            except ValueError:
                continue
            abs_mag = mag - mu
            curves.setdefault(filt, []).append((mjd, abs_mag, err))
    return curves


def plot_lightcurves(curves: dict, filter_name: str, out_path: Path) -> None:
    """Plot absolute magnitude light curves for a single filter.

    Points from different supernovae are all plotted together.  The y‐axis
    is inverted so that brighter objects (lower magnitude) appear at the
    top of the plot.  Individual points are plotted without connecting
    lines to emphasize the variety of sampling.

    Parameters
    ----------
    curves : dict
        Mapping from supernova name to a list of (mjd, abs_mag, err) points.
    filter_name : str
        The filter name (e.g., 'UVW2', 'U', 'B').
    out_path : Path
        File path to save the PNG plot.
    """
    plt.figure(figsize=(7, 5))
    for sn, points in curves.items():
        if not points:
            continue
        # sort by MJD for aesthetic ordering
        points_sorted = sorted(points, key=lambda x: x[0])
        mjd_vals = [p[0] for p in points_sorted]
        abs_mag_vals = [p[1] for p in points_sorted]
        errs = [p[2] for p in points_sorted]
        # use small markers for clarity; no legend to avoid overcrowding
        plt.errorbar(mjd_vals, abs_mag_vals, yerr=errs, fmt='.', markersize=2, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('MJD')
    plt.ylabel('Absolute Magnitude')
    plt.title(f'Swift SNe Ia absolute‐magnitude light curves ({filter_name})')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(info_csv: str, data_dir: str, output_dir: str) -> None:
    # ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load metadata and plot histogram
    info = load_info(info_csv)
    hist_path = out_dir / 'swift_redshift_hist.png'
    histogram_redshift(info, hist_path)
    print(f'Wrote redshift histogram to {hist_path}')

    # precompute distance modulus for each SN
    mu_cache = {}
    for _, row in info.dropna(subset=['redshift']).iterrows():
        sn = row['SNname']
        try:
            z = float(row['redshift'])
        except (TypeError, ValueError):
            continue
        mu_cache[sn] = distance_modulus(z)

    # prepare per‑filter dictionary: filter -> sn -> list of points
    filter_curves = {}

    for sn, mu in mu_cache.items():
        # find all dat files starting with this SN name
        for entry in os.scandir(data_dir):
            if entry.is_file() and entry.name.startswith(sn) and entry.name.endswith('.dat'):
                curves = parse_dat_file(Path(entry.path), mu)
                for filt, points in curves.items():
                    filter_curves.setdefault(filt, {}).setdefault(sn, []).extend(points)

    # generate a plot for each filter
    for filt_name, curves_by_sn in filter_curves.items():
        out_path = out_dir / f'swift_lightcurves_{filt_name}.png'
        plot_lightcurves(curves_by_sn, filt_name, out_path)
        print(f'Wrote light curve plot for {filt_name} to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise Swift SNe Ia data")
    parser.add_argument('--info-csv', default='Swift_Ia_light_curves_list.csv',
                        help='Path to the CSV file containing SN names and redshifts')
    parser.add_argument('--data-dir', default='/home/qinanwang/data/swift',
                        help='Directory containing downloaded .dat light‑curve files')
    parser.add_argument('--output-dir', default='swift_plots',
                        help='Directory in which to save generated plots')
    args = parser.parse_args()
    main(args.info_csv, args.data_dir, args.output_dir)