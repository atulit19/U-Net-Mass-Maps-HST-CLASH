# ===================== IMPORTS =====================
import os
import logging
import warnings

import numpy as np
import h5py
import matplotlib.pyplot as plt

from astropy.utils.exceptions import AstropyWarning
from scipy.ndimage import gaussian_filter

import Pk_library as PKL


# ===================== SETTINGS =====================
warnings.simplefilter('ignore', category=AstropyWarning)
logging.getLogger('astropy.wcs').setLevel(logging.ERROR)

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})


# ===================== DATA LOADER =====================
def fetch_predicted_data_CLASH(file_h5, prediction_models):
    """
    Load ML predictions and true mass maps from CLASH HDF5 file.
    """
    predictions = {}

    with h5py.File(file_h5, "r") as f:
        cluster_names = list(f.keys())

        for model in prediction_models:
            predictions[model] = np.array([
                f[f"{cl}/predictions/{model}"][:] for cl in cluster_names
            ])

        true_mass = np.array([
            f[f"{cl}/true_mass"][:] for cl in cluster_names
        ])

    return predictions, true_mass


# ===================== SURFACE DENSITY =====================
def surface_density_profile(projected_mass):
    """
    Compute radial surface density profile Σ(R).
    """
    map_size_kpc = 500.0
    n_pix = projected_mass.shape[0]
    pixel_scale = map_size_kpc / n_pix

    center = (n_pix - 1) / 2
    X, Y = np.meshgrid(np.arange(n_pix), np.arange(n_pix))

    r_kpc = np.sqrt((X - center)**2 + (Y - center)**2) * pixel_scale

    bins = np.logspace(np.log10(1.0), np.log10(map_size_kpc / 2), 12)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    r_flat = r_kpc.ravel()
    mass_flat = projected_mass.ravel()

    mask = np.isfinite(mass_flat)
    r_flat = r_flat[mask]
    mass_flat = mass_flat[mask]

    digit = np.digitize(r_flat, bins)

    sigma = np.zeros(len(bins) - 1)
    sigma_err = np.zeros_like(sigma)

    area = np.pi * (bins[1:]**2 - bins[:-1]**2)

    for i in range(1, len(bins)):
        vals = mass_flat[digit == i]

        if len(vals) > 0:
            total_mass = np.sum(vals)
            sigma[i - 1] = total_mass / area[i - 1]
            sigma_err[i - 1] = np.std(vals) * np.sqrt(len(vals)) / area[i - 1]
        else:
            sigma[i - 1] = np.nan
            sigma_err[i - 1] = np.nan

    valid = np.isfinite(sigma)

    return sigma[valid], sigma_err[valid], bin_centers[valid]


# ===================== POWER SPECTRUM =====================
def compute_power_spectrum(image):
    """
    Compute 2D power spectrum of a mass map.
    """
    BoxSize = 500  # kpc
    MAS = 'None'
    threads = 1

    delta = image.astype(np.float64).copy()

    nonzero_mask = delta > 0
    if nonzero_mask.sum() > 0:
        delta[~nonzero_mask] = delta[nonzero_mask].mean()

    mean = delta.mean()
    delta = ((delta - mean) / mean).astype(np.float32)

    Pk2D = PKL.Pk_plane(delta, BoxSize, MAS, threads, verbose=False)

    k = Pk2D.k
    Pk = Pk2D.Pk

    wavelength = (2 * np.pi) / k

    return wavelength, Pk


# ===================== EXAMPLE USAGE =====================
if __name__ == "__main__":

    FILE = "CLASH-Spectrosopic-Sample.h5"
    MODELS = ['F606w', 'F625w', 'F775w', 'F814w', 'F850lp',
              'MultiChannel', 'MultiEncoder']

    # Load data
    predictions, true_mass = fetch_predicted_data_CLASH(FILE, MODELS)

    # Select one cluster (index 0) and one model
    model_name = 'MultiEncoder'
    cluster_idx = 0

    pred_map = predictions[model_name][cluster_idx]
    true_map = true_mass[cluster_idx]

    # ---------- Surface Density ----------
    sigma_pred, err_pred, r = surface_density_profile(pred_map)
    sigma_true, err_true, _ = surface_density_profile(true_map)

    plt.figure()
    plt.loglog(r, sigma_true, label='True')
    plt.loglog(r, sigma_pred, '--', label=f'{model_name}')
    plt.xlabel("R [kpc]")
    plt.ylabel(r"$\Sigma(R)$")
    plt.legend()
    plt.title("Surface Density Profile")
    plt.show()

    # ---------- Power Spectrum ----------
    wl_true, pk_true = compute_power_spectrum(true_map)
    wl_pred, pk_pred = compute_power_spectrum(pred_map)

    plt.figure()
    plt.loglog(wl_true, pk_true, label='True')
    plt.loglog(wl_pred, pk_pred, '--', label=f'{model_name}')
    plt.xlabel("Wavelength [kpc]")
    plt.ylabel("P(k)")
    plt.legend()
    plt.title("Power Spectrum")
    plt.show()
