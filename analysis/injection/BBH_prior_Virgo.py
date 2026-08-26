#!/usr/bin/env python3

"""
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/Virgo-DeepClean.git
Creation Date   : February 2024
Description     : This script generates GW injection parameter estimation priors using the Bilby framework.
                  It calculates the mass range for ISCO frequencies and sets up priors for GW signal parameters.
                  The priors are saved to a file for subsequent analysis, facilitating the assessment of GW signal
                  models and parameter estimation techniques in gravitational wave astronomy.
Usage           : Run the script directly without any arguments. Adjust the constants as needed for different analyses.
---------------------------------------------------------------------------------------------------
"""

import bilby
import numpy as np
import os


# Constants
G = 6.6743e-11  # Newton's gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light in vacuum (m/s)
Sun_Mass = 1.988409870698051e30  # Mass of the sun (kg)
fmin = 15  # Minimum frequency for analysis (Hz)
fmax = 420  # Maximum frequency for analysis (Hz)

outdir = "./data"  # Output directory for the prior file
os.makedirs(outdir, exist_ok=True)


def gw_frequency_at_isco(mass):
    """Calculate GW frequency at ISCO for a given total mass."""
    return c**3 / (6**1.5 * np.pi * G * mass * Sun_Mass)


def get_M_for_fisco(fisco):
    """Calculate the mass for a given GW frequency at ISCO."""
    return c**3 / (6**1.5 * np.pi * G * fisco * Sun_Mass)


# Calculate minimum and maximum mass for the specified frequency range
Mmin = get_M_for_fisco(fisco=fmax)
Mmax = get_M_for_fisco(fisco=fmin)

# Initialize PriorDict
priors = bilby.core.prior.PriorDict()

# Define parameter priors for GW signal analysis
priors["mass_1"] = bilby.core.prior.Uniform(
    minimum=5.0,
    maximum=50.0,
    name="mass_1",
    latex_label="$m_1$",
    unit="$M_{\odot}$",
    boundary="reflective",
)
priors["mass_2"] = bilby.core.prior.Uniform(
    minimum=5.0,
    maximum=50.0,
    name="mass_2",
    latex_label="$m_2$",
    unit="$M_{\odot}$",
    boundary="reflective",
)
priors["total_mass"] = bilby.core.prior.Constraint(
    name="total_mass", minimum=Mmin, maximum=Mmax, latex_label="$M$", unit="$M_{\odot}$"
)
priors["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(
    name="luminosity_distance",
    minimum=10,
    maximum=1000,
    latex_label="$D_L$",
    unit="Mpc",
)
priors["mass_ratio"] = bilby.core.prior.Constraint(
    name="mass_ratio", minimum=0.4, maximum=1.0, latex_label="$q$"
)
priors["a_1"] = bilby.core.prior.Uniform(
    name="a_1", minimum=0, maximum=0.99, latex_label="$a_1$", boundary="reflective"
)
priors["a_2"] = bilby.core.prior.Uniform(
    name="a_2", minimum=0, maximum=0.99, latex_label="$a_2$", boundary="reflective"
)
priors["tilt_1"] = 0
priors["tilt_2"] = 0
priors["phi_12"] = 0
priors["phi_jl"] = 0
priors["dec"] = bilby.core.prior.Cosine(
    name="dec", latex_label="$\delta$", boundary="reflective"
)
priors["ra"] = bilby.core.prior.Uniform(
    name="ra",
    minimum=0,
    maximum=2 * np.pi,
    latex_label="$\\alpha$",
    boundary="periodic",
)
priors["theta_jn"] = bilby.core.prior.Sine(
    name="theta_jn", latex_label="$\\theta_{jn}$", boundary="reflective"
)
priors["psi"] = bilby.core.prior.Uniform(
    name="psi", minimum=0, maximum=np.pi, latex_label="$\psi$", boundary="periodic"
)
priors["phase"] = bilby.core.prior.Uniform(
    name="phase",
    minimum=0,
    maximum=2 * np.pi,
    latex_label="$\phi$",
    boundary="periodic",
)

# Write the priors to a file using PriorDict.to_file() method

label = f"BBH_V1_{fmin}-{fmax}_Hz"
priors.to_file(outdir=outdir, label=label),

print(f"Priors file {os.path.join(outdir, label)} has been generated successfully.")
