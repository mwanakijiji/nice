# Calculate the SNR of a given source and dispersion law

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb

from astropy.modeling.physical_models import BlackBody as blackbody_spectrum
from astropy import units as u

# ----------------------------------------------------------------------------
# source stuff

class Source:
    def __init__(self, name, wavelength, flux, pattern):
        self.name = name
        self.wavelength = wavelength   # array, e.g. um
        self.flux = flux               # spectrum on that grid
        self.pattern = pattern         # 2D light at the output (normalized to 1)

def source_blackbody(T=1200*u.K, wavelength=np.linspace(0.5, 5, 1000)*u.um, pattern=None):

    bb = blackbody_spectrum(temperature=T) # B_nu(T)
    flux = bb(wavelength)

    # 2D emission pattern ## ## TODO: put into dims of microns?
    pattern = np.zeros((100,100))
    # make central region a circle
    radius = 10 # pixels
    yy, xx = np.meshgrid(np.arange(pattern.shape[0]), np.arange(pattern.shape[1]), indexing='ij')
    mask = (xx - pattern.shape[0]/2)**2 + (yy - pattern.shape[1]/2)**2 <= radius**2
    pattern[mask] = 1
    pattern = pattern / np.sum(pattern) # normalize

    return Source('blackbody', wavelength, flux, pattern)

def source_laser(wavelength_line, power, wavelength, pattern, width=None):
    flux = np.zeros_like(wavelength)
    # put all the power in one bin, or a narrow Gaussian
    ...
    return Source('laser', wavelength, flux, pattern)

# ----------------------------------------------------------------------------

class InterveningOptics:
    def __init__(self, name, acceptance_in, emission_out):
        self.name = name

        # 2D acceptance function
        self.acceptance_in = acceptance_in

        # 2D emission function
        self.emission_out = emission_out

# ----------------------------------------------------------------------------

class DispersionLaw:
    def __init__(self, name, dispersion_law):
        self.name = name
        self.dispersion_law = dispersion_law


# ----------------------------------------------------------------------------

class Camera:
    def __init__(
        self, 
        name, 
        quantum_efficiency, 
        pixel_scale,
        dark_current, 
        read_noise, 
        integration_time,
        T_enclosure
    ):
        self.name = name
        self.quantum_efficiency = quantum_efficiency
        self.pixel_scale = pixel_scale
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.integration_time = integration_time
        self.T_enclosure = T_enclosure

# ----------------------------------------------------------------------------

def main():

    ipdb.set_trace()

    # generate the source
    #src = source_blackbody(T=1000, wavelength=wav, pattern=pattern)

    # debug
    #test = source_blackbody(T=1200*u.K, wavelength=np.linspace(0.5, 5, 1000)*u.um)
    #plt.plot(test.wavelength,test.flux)
    #plt.show()


    # apply the acceptance angle

    # intervening optics (placeholder for now)

    # apply dispersion law (includes effect of prism and camera lens)

    # calculate source photon flux on the detector

    # calculate background photon flux on the detector

    # calculate signal-to-noise ratio


    # plot the results

    # save the results

    return



if __name__ == "__main__":

    main()