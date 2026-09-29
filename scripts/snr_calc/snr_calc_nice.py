# Calculate the SNR of a given source and dispersion law

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb

from astropy.modeling.physical_models import BlackBody as blackbody_spectrum
from astropy import units as u

# ----------------------------------------------------------------------------
# source stuff

def _simple_circular_pattern():
    # a circular pattern (units pixels for now)

    pattern = np.zeros((100,100))
    # make central region a circle
    radius = 10 # pixels
    yy, xx = np.meshgrid(np.arange(pattern.shape[0]), np.arange(pattern.shape[1]), indexing='ij')
    mask = (xx - pattern.shape[0]/2)**2 + (yy - pattern.shape[1]/2)**2 <= radius**2
    pattern[mask] = 1
    pattern = pattern / np.sum(pattern) # normalize

    return pattern


class Source:
    def __init__(self, name, wavelength, flux, pattern_output):
        self.name = name
        self.wavelength = wavelength   # array [um]
        self.flux = flux               # spectrum on that grid [unitless; will depend on source ## ## TODO: make consistent]
        self.pattern_output = pattern_output         # 2D light at the output (normalized to 1)

def source_blackbody(T=1200*u.K, wavelength=np.linspace(0.5, 5, 1000)*u.um, pattern_output=None):
    '''
    Return a BB Source object, given parameter inputs

    INPUTS:
    T: temperature [K]
    wavelength: wavelength grid [um]
    pattern: 2D emission pattern [normalized to 1]

    OUTPUTS:
    Source object, updated parameters:
        wavelength: wavelengths [um]
        flux: spectrum [unitless]
        pattern: 2D light at the output (normalized to 1)
    '''

    bb = blackbody_spectrum(temperature=T) # B_nu(T)
    flux = bb(wavelength)

    # 2D emission pattern ## ## TODO: put into dims of microns?
    pattern_output = _simple_circular_pattern()

    return Source('blackbody', wavelength, flux, pattern_output)


def source_laser(pattern_output=None):
    '''
    Read in a laser spectrum from a file and return a Source object

    INPUTS:
    pattern_output: 2D emission pattern [normalized to 1]

    OUTPUTS:
    Source object, updated parameters:
        wavelength: wavelengths [um]
        flux: spectrum [unitless]
        pattern: 2D light at the output (normalized to 1)
    '''

    #flux = np.zeros_like(wavelength)
    # put all the power in one bin, or a narrow Gaussian
    
    # read in laser spectrum 
    # ## PLACEHOLDER
    file_name = (
        '/Users/eckhartspalding/Documents/git.repos/nice/scripts/'
        'snr_calc/data/example_laser_source.csv'
    )
   
    df_laser = pd.read_csv(file_name, sep=r"\s+", skiprows=2)
    wavelength = df_laser['wavel_um'].values * u.um
    flux = df_laser['psd_dbm_nm'].values

    # 2D emission pattern ## ## TODO: put into dims of microns?
    pattern_output = _simple_circular_pattern()

    return Source('laser', wavelength, flux, pattern_output)

# ----------------------------------------------------------------------------

class InterveningOptics:
    def __init__(self, name, pattern_input=None, pattern_output=None):
        self.name = name

        # 2D acceptance function (default simple circle)
        self.pattern_input = (
            _simple_circular_pattern() if pattern_input is None else pattern_input
        )

        # 2D emission function (default simple circle)
        self.pattern_output = (
            _simple_circular_pattern() if pattern_output is None else pattern_output
        )

# ----------------------------------------------------------------------------

class DispersionLaw:
    def __init__(self, name, dispersion_law, detector_array_canvas):
        self.name = name
        self.dispersion_law = dispersion_law
        self.detector_array_canvas = detector_array_canvas

def dispersion_law_example(wavelength):
    '''
    Example dispersion law
    '''

    m = 10./165.
    b = 69.

    detector_array_canvas = np.zeros((200, 200))

    yy, xx = np.meshgrid(
        np.arange(detector_array_canvas.shape[0]), 
        np.arange(detector_array_canvas.shape[1]), 
        indexing='ij')

    return wavelength


def spectral_footprint_example(n_y, n_x, width=3,
                       wav_start=0.5*u.um, wav_end=5.0*u.um):
    '''
    Example spectral footprint (bool) and lambda map (um)

    INPUTS:
    n_y: number of rows in the detector array
    n_x: number of columns in the detector array
    width: width of the spectral footprint [pixels]
    wav_start: start wavelength [um]
    wav_end: end wavelength [um]

    OUTPUTS:
    footprint (array): boolean mask of the spectral footprint
    lambda_map (Quantity): wavelength map [um]
    '''

    # slope and intercept of dispersion law
    m_slope = 10./165.
    b_intercept = 69.

    yy, xx = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing='ij')
    # perpendicular distance to y = m x + b
    dist = np.abs(yy - (m_slope * xx + b_intercept)) / np.sqrt(1 + m_slope**2)
    on_strip = dist <= width / 2
    # part of the line that actually hits the array
    y_line = m_slope * np.arange(n_x) + b_intercept
    x_on_det = np.where((y_line >= 0) & (y_line < n_y))[0]
    x_left, x_right = x_on_det[0], x_on_det[-1]
    # only the restricted segment (leftmost to rightmost on-detector)
    on_segment = on_strip & (xx >= x_left) & (xx <= x_right)
    # path length along the line, from the left end
    s = (xx - x_left) * np.sqrt(1 + m_slope**2)
    s_m_slopeax = (x_right - x_left) * np.sqrt(1 + m**2)
    lambda_map = np.full((n_y, n_x), np.nan)
    lambda_map[on_segment] = wav_start + (s[on_segment] / s_max) * (wav_end - wav_start)

    # add units
    lambda_map = lambda_map * u.um

    footprint = on_segment.astype(float)

    return footprint, lambda_map


def apply_dispersion_law(wavel_input, flux_input, lambda_map):
    '''
    Apply the dispersion law to the flux entering the camera

    INPUTS: 
    wavel_input (Quantity, array): wavelength grid of the input spectrum [um]
    flux_input (Quantity, array): flux of the input spectrum as fcn of wavelength [photons/s/um]
    lambda_map (Quantity, array): wavelength map of the detector array [um]

    OUTPUTS:
    flux_disp_on_det (array): flux on the detector [photons/s/pix]
    '''

    return TBD

# ----------------------------------------------------------------------------

class Camera:
    '''
    Hardware only
    '''
    def __init__(
        self, 
        name, 
        quantum_efficiency, 
        gain,
        pixel_pitch,
        dark_current, 
        read_noise, 
        integration_time,
        T_enclosure, 
        n_y=200, 
        n_x=200
    ):
        self.name = name
        self.quantum_efficiency = quantum_efficiency
        self.pixel_pitch = pixel_pitch
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.integration_time = integration_time
        self.T_enclosure = T_enclosure

    @property
    def shape(self):
        return (self.n_y, self.n_x)



    def sci_signal_electrons(self, flux_disp_on_det_pix):
        '''
        'Science' signal in e

        INPUTS:
        flux_disp_on_det_pix: flux on the detector [photons/s/pix]

        OUTPUTS:
        signal_electrons: signal in electrons
        '''

        return flux_disp_on_det * self.quantum_efficiency * self.integration_time


    def noise_electrons(self, signal, background):
        return np.sqrt(
            signal + background
            + self.dark_current * self.integration_time
            + self.read_noise**2
        )

    def snr(self, flux_on_det, background_flux):
        s = self.signal_electrons(flux_on_det)
        b = self.signal_electrons(background_flux)
        return s / self.noise_electrons(s, b)


# ----------------------------------------------------------------------------

class Readout:
    '''
    A readout from the detector
    '''

    def __init__(
        self, 
        camera, 
        sci_sig=None, 
        enclosure_sig=None, 
        other_bkg_sig=None
        ):

        self.sci_sig = sci_sig                 # science signal [ph/s/pix]
        self.enclosure_sig = enclosure_sig     # thermal emission from enclosure [ph/s/pix]
        self.other_bkg_sig = other_bkg_sig      # other signal [ph/s/pix]


# ----------------------------------------------------------------------------

def main():

    ipdb.set_trace()

    # generate starting source
    wavel = np.linspace(0.5, 5, 1000) * u.um
    src = source_blackbody(T=1000*u.K, wavelength=wavel)
    ipdb.set_trace()

    # generate intervening optics (placeholder for now)
    io = InterveningOptics(name='simple_pass_through')

    # instantiate camera (also needed for dispersion law, readouts)
    camera = Camera(
        name='example_camera',
        quantum_efficiency=0.9,
        gain=1.0,
        pixel_pitch=10.0*u.um,
        dark_current=0.0,
        read_noise=0.0,
        integration_time=1.0*u.s, 
        n_y=200, 
        n_x=200
    )

    # generate dispersion law specific to detector array
    # (includes effect of prism and camera lens)
    footprint, lambda_map = spectral_footprint_example(
        n_y=camera.n_y, 
        n_x=camera.n_x
        )

    # instantiate readout
    footprint, lambda_map = spectral_footprint_example(
        n_y=camera.n_y, 
        n_x=camera.n_x, m=10./165., b=69.)




    # apply dispersion law to spectrum entering the camera
    flux_disp_on_det = apply_dispersion_law(
        wavel, src.flux, lambda_map
    )

    # calculate science photon flux on the detector

    # calculate background photon flux on the detector

    # calculate signal-to-noise ratio


    # plot the results

    # save the results

    return



if __name__ == "__main__":

    main()