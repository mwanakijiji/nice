# Calculate the SNR of a given source and dispersion law

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb

from astropy.modeling.physical_models import BlackBody as blackbody_spectrum
from astropy import units as u
from astropy import constants as const

# import some fcns from radiometry_mct
import sys
from pathlib import Path
_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir / 'radiometry_mct'))
from radiometry_mct import photon_rate_per_pixel, wavel_to_nu, flux_nu_to_photons

# ----------------------------------------------------------------------------
# source stuff

def _simple_circular_pattern(type): # type='bool' or 'normalized'
    '''
    Make a circular pattern (units pixels for now)

    INPUTS:
    type (str): 
        'input': a boolean mask (i.e., a circle of transmission 1)
        'output': a normalized transmission amplitude (to preserve flux)

    OUTPUTS:
    pattern (array): 2D pattern (bool or normalized)
    '''

    pattern = np.zeros((100,100))
    # make central region a circle
    radius = 10 # pixels
    yy, xx = np.meshgrid(np.arange(pattern.shape[0]), np.arange(pattern.shape[1]), indexing='ij')
    mask = (xx - pattern.shape[0]/2)**2 + (yy - pattern.shape[1]/2)**2 <= radius**2
    pattern[mask] = 1 # note this is a transmission amplitude (not normalized)

    if type == 'input':
        pattern = pattern.astype(bool)
    elif type == 'output':
        pattern = pattern / np.sum(pattern) # normalize
    else:
        raise ValueError(f"Invalid type: {type}")

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
        flux: spectrum [photon/s/µm/cm²/sr]
        pattern: 2D light at the output (normalized to 1)
    '''

    bb = blackbody_spectrum(temperature=T) # B_nu(T) [erg/(cm2 Hz s sr)]
    nu = wavel_to_nu(wavelength)
    # B_nu/(h nu) is photons per Hz; c/lambda^2 converts that to per µm
    flux = flux_nu_to_photons(bb(wavelength), nu) * (const.c / wavelength**2)
    flux = flux.to(u.photon / u.s / u.um / u.cm**2 / u.sr)

    # 2D emission pattern ## ## TODO: put into dims of microns?
    ## ## TODO: COLLECTING AREA? SOLID ANGLE? 
    # stand-in for now
    solid_angle = 1e-5 * u.sr # physical meaning here?
    collecting_area = 0.5 * u.cm**2
    flux = flux * solid_angle * collecting_area

    pattern_output = _simple_circular_pattern(type='output')

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
    pattern_output = _simple_circular_pattern(type='output')

    return Source('laser', wavelength, flux, pattern_output)

# ----------------------------------------------------------------------------

class InterveningOptics:
    def __init__(self, name, pattern_input=None, pattern_output=None):
        self.name = name

        # 2D acceptance function (default simple circle)
        self.pattern_input = (
            _simple_circular_pattern(type='input') if pattern_input is None else pattern_input
        )

        # 2D emission function (default simple circle)
        self.pattern_output = (
            _simple_circular_pattern(type='output') if pattern_output is None else pattern_output
        )

    def transmit(self, wavelength, flux):
        '''
        Spread one input spectrum over pattern_output.

        INPUTS:
        wavelength (Quantity): wavelength grid [um]
        flux (array): spectrum on that grid, already scaled by the
            coupling fraction eta

        OUTPUTS:
        wavelength (Quantity): the same wavelength grid
        flux_output (array): cube (n_wavelength, ny, nx)
        '''

        flux_output = flux[:, np.newaxis, np.newaxis] * self.pattern_output
        return wavelength, flux_output

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


def spectral_footprint_example(camera, width=3,
                       wav_start=0.5*u.um, wav_end=5.0*u.um):
    '''
    Example spectral footprint (bool) and lambda map (um)

    INPUTS:
    camera (Camera): camera object (needed for detector array shape)
    width (float): width of the spectral footprint [pixels]
    wav_start (Quantity): start wavelength [um]
    wav_end (Quantity): end wavelength [um]

    OUTPUTS:
    footprint (array): boolean mask of the spectral footprint
    lambda_map (Quantity): wavelength map [um]
    '''

    n_y = camera.n_y
    n_x = camera.n_x

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
    s_max = (x_right - x_left) * np.sqrt(1 + m_slope**2)
    lambda_map = np.full((n_y, n_x), np.nan)
    lambda_map[on_segment] = wav_start + (s[on_segment] / s_max) * (wav_end - wav_start)

    # add units
    lambda_map = lambda_map * u.um

    footprint = on_segment.astype(float)

    return footprint, lambda_map


def apply_dispersion_law(wavel_input, flux_input, lambda_map):
    '''
    Apply the dispersion law to the flux entering the camera.

    The spectrum is sampled at each detector pixel's wavelength and
    multiplied by that column's wavelength width. Pixels that share a
    column split the flux so the strip conserves photons/s.

    INPUTS:
    wavel_input (Quantity, array): wavelength grid of the input spectrum [um]
    flux_input (array): spectrum [photons/s/um], or the
        (n_wavelength, ny, nx) cube from InterveningOptics.transmit.
        A cube is summed over the pattern first.
    lambda_map (Quantity, array): wavelength at each detector pixel [um];
        pixels off the footprint are NaN

    OUTPUTS:
    flux_disp_on_det (array): flux on the detector [photons/s/pix]
    '''

    ## ## TODO: MAKE A REAL DISPERSION BY ADDING GAUSSIANS TOGETHER ALONG THE SPECTRUM

    if hasattr(wavel_input, 'to'):
        wavel_um = np.asarray(wavel_input.to(u.um).value, dtype=float)
    else:
        wavel_um = np.asarray(wavel_input, dtype=float)

    flux = flux_input
    if np.ndim(flux) == 3:
        flux = np.sum(flux, axis=(1, 2))

    if hasattr(flux, 'unit'):
        flux_unit = flux.unit
        flux_1d = np.asarray(flux.value, dtype=float)
    else:
        flux_unit = None
        flux_1d = np.asarray(flux, dtype=float)

    if hasattr(lambda_map, 'to'):
        lam_um = np.asarray(lambda_map.to(u.um).value, dtype=float)
    else:
        lam_um = np.asarray(lambda_map, dtype=float)

    on = np.isfinite(lam_um)
    flux_at_pix = np.interp(lam_um, wavel_um, flux_1d, left=0.0, right=0.0)

    # one wavelength per column; dλ is the width of that column
    count = np.sum(on, axis=0)
    sum_lam = np.sum(np.where(on, lam_um, 0.0), axis=0)
    col_lam = np.divide(
        sum_lam, count, out=np.full(count.shape, np.nan), where=count > 0
    )
    good = np.isfinite(col_lam)
    dlam = np.zeros_like(col_lam)
    if np.count_nonzero(good) >= 2:
        dlam[good] = np.abs(np.gradient(col_lam[good]))

    n_per_col = count.astype(float)
    n_per_col[n_per_col == 0] = 1.0

    flux_disp = flux_at_pix * dlam[np.newaxis, :] / n_per_col[np.newaxis, :]
    flux_disp[~on] = 0.0

    if flux_unit is not None:
        flux_disp = flux_disp * flux_unit * u.um

    # flux_input is per µm; integrate over wavelength so both sides are photons/s
    integrated_input = np.trapezoid(flux_1d, wavel_um)
    integrated_output = np.sum(flux_disp.value if hasattr(flux_disp, 'value') else flux_disp)
    if not np.isclose(integrated_output, integrated_input, rtol=1e-2):
        raise ValueError(
            "Flux conservation not satisfied: "
            f"detector {integrated_output} != input {integrated_input}"
        )

    return flux_disp

# ----------------------------------------------------------------------------

class Camera:
    '''
    Hardware only
    '''
    def __init__(
        self, 
        name, 
        cuton_wavelength,
        cutoff_wavelength,
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
        self.cuton_wavelength = cuton_wavelength
        self.cutoff_wavelength = cutoff_wavelength
        self.quantum_efficiency = quantum_efficiency
        self.gain = gain
        self.pixel_pitch = pixel_pitch
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.integration_time = integration_time
        self.T_enclosure = T_enclosure
        self.n_y = n_y
        self.n_x = n_x

    @property
    def shape(self):
        return (self.n_y, self.n_x)


    def photons_to_electrons(camera, readout_true_photons):
        '''
        Convert the photons landing on the detector into electrons
        for a single read.

        INPUTS:
        camera (Camera): camera object
        readout_true_photons (ReadoutTruePhotons): readout object (real incident photon rate only!)

        OUTPUTS:
        sci_e (2D array): science signal [electrons/pix]
        bkg_e (2D array): background signal [electrons/pix]
        '''

        t = camera.integration_time # [s]
        qe = camera.quantum_efficiency # [unitless]
        gain = camera.gain # electrons/photon

        sci_e = readout_true_photons.sci_sig * qe * t * gain # [electrons/pix]
        #bkg_e = (readout_true_photons.enclosure_sig + readout_true_photons.other_bkg_sig) * qe * t * gain # [electrons/pix]
        bkg_e = readout_true_photons.enclosure_sig * qe * t * gain # [electrons/pix]

        return sci_e, bkg_e


    def add_counts(camera, sci_e, bkg_e):
        '''
        Add science, background, and detector counts for a single read.

        INPUTS:
        camera (Camera): camera object
        sci_e (2D array): science signal [electrons/pix]
        bkg_e (2D array): background signal [electrons/pix]

        OUTPUTS:
        total_e (2D array): total signal [electrons/pix]
        noise_e (2D array): noise [electrons/pix]
        '''

        t = camera.integration_time # [s]

        # dark current pedestal; /pix is the same count as the detector maps
        dark_pedestal_e = camera.dark_current * t #  [e-/pix]
        read_noise_e = camera.read_noise # [e-/pix rms]
        sci = _numeric(sci_e)
        bkg = _numeric(bkg_e)
        dark = float(_numeric(dark_pedestal_e))
        read = float(_numeric(read_noise_e))

        # read noise is per read, not a flux; add in variance
        noise_e = np.sqrt(sci + bkg + dark + read**2)
        total_e = sci + bkg + dark

        # dark pedestal still needed for later SNR calculation
        return dark_pedestal_e, total_e, noise_e


    def enclosure_signal(camera):
        '''
        Calculate the background photon flux on the detector
        (i.e., interior emission from camera enclosure)

        This assumes a simple blackbody emission from the underside 
        of a hemispherical surface.
        '''

        # use the old radiometry function, with no pinhole
        counts_per_pixel = photon_rate_per_pixel(
            T_hemisphere = camera.T_enclosure, 
            T_pinhole = camera.T_enclosure,
            width_pinhole = 0., 
            length_pinhole = 0., 
            dist_pinhole = 10, 
            nu_det_cuton = wavel_to_nu(camera.cuton_wavelength),
            nu_det_cutoff = wavel_to_nu(camera.cutoff_wavelength),
            gain_det = 1
            )

        # spread these counts on all pixels of the array (does not obey dispersion law)
        counts_all_pixels = counts_per_pixel * np.ones(camera.shape)

        return counts_all_pixels


# ----------------------------------------------------------------------------

class ReadoutTruePhotons:
    '''
    The photon map onto the detector (like a readout in terms of photons, 
    except without dark current or read noise)
    '''

    def __init__(
        self, 
        camera, 
        sci_sig=None, 
        enclosure_sig=None, 
        other_bkg_sig=None,
        ):

        self.sci_sig = sci_sig                 # science signal [ph/s/pix]
        self.enclosure_sig = enclosure_sig     # thermal emission from enclosure [ph/s/pix]
        self.other_bkg_sig = other_bkg_sig      # other signal [ph/s/pix]

# ----------------------------------------------------------------------------

def _numeric(quantity):
    return np.asarray(quantity.value if hasattr(quantity, 'value') else quantity, dtype=float)


def snr_spectrum(sci_e, bkg_e, dark_pedestal_e, read_noise_e, lambda_map):
    '''
    Signal-to-noise ratio of one read, as a spectrum.

    Pixels in one detector column share a wavelength. Science and
    background are summed over that column. Dark current and read noise
    stay one number per pixel and are multiplied by the pixel count.

    INPUTS:
    sci_e (2D array): science signal [electrons/pix]
    bkg_e (2D array): background signal [electrons/pix]
    dark_pedestal_e (float): dark current pedestal [electrons/pix]
    read_noise_e (float): read noise [electrons/pix rms]
    lambda_map (Quantity): wavelength at each detector pixel [um]

    OUTPUTS:
    wavelength (Quantity): wavelength of each spectral bin [um]
    snr (array): signal-to-noise ratio in that bin
    '''

    lam = _numeric(lambda_map)
    sci = _numeric(sci_e)
    bkg = _numeric(bkg_e)
    dark = float(_numeric(dark_pedestal_e))
    read = float(_numeric(read_noise_e))

    # spectrum footprint
    footprint_bool = np.isfinite(lam)

    # width of spectrum cross-section (consider this to be the wavelength bin)
    n_per_col = np.sum(footprint_bool, axis=0) 
    cols = n_per_col > 0
    n = n_per_col[cols].astype(float)

    # marginalize along the spectrum cross-section
    signal = np.sum(np.where(footprint_bool, sci, 0.0), axis=0)[cols]
    background = np.sum(np.where(footprint_bool, bkg, 0.0), axis=0)[cols]

    variance = signal + background + n * dark + n * read**2
    snr = np.divide(signal, np.sqrt(variance), where=variance > 0)

    # a marginalization of sorts to get an 'average' wavelength in each bin
    sum_lam = np.sum(np.where(footprint_bool, lam, 0.0), axis=0)
    wavelength = np.divide(sum_lam, n_per_col)[cols]
    if hasattr(lambda_map, 'unit'):
        wavelength = wavelength * lambda_map.unit

    return wavelength, snr


def main():

    # generate starting 'science'source
    wavel = np.linspace(0.5, 5, 1000) * u.um
    src_sci = source_blackbody(T=1000*u.K, wavelength=wavel)
    #src_sci = source_laser()

    # instantiate intervening optics (placeholder for now)
    io = InterveningOptics(name='simple_pass_through')

    # instantiate camera (also needed for dispersion law, readouts)
    camera = Camera(
        name='example_camera',
        cuton_wavelength = 3.0*u.um,
        cutoff_wavelength = 5.0*u.um,
        quantum_efficiency = 0.6,
        gain = 400.0 * u.electron/u.photon,
        pixel_pitch = 10.0 * u.um,
        dark_current = 0.0 * u.electron/u.s/u.pix, # rate
        read_noise = 727 * u.electron/u.pix, # rms
        integration_time = 1.0 * u.s, 
        T_enclosure = 300*u.K,
        n_y = 200, # pixels along y
        n_x = 200 # pixels along x
    )

    # generate dispersion law specific to detector array
    # (includes effect of prism and camera lens)
    footprint, lambda_map = spectral_footprint_example(camera)

    # instantiate photon map on the detector
    photon_map = ReadoutTruePhotons(camera)

    # now send the photons along--
    # from the original source into the intervening optics
    coupled = src_sci.pattern_output * io.pattern_input
    eta = np.sum(coupled)
    wavel_post_io, flux_post_io = io.transmit(src_sci.wavelength, src_sci.flux * eta)
    # from the intervening optics into the dispersing element

    # send photons through dispersion law and onto the detector photon map
    photon_map.sci_sig = apply_dispersion_law(wavel_post_io, flux_post_io, lambda_map)

    # calculate background photon flux on the detector photon map
    # (i.e., interior emission from camera enclosure)
    photon_map.enclosure_sig = camera.enclosure_signal()

    # convert photons landing on the detector into electrons
    sci_e, bkg_e = camera.photons_to_electrons(photon_map)

    # add science, background, and detector counts
    dark_pedestal_e, total_e, noise_e = camera.add_counts(sci_e, bkg_e)

    # collapse each detector column to one wavelength bin
    wavelength, snr = snr_spectrum(
        sci_e, bkg_e, dark_pedestal_e, camera.read_noise, lambda_map
    )

    # plot the results
    plt.plot(wavelength, snr)
    plt.xlabel('Wavelength [um]')
    plt.ylabel('SNR')
    plt.show()

    # save the results

    return



if __name__ == "__main__":

    main()