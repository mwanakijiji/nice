import numpy as np
from astropy.modeling import models
from astropy import units as u
from matplotlib import pyplot as plt
from astropy import constants as const
from astropy.visualization import quantity_support
import scipy.integrate as integrate


def wavel_to_nu(wav):
    # note wav needs to have units
    return (const.c / wav).to(u.Hz)


def flux_nu_to_photons(flux_nu, nu):
    # input variables need to have units
    return (flux_nu / (const.h * nu)).decompose() * u.photon


# fcn to find photon rate per pixel
def photon_rate_per_pixel(
    T_hemisphere, 
    T_pinhole, 
    width_pinhole, 
    length_pinhole, 
    dist_pinhole, 
    nu_det_cuton,
    nu_det_cutoff,
    gain_det = 1
    ):
    '''
    Calculate the photon rate per pixel for a given hemisphere temperature, pinhole temperature, and pinhole dims

    INPUTS:
    T_hemisphere: temperature of the hemisphere (K)
    T_pinhole: temperature of the pinhole (K)
    width_pinhole: width of the pinhole (mm)
    length_pinhole: length of the pinhole (mm)
    dist_pinhole: distance of the pinhole from the detector (mm)
    nu_det_cuton: detector cut-on frequency (Hz)
    nu_det_cutoff: detector cut-off frequency (Hz)
    gain_det: gain of the detector (e/ADU)

    OUTPUTS:
    photon_rate_per_pixel: photon rate per pixel (ph s-1)
    '''

    # pinhole area
    A_pin = width_pinhole * length_pinhole

    # distance of pinhole from detector
    L_pinhole = dist_pinhole

    #pitch_pixels = 18*u.um
    # detector area
    A_det = (3.7*u.cm)**2
    N_pixels = (2048*u.pix)**2
    qe_det = 0.8
    #wavel_det_cuton = 1.0*u.um  
    #wavel_det_cutoff = 5.3*u.um

    bb_nu_hemisphere = models.BlackBody(temperature=T_hemisphere)
    bb_nu_pinhole = models.BlackBody(temperature=T_pinhole)

    wavel_array = np.linspace(0.1, 10, 100) * u.um

    nu_hemisphere = wavel_to_nu(wavel_array)
    nu_pinhole = wavel_to_nu(wavel_array)

    #wav_hemisphere = np.linspace(0.1, 10, 100) * u.um

    flux_nu_hemisphere_from_lambda = bb_nu_hemisphere(wavel_array)
    flux_nu_hemisphere_from_nu = bb_nu_hemisphere(nu_hemisphere)
    flux_nu_pinhole_from_lambda = bb_nu_pinhole(wavel_array)
    flux_nu_pinhole_from_nu = bb_nu_pinhole(nu_hemisphere)

    # get photon flux: (B_nu / h*nu)
    flux_nu_hemisphere_photons = flux_nu_to_photons(
                                            flux_nu = flux_nu_hemisphere_from_nu, 
                                            nu = nu_hemisphere
                                            )
    flux_nu_pinhole_photons = flux_nu_to_photons(
                                            flux_nu = flux_nu_pinhole_from_nu, 
                                            nu = nu_hemisphere
                                            )

    # integrate (note limits of integration and the ordinate appear reversed;
    # appearminus sign: effectively is integration is in reverse, since we're using nu, not wavel)
    #nu_lim_low = nu_det_cutoff # corresponds to detector wavelength cuton
    #nu_lim_high = nu_det_cuton # wavelength cutoff

    #nu_hemisphere_ascending = np.flip(nu_hemisphere)
    #flux_nu_hemisphere_nu_ascending = np.flip(flux_nu_hemisphere_photons)

    # set the limits of integration, between the detector cuton and cutoff
    idx = np.where(np.logical_and(nu_hemisphere > nu_det_cutoff, nu_hemisphere < nu_det_cuton))

    nu_hemisphere_limited = nu_hemisphere[idx]
    flux_nu_hemisphere_photons_limited = flux_nu_hemisphere_photons[idx]
    flux_nu_pinhole_photons_limited = flux_nu_pinhole_photons[idx]

    # integral_1 (note the minus sign, since we're integrating from small nu to large nu)
    # int (S * B_photons_hemisphere) dnu, where S is detector response (i.e., QE) and 
    # B_photons_hemisphere is the photon flux from the hemisphere
    integral_1_hemisphere = - qe_det * np.trapz(y=flux_nu_hemisphere_photons_limited, x=nu_hemisphere_limited)

    # integral_2, where we include the pinhole:
    # int (S * (B_photons_pinhole - B_photons_hemisphere)) dnu
    integral_2_hemisphere_pinhole = - qe_det * np.trapz(y=(flux_nu_pinhole_photons_limited - flux_nu_hemisphere_photons_limited), x=nu_hemisphere_limited)

    # put it all together
    D_photons_pix = ((A_det/N_pixels)*u.pix**2) * ((np.pi*u.rad**2) * integral_1_hemisphere + ((A_pin/L_pinhole**2)*u.rad**2) * integral_2_hemisphere_pinhole)
    D_photons_pix = D_photons_pix.decompose()

    # covnert to counts (assumes 1 photon -> 1 electron)
    D_counts_pix = D_photons_pix / gain_det

    return D_counts_pix



def main():
    '''
    Teff_room = 293*u.K # ambient room
    Teff_det = 55*u.K # detector
    Teff_inner = 90*u.K # inner sanctum
    Teff_outer = 273*u.K # outer radiation shield (pinhole originally mounted here)

    Teff_hemisphere = 293*u.K # hemisphere
    Teff_pinhole = 273*u.K # pinhole

    # pinhole area
    W_pinhole = 1*u.mm
    L_pinhole = 1*u.mm
    A_pin = W_pinhole * L_pinhole

    # distance of pinhole from detector
    L_pinhole = 30*u.mm
    '''

    # set some constants 
    #pitch_pixels = 18*u.um
    # detector area
    A_det = (3.7*u.cm)**2
    N_pixels = (2048*u.pix)**2
    qe_det = 0.8
    wavel_det_cuton = 1.0*u.um  
    wavel_det_cutoff = 5.3*u.um

    nu_det_cuton = wavel_to_nu(wavel_det_cuton)
    nu_det_cutoff = wavel_to_nu(wavel_det_cutoff)

    # See doc radiometric_reasoning.tex for derivation of the following. The model being considered
    # is a blackbody hemisphere surrounding the detector, and in the center of the hemisphere
    # is a pinhole at a temperature different from that of the hemisphere.

    # set up the BBs
    '''
    bb_nu_hemisphere = models.BlackBody(temperature=Teff_hemisphere)
    wav_hemisphere = np.linspace(0.1, 10, 100) * u.um

    nu_hemisphere = wavel_to_nu(wav_hemisphere)

    bb_nu_pinhole = models.BlackBody(temperature=Teff_pinhole)
    wav_hemisphere = np.linspace(0.1, 10, 100) * u.um
    nu_pinhole = wavel_to_nu(wav_hemisphere)

    flux_nu_hemisphere_from_lambda = bb_nu_hemisphere(wav_hemisphere)
    flux_nu_hemisphere_from_nu = bb_nu_hemisphere(nu_hemisphere)
    flux_nu_pinhole_from_lambda = bb_nu_pinhole(wav_hemisphere)
    flux_nu_pinhole_from_nu = bb_nu_pinhole(nu_hemisphere)
    '''

    # FYI plots
    '''
    # irradiance
    with quantity_support():
        plt.figure()
        plt.plot(wav_hemisphere, flux_nu_hemisphere_from_lambda)
        plt.title('FYI: Flux')
        #plt.semilogx(wav, flux)
        #plt.xlim(0.1, 16)
        plt.show()

    with quantity_support():
        plt.figure()
        plt.semilogx(nu_hemisphere, flux_nu_hemisphere_from_nu)
        plt.title('FYI: Flux')
        #plt.semilogx(wav, flux)
        #plt.xlim(0.1, 16)
        plt.show()
    '''

    # break down into photons

    '''
    # for just one pinhole size, temp

    T_pinhole_fixed =167*u.K
    T_hemisphere_array = np.linspace(55, 273, 100) * u.K
    adu_rate_pix_absorbed_vs_temp = u.Quantity(
        [
            photon_rate_per_pixel(
                T_hemisphere=T,
                T_pinhole=T_pinhole_fixed,
                width_pinhole=0 * u.um,
                length_pinhole=0 * u.um,
                dist_pinhole=10 * u.mm,
                nu_det_cuton=nu_det_cuton,
                nu_det_cutoff=nu_det_cutoff,
            )
            for T in T_hemisphere_array
        ]
    )

    plt.semilogy(T_hemisphere_array, adu_rate_pix_absorbed_vs_temp)
    plt.xlabel(f'Temperature of hemisphere({T_hemisphere_array.unit})')
    plt.ylabel(f'ADU per pixel ({adu_rate_pix_absorbed_vs_temp.unit})')
    plt.ylim(1e-6, 1e10)
    plt.axhline(y=1, color = 'k', linestyle = '--')
    plt.title(f'Detector counts')
    plt.show()
    '''

    # for overplotting multiple pinhole size, temp

    T_pinhole_fixed =167*u.K
    T_hemisphere_array = np.linspace(55, 273, 100) * u.K

    adu_rate_pix_absorbed_vs_temp_no_pinhole = u.Quantity(
        [
            photon_rate_per_pixel(
                T_hemisphere=T,
                T_pinhole=T_pinhole_fixed,
                width_pinhole=0 * u.um,
                length_pinhole=0 * u.um,
                dist_pinhole=10 * u.mm,
                nu_det_cuton=nu_det_cuton,
                nu_det_cutoff=nu_det_cutoff,
            )
            for T in T_hemisphere_array
        ]
    )

    # small pinhole is, I think, a D=30 um diameter circle, so for a square it 
    # would have lengths of D/2 * sqrt(pi)
    adu_rate_pix_absorbed_vs_temp_small_pinhole = u.Quantity(
        [
            photon_rate_per_pixel(
                T_hemisphere=T,
                T_pinhole=T_pinhole_fixed,
                width_pinhole=(30/2) * np.sqrt(np.pi) * u.um,
                length_pinhole=(30/2) * np.sqrt(np.pi) * u.um,
                dist_pinhole=10 * u.mm,
                nu_det_cuton=nu_det_cuton,
                nu_det_cutoff=nu_det_cutoff,
            )
            for T in T_hemisphere_array
        ]
    )

    adu_rate_pix_absorbed_vs_temp_large_pinhole = u.Quantity(
        [
            photon_rate_per_pixel(
                T_hemisphere=T,
                T_pinhole=T_pinhole_fixed,
                width_pinhole=0.5 * u.mm,
                length_pinhole=30 * u.mm,
                dist_pinhole=10 * u.mm,
                nu_det_cuton=nu_det_cuton,
                nu_det_cutoff=nu_det_cutoff,
            )
            for T in T_hemisphere_array
        ]
    )

    frame_time = 1.47528 # sec
    saturation_counts = 2**16
    max_adu_rate = saturation_counts / frame_time
    readout_noise_up_limit = 1200

    fig = plt.figure(figsize=(8, 5))
    plt.semilogy(T_hemisphere_array, adu_rate_pix_absorbed_vs_temp_no_pinhole, label='No pinhole')
    plt.semilogy(T_hemisphere_array, adu_rate_pix_absorbed_vs_temp_small_pinhole, label='30 um pinhole')
    plt.semilogy(T_hemisphere_array, adu_rate_pix_absorbed_vs_temp_large_pinhole, label='3x0.5 mm slit')
    plt.axhline(y=max_adu_rate, color = 'r', linestyle = '--', label=f'Max ADU rate')
    plt.axhline(y=readout_noise_up_limit, color = 'k', linestyle = '--', alpha=0.5, label=f'Approx. upper limit to read noise (with diff detec. output)')
    plt.axhline(y=3*readout_noise_up_limit, color = 'k', linestyle = '--', alpha=1, label=f'3x upper limit to read noise (with diff detec. output)')
    plt.xlabel(f'Temperature of hemisphere ({T_hemisphere_array.unit})')
    plt.ylabel(f'ADU per pixel ({adu_rate_pix_absorbed_vs_temp_no_pinhole.unit})')
    plt.ylim(1e-6, 1e10)
    plt.xlim(np.min(T_hemisphere_array.value), np.max(T_hemisphere_array.value))
    plt.legend(loc='lower right')
    plt.grid(True, which="both")
    plt.title(f'Detector counts\n Pinhole temp = {T_pinhole_fixed}')
    plt.show()


if __name__ == "__main__":
    main()
