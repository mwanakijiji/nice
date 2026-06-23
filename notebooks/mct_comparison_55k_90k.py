# Reads in MCT FITS files taken at two temps
# frames are taken at temperatures 55K and 90K

from astropy.io import fits
from astropy.visualization import ZScaleInterval
import numpy as np
import glob
import matplotlib.pyplot as plt
import ipdb
import os


def save_individ_fits_frame_inspection_plot(file_name, index, out_path=None):
    '''
    Load FITS primary HDU, plot full frame and center row/column profiles, save PNG.

    INPUTS:
    file_name : str
        Path to FITS file
    index : int
        Index of the file
    out_path : str, optional
        Path to save the plot

    OUTPUTS:
    None; Saves a PNG file to the path specified by out_path.
    '''

    if out_path is None:
        out_path = f"junk_{index:02d}.png"
    with fits.open(file_name) as hdul:
        data = hdul[0].data

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axs[0].imshow(data, origin="lower")
    axs[0].set_title("Full Image")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    middle_row = data[data.shape[0] // 2, :]
    axs[1].plot(middle_row)
    axs[1].set_title("Middle Row Profile")
    axs[1].set_xlabel("Column")
    axs[1].set_ylabel("Value")

    central_col = data[:, data.shape[0] // 2]
    axs[2].plot(central_col)
    axs[2].set_title("Central col Profile")
    axs[2].set_xlabel("Row")
    axs[2].set_ylabel("Value")

    fig.suptitle(f"{file_name}\nmean: {np.mean(data):.2f}, std: {np.std(data):.2f}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(out_path)
    plt.close(fig)
    print("Wrote ", out_path)

    return


def TBC_write_hot_pixel_map_pngs_from_fits_list(
    file_name_list,
    width_col=64,
    n_cols=32,
    out_prefix="frame_hot_",
    threshold_sigma=5,
    dpi=150,
):
    '''
    Median-subtract each vertical readout band, threshold frame, save binary hot-pixel map PNGs.

    INPUTS:
    file_name_list : sequence of str
        Paths to FITS files
    width_col : int
        Width of each vertical readout band in pixels
    n_cols : int
        Number of vertical readout bands
    out_prefix : str, optional
        Prefix for the output file names
    threshold_sigma : float, optional
        Threshold in sigma above the median for hot pixels
    dpi : int, optional
        DPI for the output images

    OUTPUTS:
    None; Saves PNG files to the path specified by out_prefix.
    '''
    for file_name_num, file_name in enumerate(file_name_list):
        print(file_name)
        with fits.open(file_name) as hdul:
            frame_data = np.asarray(hdul[0].data, dtype=float).copy()

        for col_num in range(n_cols):
            idx1, idx2 = width_col * col_num, width_col * (col_num + 1)
            col_data = frame_data[:, idx1:idx2]
            median_val = np.median(col_data)
            frame_data[:, idx1:idx2] = col_data - median_val

        threshold = np.median(frame_data) + threshold_sigma * np.std(frame_data)
        pix_map = np.zeros_like(frame_data, dtype=np.uint8)
        pix_map[frame_data > threshold] = 0
        pix_map[frame_data < threshold] = 1

        file_name_plot = f"{out_prefix}{file_name_num}.png"
        plt.figure(figsize=(5, 5))
        plt.imshow(pix_map, origin="lower", cmap="gray")
        plt.title(f"Hot Pixel Map: {file_name_num}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(file_name_plot, dpi=dpi)
        plt.close()
        n_hot = int(np.sum(frame_data > threshold))
        print(f"Wrote {file_name_plot} ({n_hot} hot pixels)")

    return


def per_pixel_linear_fit_slopes_intercepts(
    file_name_list,
    length_edge=2048,
    first_idx_to_fit=20,
    readout_width=64,
    fyi_plot_path=None,
):
    '''
    Stack FITS frames, median-correct by readout column band, fit linear drift vs frame index.

    INPUTS: 
    file_name_list : sequence of str
        Paths to FITS files
    length_edge : int
        Size of the frame in pixels (one edge)
    first_idx_to_fit : int
        Drop frame indices [0, first_idx_to_fit) before the per-pixel lstsq.
    readout_width : int
        Width in columns of each readout block for per-frame median subtraction (if applied)
    fyi_plot_path : str, optional
        Path to save the fyi plot
    OUTPUTS:
    slopes, intercepts : ndarray, shape (length_edge, length_edge)
        Per-pixel linear fit coefficients vs frame index (same x for every pixel).
    '''

    paths = list(file_name_list)
    paths = sorted(paths)
    if not paths:
        raise ValueError("file_name_list is empty")

    # check size of frame
    with fits.open(paths[0]) as hdul:
        data_test = hdul[0].data
    ny, nx = int(data_test.shape[0]), int(data_test.shape[1])
    if length_edge > ny or length_edge > nx:
        raise ValueError(f"length_edge {length_edge} exceeds frame shape {(ny, nx)}")

    # put all the frames into a cube
    data_cube = np.zeros((len(paths), ny, nx), dtype=np.asarray(data_test).dtype)
    for i, file_name in enumerate(paths):
        with fits.open(file_name) as hdul:
            data_cube[i, :, :] = hdul[0].data

    x = np.arange(len(paths), dtype=float)

    data_cube_corr = data_cube[:, :length_edge, :length_edge].copy()

    '''
    for col_start in range(0, length_edge, readout_width):
        col_end = min(col_start + readout_width, length_edge)
        readout = data_cube_corr[:, :, col_start:col_end]
        readout_median = np.median(readout, axis=(1, 2), keepdims=True)
        data_cube_corr[:, :, col_start:col_end] = readout - readout_median
    '''

    data_cube_corr_trunc = data_cube_corr[first_idx_to_fit:, :, :]
    x_trunc = x[first_idx_to_fit:]
    y = data_cube_corr_trunc.reshape(len(x_trunc), -1)
    a = np.vstack([x_trunc, np.ones_like(x_trunc)]).T
    coeffs, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    slopes = coeffs[0].reshape(length_edge, length_edge)
    intercepts = coeffs[1].reshape(length_edge, length_edge)

    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title('Slopes')
    s_lo, s_hi = ZScaleInterval().get_limits(np.asarray(slopes))
    im0 = ax[0].imshow(slopes, origin='lower', vmin=s_lo, vmax=s_hi)
    fig.colorbar(im0, ax=ax[0], label='counts/read')
    ax[1].set_title('Intercepts')
    im1 = ax[1].imshow(intercepts, origin='lower')
    fig.colorbar(im1, ax=ax[1], label='counts')
    plt.savefig(fyi_plot_path)
    plt.close()
    print('Wrote ', fyi_plot_path)
    
    return slopes, intercepts, data_cube


def plot_pixel_responses(data_cube, N_sample, fyi_plot_path):
    '''
    Plot some random pixel responses.
    '''

    # abcissa: frame index
    x = np.arange(len(data_cube[:,0,0]), dtype=float)

    n_slices = len(data_cube[:,0,0])
    size_window = data_cube.shape[1]

    plt.clf()
    for i in range(N_sample):
        N_ = np.random.randint(0, size_window)
        M_ = np.random.randint(0, size_window)
        plt.scatter(x, data_cube[:,N_,M_])
        print('stdev of these randomly-chosenpixel responses: ', np.std(data_cube[:,N_,M_]))
    plt.xlabel('read number')
    plt.ylabel('counts')
    plt.savefig(fyi_plot_path)  
    plt.close()
    print('Wrote ', fyi_plot_path)
    

    return


def main():

    #stem_55k = '/Users/eckhartspalding/Documents/git.repos/nice2/data/20260506085436/'
    
    #stem_55k = '/Users/eckhartspalding/Documents/job_science/postdoc_eth/nice/mct_detector/eckhart_20260508/20260508090532/' # 50 up the ramp reads each, detector 55 K, normal clocking, WITH preamp ktC removal
    #stem_90k = '/Users/eckhartspalding/Documents/job_science/postdoc_eth/nice/mct_detector/eckhart_20260510/20260510083709/' # 50 up the ramp reads each, detector 90 K, normal clocking, WITH preamp ktC removal

    stem_55k = '/Users/eckhartspalding/Documents/job_science/postdoc_eth/nice/mct_detector/eckhart_20260508/20260508091458/' # 50 up the ramp reads each, detector 55 K, normal clocking, NO preamp ktC removal
    stem_90k = '/Users/eckhartspalding/Documents/job_science/postdoc_eth/nice/mct_detector/eckhart_20260510/20260510084937/' # 50 up the ramp reads each, detector 90 K, normal clocking, NO preamp ktC removal

    outdir_fyi = '/Users/eckhartspalding/Documents/git.repos/nice2/data/test_data/inspection_plots/'

    # check that outdir exists
    if not os.path.exists(outdir_fyi):
        os.makedirs(outdir_fyi)

    file_name_list_55k = glob.glob(stem_55k + '*.fits')
    file_name_list_55k = sorted(file_name_list_55k)

    file_name_list_90k = glob.glob(stem_90k + '*.fits')
    file_name_list_90k = sorted(file_name_list_90k)

    # initial inspection of each frame
    '''
    for file_name_num, file_name in enumerate(file_name_list_55k):
        out_path = os.path.join(outdir_fyi, f"inspect_{file_name_num:02d}.png")
        save_individ_fits_frame_inspection_plot(file_name, file_name_num, out_path=out_path)
    '''

    # fit the slopes and intercepts
    slopes_55k, intercepts_55k, cube_55k = per_pixel_linear_fit_slopes_intercepts(file_name_list_55k, fyi_plot_path=os.path.join(outdir_fyi, 'slopes_intercepts_55k.png'))
    slopes_90k, intercepts_90k, cube_90k = per_pixel_linear_fit_slopes_intercepts(file_name_list_90k, fyi_plot_path=os.path.join(outdir_fyi, 'slopes_intercepts_90k.png'))

    # convert to signed integers
    cube_55k_signed = cube_55k.astype(np.int16)
    cube_90k_signed = cube_90k.astype(np.int16)

    _ = plot_pixel_responses(cube_55k_signed, N_sample=6, fyi_plot_path=os.path.join(outdir_fyi, 'pixel_responses_random_55k.png'))
    _ = plot_pixel_responses(cube_90k_signed, N_sample=6, fyi_plot_path=os.path.join(outdir_fyi, 'pixel_responses_random_90k.png'))

    # test: take longest integration at 90K, and subtract the shortest frame at 55 K
    file_name_longest_90k = file_name_list_90k[-1]
    file_name_shortest_90k = file_name_list_90k[0]

    file_name_longest_55k = file_name_list_55k[-1]
    file_name_shortest_55k = file_name_list_55k[0]

    with fits.open(file_name_longest_90k) as hdul:
        data_longest_90k = hdul[0].data
    with fits.open(file_name_shortest_55k) as hdul:
        data_shortest_55k = hdul[0].data
    with fits.open(file_name_longest_55k) as hdul:
        data_longest_55k = hdul[0].data
    with fits.open(file_name_shortest_90k) as hdul:
        data_shortest_90k = hdul[0].data
    #data_longest_90k_corr = data_longest_90k - np.median(data_longest_90k)
    #data_shortest_55k_corr = data_shortest_55k - np.median(data_shortest_55k)
    #data_diff = data_longest_90k - data_shortest_55k
    #plt.imshow(data_diff, origin='lower')
    #plt.colorbar()

    # save the cubes as FITS files
    hdu = fits.PrimaryHDU(cube_55k)
    hdu.writeto(os.path.join(outdir_fyi, 'cube_55k.fits'), overwrite=True)
    print('Wrote ', os.path.join(outdir_fyi, 'cube_55k.fits'))

    hdu = fits.PrimaryHDU(cube_90k)
    hdu.writeto(os.path.join(outdir_fyi, 'cube_90k.fits'), overwrite=True)
    print('Wrote ', os.path.join(outdir_fyi, 'cube_90k.fits'))

    ipdb.set_trace()

    # write out hot pixel maps
    #write_hot_pixel_map_pngs_from_fits_list(file_name_list)

    # compare to an old image
    # TBD


if __name__ == "__main__":
    main()


