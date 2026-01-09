#!/usr/bin/env python3
"""
Batch colorization of hyperspectral images.

This script replicates the functionality of the MATLAB function:
    batch_colorize_hsi(input_path, output_path)
    
For each .mat file in the provided input directory, it:
  1. Loads hyperspectral data stored under the key 'img'
  2. Constructs a hyperspectral cube with wavelengths spanning 450–800 nm (351 points)
  3. Colorizes the cube into an RGB image using a process that:
       • Approximates CIE 1931 color matching functions (CMFs) via Gaussians,
       • Integrates the hyperspectral data over wavelength to yield XYZ tristimulus values,
       • Converts XYZ to linear sRGB via a standard matrix transformation,
       • Applies contrast stretching to map the output into [0, 1].
  4. Saves the image as a TIFF file in the output directory.
  
Note: The progress message prints the output filename with a ".png" extension,
      exactly as in the original MATLAB function.
      
Documentation references:
  - MATLAB hypercube.colorize: https://de.mathworks.com/help/images/ref/hypercube.colorize.html
  - MATLAB hypercube: https://de.mathworks.com/help/images/ref/hypercube.html
"""

import os
import numpy as np
import scipy.io
import hdf5storage
import imageio
import glob
from tqdm import tqdm

def load_mat_file(filepath):
    """
    Load a MATLAB file, handling both legacy formats and v7.3 format (HDF5).
    
    Parameters:
        filepath : str
            Path to the MATLAB .mat file
            
    Returns:
        dict: Dictionary containing the loaded data
    """
    try:
        # First try loading with scipy.io.loadmat (works for MATLAB 5.0 format)
        return scipy.io.loadmat(filepath)
    except (NotImplementedError, ValueError):
        # NotImplementedError: MATLAB 7.3 format (HDF5-based .mat files)
        # ValueError: Pure HDF5 files created by h5py (e.g., from converter.py)
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            # Load 'img' dataset and convert from HDF5 dataset to numpy array
            # Note: HDF5 stores MATLAB arrays in transposed form
            data['img'] = np.array(f['img']).transpose()
        return data

def batch_colorize_hsi(input_path, output_path):
    """
    Process all .mat files in the input directory, colorize the hyperspectral image data, and save them as TIFF images.
    
    Steps:
      1. List all .mat files in the input directory.
      2. Define the wavelength range (351 points from 450 to 800 nm).
      3. For each file:
         a. Load the hyperspectral image stored under key 'img'.
         b. Create a Hypercube object combining the data and wavelength axis.
         c. Convert (colorize) the hyperspectral cube to an RGB image using the 'rgb' method with contrast stretching.
         d. Save the colorized image as a TIFF file (while the progress message names it with a .png extension).
    
    Parameters:
      input_path  : str
          Directory containing the .mat files.
      output_path : str
          Directory where the output (colorized) images will be saved.
    """
    # List all .mat files in the given input directory.
    mat_files = glob.glob(os.path.join(input_path, '*.mat'))
    
    # Define wavelength range: 351 equally spaced points between 450 and 800 nm.
    wav = np.linspace(450, 800, 351, dtype=np.float32)
    print("Wavelength array type:", type(wav[0]))
    
    # Process each .mat file.
    for filepath in tqdm(mat_files, desc="Processing files", total=len(mat_files)):
        print("Processing file:", filepath)
        try:
            # Load the .mat file using the new helper function
            data = load_mat_file(filepath)
            if 'img' in data:
                hsi = data['img']
            else:
                raise KeyError(f"'img' key not found in {filepath}")
            print("Hyperspectral image data type:", type(hsi[0,0,0]))
            
            # Create a hyperspectral cube with the image data and its wavelengths.
            hcube = Hypercube(hsi, wav)
            
            # Colorize the hypercube to obtain an RGB image.
            psdRGB = colorize(hcube, method='rgb', contrast_stretching=True)
            
            # Construct output filename: same base name but with a .tiff extension.
            filename = os.path.basename(filepath)
            name, _ = os.path.splitext(filename)
            output_file = os.path.join(output_path, name + '.tiff')
            
            # Scale the floating-point RGB image (assumed to be in [0,1]) to 8-bit and save it.
            psdRGB_uint8 = (np.clip(psdRGB, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(output_file, psdRGB_uint8)
            
            # Note: The MATLAB code prints .png in the message even though a .tiff file is saved.
            print(f"Processed {filename} and saved as {name + '.png'}")
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue

class Hypercube:
    """
    A class representing a hyperspectral cube.
    
    It contains:
      - data: a 3D numpy array representing the hyperspectral image (height x width x spectral bands)
      - wavelengths: a 1D numpy array listing the wavelength associated with each spectral band.
      
    This class mimics MATLAB's hypercube object.
    """
    def __init__(self, data, wavelengths):
        """
        Initialize the Hypercube with image data and wavelength information.
        
        Parameters:
          data        : numpy.ndarray
              The hyperspectral image data.
          wavelengths : numpy.ndarray
              The corresponding wavelengths for the spectral bands.
        """
        self.data = data
        self.wavelengths = wavelengths

def colorize(hcube, method='rgb', contrast_stretching=True):
    """
    Converts a hyperspectral cube into an RGB image.
    
    The 'rgb' method implemented here approximates MATLAB's hypercube.colorize behavior.
    
    Steps:
      1. Approximate the CIE 1931 color matching functions (CMFs) using Gaussian functions:
           - x_bar: approximated with a peak around 600 nm (broad response).
           - y_bar: approximated with a peak around 550 nm (similar to luminance sensitivity).
           - z_bar: approximated with a peak around 450 nm.
         (These approximations are simplistic and serve as a demo.)
         
      2. For each pixel, compute the tristimulus values by integrating the hyperspectral
         data multiplied by the corresponding CMF over the wavelengths using the trapezoidal rule.
         
      3. Convert the computed XYZ values to linear sRGB using the standard transformation matrix.
      
      4. If enabled, perform contrast stretching on each channel to scale the image intensities
         into the [0,1] range.
    
    Parameters:
      hcube             : Hypercube
          The hyperspectral cube (data and wavelengths).
      method            : str (default 'rgb')
          The colorization method to use. Currently, only 'rgb' is supported.
      contrast_stretching : bool (default True)
          Whether to apply contrast stretching to the resulting RGB image.
    
    Returns:
      numpy.ndarray:
          The colorized RGB image as an array of shape (height, width, 3) with values in [0,1].
    """
    if method != 'rgb':
        raise ValueError("Only 'rgb' method is supported in this implementation.")
    
    # Extract hyperspectral image data and associated wavelengths.
    hsi = hcube.data    # Expected shape: (height, width, bands)
    wav = hcube.wavelengths
    
    # --- Step 1: Approximate the CIE 1931 CMFs using Gaussian functions.
    #
    # Note: The true CIE CMFs (x̄(λ), ȳ(λ), z̄(λ)) are defined over 380–780 nm with tabulated data.
    # Here we use simple Gaussian functions to mimic the sensitivity of the L-, M-, and S-cones:
    #   • x_bar: peak ~600 nm, sigma ~40 nm.
    #   • y_bar: peak ~550 nm, sigma ~30 nm.
    #   • z_bar: peak ~450 nm, sigma ~20 nm.
    x_bar = np.exp(-0.5 * ((wav - 600) / 40)**2)
    y_bar = np.exp(-0.5 * ((wav - 550) / 30)**2)
    z_bar = np.exp(-0.5 * ((wav - 450) / 20)**2)
    
    # --- Step 2: Integrate the hyperspectral data with the CMFs to compute XYZ values.
    #
    # For a given pixel spectrum f(λ), the tristimulus values are:
    #   X = ∫ f(λ) * x_bar(λ) dλ,
    #   Y = ∫ f(λ) * y_bar(λ) dλ,
    #   Z = ∫ f(λ) * z_bar(λ) dλ.
    #
    # We assume uniform wavelength sampling. Here the integration is approximated by a dot product
    # (using the trapezoidal rule, where delta is the spacing between wavelengths).
    delta = wav[1] - wav[0]  # Wavelength increment (assumed constant)
    X = np.tensordot(hsi, x_bar, axes=([2], [0])) * delta
    Y = np.tensordot(hsi, y_bar, axes=([2], [0])) * delta
    Z = np.tensordot(hsi, z_bar, axes=([2], [0])) * delta
    
    # Stack the integrated results to form an XYZ image.
    XYZ = np.stack((X, Y, Z), axis=-1)
    
    # --- Step 3: Convert from XYZ to linear sRGB.
    #
    # Using the standard conversion matrix for the D65 white point:
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
    #
    # The sRGB values for each pixel are obtained by the matrix multiplication:
    #   rgb = M * [X, Y, Z] (done for every pixel).
    rgb_linear = np.dot(XYZ, M.T)
    
    # Clip any negative values to zero (negative values are non-physical).
    rgb_linear = np.clip(rgb_linear, 0, None)
    
    # --- Step 4: Apply contrast stretching to map the data into the [0,1] range.
    #
    # Contrast stretching is performed on each channel independently.
    if contrast_stretching:
        rgb_stretched = contrast_stretch(rgb_linear)
    else:
        rgb_stretched = rgb_linear
    
    return rgb_stretched

def contrast_stretch(image):
    """
    Apply contrast stretching to an image on a per-channel basis.
    
    For each channel, the function finds the minimum and maximum pixel values and
    scales the channel's data linearly so that the minimum maps to 0 and the maximum to 1.
    
    Parameters:
      image : numpy.ndarray
          Input image array of shape (height, width, channels).
    
    Returns:
      numpy.ndarray:
          The contrast-stretched image with the same shape and values constrained to [0,1].
    """
    stretched = np.empty_like(image)
    # Process each channel independently.
    for channel in range(image.shape[2]):
        channel_data = image[..., channel]
        min_val = channel_data.min()
        max_val = channel_data.max()
        # Avoid division by zero if the min and max are equal.
        if max_val > min_val:
            stretched[..., channel] = (channel_data - min_val) / (max_val - min_val)
        else:
            stretched[..., channel] = channel_data
    return stretched

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch colorize hyperspectral images from .mat files."
    )
    parser.add_argument(
        "input_path", type=str,
        help="Directory containing .mat files with hyperspectral data."
    )
    parser.add_argument(
        "output_path", type=str,
        help="Directory where the colorized images will be saved."
    )
    args = parser.parse_args()
    
    # Create the output directory if it does not exist.
    os.makedirs(args.output_path, exist_ok=True)
    
    batch_colorize_hsi(args.input_path, args.output_path) 