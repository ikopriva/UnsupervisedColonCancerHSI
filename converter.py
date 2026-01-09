import os
import numpy as np
import scipy.io
from sklearn.decomposition import PCA
from tqdm import tqdm
import multiprocessing
import h5py
import hdf5storage
import scipy.io as sio
import shutil  # Import shutil for file copying

def load_mat_file(file_path):
    """
    Load a .mat file, handling both older versions and v7.3 (HDF5) format.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if 'img' in f:
                img = np.array(f['img'])
                # HDF5 files often store arrays in a different order
                if img.ndim == 3 and img.shape[0] < img.shape[-1]:
                    img = np.transpose(img, (1, 2, 0))
                return img
    # exception, read with scipy.io
    except ValueError:
        img = sio.loadmat(file_path)['img']
        return img

def process_reference_image(I_ref, n_stains=2, beta=0.15):
    """
    Process the reference image to compute stain vectors and percentiles.

    Parameters:
        I_ref (numpy.ndarray): Hyperspectral reference image.
        n_stains (int): Number of stains (principal components).
        beta (float): Threshold for OD values to remove pixels with no stains.

    Returns:
        V_ref (numpy.ndarray): Stain vectors from PCA.
        P1_C_ref (numpy.ndarray): 1st percentiles of the concentration matrix.
        P99_C_ref (numpy.ndarray): 99th percentiles of the concentration matrix.
        max_I_ref (numpy.ndarray): Maximum intensity per band in the reference image.
    """
    # Compute max intensity per band
    max_I_ref = np.max(I_ref, axis=(0,1))
    
    # Compute OD
    OD_ref = -np.log10((I_ref + 1) / (max_I_ref + 1e-8))
    
    # Threshold OD to remove pixels with no stains
    OD_ref_flat = OD_ref.reshape(-1, OD_ref.shape[2])
    OD_ref_norms = np.linalg.norm(OD_ref_flat, axis=1)
    OD_ref_nonzero = OD_ref_flat[OD_ref_norms > beta, :]
    
    # Perform PCA
    pca = PCA(n_components=n_stains)
    pca.fit(OD_ref_nonzero)
    V_ref = pca.components_.T  # Stain vectors
    
    # Project OD onto stain vectors to get concentration matrix
    C_ref = OD_ref_flat @ V_ref
    
    # Compute 1st and 99th percentiles of concentration values
    P1_C_ref = np.percentile(C_ref, 1, axis=0)
    P99_C_ref = np.percentile(C_ref, 99, axis=0)
    
    return V_ref, P1_C_ref, P99_C_ref, max_I_ref

def process_image_star(args):
    """Helper function to unpack arguments for multiprocessing."""
    return process_image(*args)

def process_image(image_path, V_ref, P1_C_ref, P99_C_ref, max_I_ref, output_folder, n_stains=2, beta=0.15):
    """
    Process a single image to apply Macenko normalization.

    Parameters:
        image_path (str): Path to the input image file.
        V_ref (numpy.ndarray): Stain vectors from reference image.
        P1_C_ref (numpy.ndarray): 1st percentiles from reference image.
        P99_C_ref (numpy.ndarray): 99th percentiles from reference image.
        max_I_ref (numpy.ndarray): Max intensity per band from reference image.
        output_folder (str): Path to the output folder.
        n_stains (int): Number of stains (principal components).
        beta (float): Threshold for OD values.
    """
    # Get the image filename
    image_filename = os.path.basename(image_path)
    
    # Load the input image
    try:
        I_input = load_mat_file(image_path)
    except ValueError as e:
        print(f"Skipping {image_filename}: {str(e)}")
        return
    
    # Compute max intensity per band
    max_I_input = np.max(I_input, axis=(0,1))
    
    # Compute OD
    OD_input = -np.log10((I_input + 1) / (max_I_input + 1e-8))
    
    # Threshold OD to remove pixels with no stains
    OD_input_flat = OD_input.reshape(-1, OD_input.shape[2])
    OD_input_norms = np.linalg.norm(OD_input_flat, axis=1)
    OD_input_nonzero = OD_input_flat[OD_input_norms > beta, :]
    
    if OD_input_nonzero.shape[0] == 0:
        print(f"Skipping {image_filename}: No pixels with OD above threshold.")
        return
    
    # Perform PCA
    pca = PCA(n_components=n_stains)
    pca.fit(OD_input_nonzero)
    V_input = pca.components_.T  # Stain vectors
    
    # Project OD onto stain vectors to get concentration matrix
    C_input = OD_input_flat @ V_input
    
    # Normalize concentration matrix
    P1_C_input = np.percentile(C_input, 1, axis=0)
    P99_C_input = np.percentile(C_input, 99, axis=0)
    
    epsilon = 1e-8
    C_norm = ( (C_input - P1_C_input) / (P99_C_input - P1_C_input + epsilon) ) * (P99_C_ref - P1_C_ref) + P1_C_ref
    
    # Reconstruct normalized OD
    OD_norm_flat = C_norm @ V_ref.T
    
    # Reshape back to image dimensions
    OD_norm = OD_norm_flat.reshape(OD_input.shape)
    
    # Store original shape and dimension order
    original_shape = I_input.shape
    
    # Reconstruct the normalized image
    I_norm = (10**(-OD_norm) * max_I_ref[np.newaxis, np.newaxis, :]) - 1
    I_norm[I_norm < 0] = 0
    
    # Reshape back to original dimensions
    I_norm = I_norm.reshape(original_shape)
    
    # reshape to match the original dimensions
    I_norm = np.transpose(I_norm, (2, 0, 1))
    
    # Save the normalized image using HDF5/MATLAB v7.3 format
    output_file = os.path.join(output_folder, image_filename)
    with h5py.File(output_file, 'w') as f:
        # Create dataset with compression
        f.create_dataset(
            'img', 
            data=I_norm,
            compression='gzip',
            compression_opts=9
        )

def macenko_normalization_hyperspectral(input_folder, output_folder, reference_image_path, n_stains=2):
    """
    Perform Macenko normalization on all .mat files in the input folder.

    Parameters:
        input_folder (str): Path to the folder containing input .mat files.
        output_folder (str): Path to the folder where normalized images will be saved.
        reference_image_path (str): Path to the reference image .mat file.
        n_stains (int): Number of stains (principal components) to use.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the reference image
    print("Loading reference image...")
    I_ref = load_mat_file(reference_image_path)
    
    # Process the reference image
    print("Processing reference image...")
    V_ref, P1_C_ref, P99_C_ref, max_I_ref = process_reference_image(I_ref, n_stains=n_stains, beta=0.15)
    
    # Get list of all .mat files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
    
    # Exclude the reference image from processing
    ref_filename = os.path.basename(reference_image_path)
    files = [f for f in files if f != ref_filename]
    
    # Prepare the arguments for multiprocessing
    args_list = []
    for f in files:
        image_path = os.path.join(input_folder, f)
        args = (image_path, V_ref, P1_C_ref, P99_C_ref, max_I_ref, output_folder, n_stains, 0.15)
        args_list.append(args)
    
    # Use multiprocessing to process images in parallel, limited to 2 workers
    pool = multiprocessing.Pool(processes=2)
    
    print("Processing images...")
    # Use tqdm to show progress bar
    for _ in tqdm(pool.imap_unordered(process_image_star, args_list), total=len(args_list)):
        pass
    
    pool.close()
    pool.join()
    print("Processing completed.")

if __name__ == '__main__':
    # Example usage:
    import argparse

    parser = argparse.ArgumentParser(description='Macenko normalization for hyperspectral images.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing .mat files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder where normalized images will be saved.')
    parser.add_argument('--reference_image', type=str, required=True, help='Path to the reference image .mat file.')
    parser.add_argument('--n_stains', type=int, default=2, help='Number of stains (principal components) to use.')
    args = parser.parse_args()

    # Automatically copy the target image to the output folder
    try:
        print(f"Copying {args.reference_image} to {args.output_folder}")
        shutil.copy(args.reference_image, os.path.join(args.output_folder, os.path.basename(args.reference_image)))  # Copy the original image
    except Exception as e:
        print(f"Error copying {args.reference_image}: {str(e)}")
    
    macenko_normalization_hyperspectral(args.input_folder, args.output_folder, args.reference_image, n_stains=args.n_stains)