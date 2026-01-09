# Hyperspectral Stain Normalization

Tools for normalizing and visualizing hyperspectral images using the Macenko method.

## Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Stain Normalization (`converter.py`)

Normalizes hyperspectral images against a reference image using the Macenko method.

```bash
python converter.py --input_folder <input_dir> --output_folder <output_dir> --reference_image <reference.mat> [--n_stains 2]
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--input_folder` | Yes | Directory containing .mat files to normalize |
| `--output_folder` | Yes | Directory where normalized images will be saved |
| `--reference_image` | Yes | Path to the reference .mat file |
| `--n_stains` | No | Number of stain vectors to extract (default: 2) |

**Example:**

```bash
python converter.py \
    --input_folder input_data/ \
    --output_folder output_data/normalized/ \
    --reference_image input_data/reference_sample.mat \
    --n_stains 2
```

The reference image is automatically copied to the output folder.

### Colorization (`batch_colorize_hsi.py`)

Converts hyperspectral images to RGB TIFF files using CIE color matching functions.

```bash
python batch_colorize_hsi.py <input_path> <output_path>
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `input_path` | Directory containing .mat files |
| `output_path` | Directory where TIFF images will be saved |

**Example:**

```bash
python batch_colorize_hsi.py output_data/normalized/ output_data/colorized/
```

## Typical Workflow

1. **Normalize** your hyperspectral images against a reference:
   ```bash
   python converter.py \
       --input_folder input_data/ \
       --output_folder output_data/normalized/ \
       --reference_image input_data/reference.mat
   ```

2. **Colorize** the normalized images for visualization:
   ```bash
   python batch_colorize_hsi.py output_data/normalized/ output_data/colorized/
   ```

## Data Format

- **Input:** MATLAB .mat files with hyperspectral data stored under the `'img'` key
- **Dimensions:** 3D arrays (height x width x spectral_bands)
- **Supported formats:** Both legacy MATLAB (.mat v5) and HDF5-based (.mat v7.3)

## Output

- **Normalized images:** HDF5-based .mat files with gzip compression
- **Colorized images:** 8-bit RGB TIFF files
