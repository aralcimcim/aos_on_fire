# AOS on Fire

Practical work at Johannes Kepler University Linz providing a desktop interface for image preprocessing, camera pose estimation, and integration with an external SAI application.

The simplicity of the interface is aimed for Austrian firefighters with quick deployment in the field. The result is a fully working prototype.

The interface brings three pose-estimation pipelines into a single workflow: Depth Anything 3, VGGT, and COLMAP.

## Features

- Select an image folder through a graphical interface.
- Automatically center-crop images to a square format.
- Choose between three pose-estimation methods.
- Export camera information to `poses.json`.
- Export a colored point cloud when using VGGT.
- Monitor progress and processing logs.
- Launch the external `SAI.exe` application after processing.

## Pose-estimation methods

| Method | Pipeline |
|---|---|
| **Depth Anything 3** | Runs `depth-anything/da3-large` and exports predicted extrinsic matrices |
| **VGGT** | Runs `facebook/VGGT-1B`, exports camera matrices, and generates a colored point cloud |
| **COLMAP** | Performs feature extraction, exhaustive matching, and sparse reconstruction before converting the registered camera positions |

## Workflow

1. Select a folder containing JPG, JPEG, or PNG images.
2. Choose DA3, VGGT, or COLMAP.
3. Click **GENERATE POSES**.
4. Images are copied and center-cropped into the SAI working directory.
5. The selected pipeline generates camera information.
6. Results are saved, and `SAI.exe` is launched.

The workflow clears the SAI working directory’s `images/` folder before copying the selected images. Keep original images in a separate source folder.

## Repository structure

```text
aos_on_fire/
├── LFR_executable.py   # GUI and processing pipelines
├── aos_env.yml         # Exported Conda environment
├── aos_fire.spec       # PyInstaller build configuration
└── build/aos_fire/     # Generated Windows build artifacts
```

## Requirements

The current implementation targets **Windows** and uses local paths to external applications.

Required components include:

- Python 3.11
- Tkinter and ttkthemes
- PyTorch
- Depth Anything 3
- VGGT
- NumPy, Pillow, and trimesh
- COLMAP, when using the COLMAP pipeline
- The external `SAI.exe` application

The neural pipelines download pretrained model weights. An NVIDIA GPU with a compatible PyTorch/CUDA installation is recommended. The VGGT implementation currently uses CUDA autocasting, so CPU execution may require changes.

`SAI.exe`, the COLMAP installation, and input image datasets are not included.

## Setup

Clone the repository:

```bash
git clone https://github.com/aralcimcim/aos_on_fire.git
cd aos_on_fire
```

The supplied `aos_env.yml` records the original Windows environment. It includes platform-specific packages, exact build identifiers, and CUDA-specific dependencies, so it may require adaptation before installation:

Install Depth Anything 3 and VGGT using their respective project instructions if they cannot be resolved from the environment export:

- https://github.com/ByteDance-Seed/Depth-Anything-3
- https://github.com/facebookresearch/vggt

Download and configure COLMAP separately if required:

- https://github.com/colmap/colmap
