# Deep Geometric Prior for Surface Reconstruction
The reference implementaiton for the CVPR 2019 paper [Deep Geometric Prior for Surface Reconstruction](https://arxiv.org/pdf/1811.10943.pdf).

#### NOTE: Scripts to reproduce exact results in the paper will be made available soon. To perform comparisons on the surface reconstruction benchmark, use the data [available for download here](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view?usp=sharing). 

![](https://github.com/fwilliams/deep-geometric-prior/blob/master/data/teaser.png)

## Code Overview
There are several programs in this repository explained in detail below. The documentation for each program can be seen by running it with the `-h` flag. The code is also extensively commented and should be easy to follow. Please create GitHub issues or reach out to me by email if you run into any problems.

- #### `reconstruct_surface.py`:
  Compute a set of patches which represent a surface. 

  This program produces a file (defaulting to `out.pt`) as output which can be used to upsample a point cloud with `export_point_cloud.py`. You can optionally plot the reconstruction with `plot_reconstruction.py`.
   
- #### `reconstruct_single_patch.py` 
  Compute a single surface patch fitted to a point cloud.

  As with `reconstruct_surface.py`, this program produces a file (defaulting to `out.pt`) as output which can be used to upsample a point cloud with `export_point_cloud.py`. You can optionally plot the reconstruction with `plot_reconstruction.py`.

   
- #### `plot_reconstruction.py` 
  Plots a reconstructed point cloud produced by `reconstruct_surface.py` or `reconstruct_single_patch.py`
   
- #### `export_point_cloud.py` 
  Exports a dense point cloud from a reconstruction file produced by `reconstruct_surface.py` or `reconstruct_single_patch,py`. 
  This can be fed into a standard algorithm such as [Screened Poisson Surface Reconstruction](https://github.com/mkazhdan/PoissonRecon) to extract a triangle mesh.


## Setting up and Running the Code
  
### With [`conda`](https://conda.io/en/latest/) (Recommended)
All dependencies can be automatically installed with [`conda`](https://conda.io/en/latest/) using the provided `environment.yml`
Simply run the following from the root of the repository:
  
```
conda env create -f environment.yml
```
  
This will create a conda environment named `deep-surface-prior` with all the correct dependencies installed. You can activate the environment by running:
```
conda activate deep-surface-prior
```

Note: this code will not work on Windows due to lack of support by the [Point Cloud Utils](https://github.com/fwilliams/point_cloud_utils) dependency. 

### Installing Dependencies Manually (Not Recommended)
If you are not using Conda, you can manually install the following dependencies:
- Python 3.6 (or later)
- PyTorch 1.0
- NumPy 1.15 (or later)
- SciPy 1.1.0 (or later)
- [FML](https://github.com/fwilliams/fml) 0.1 (or later)
- [Point Cloud Utils](https://github.com/fwilliams/point_cloud_utils) 0.52 (or later) 
- [Mayavi](https://docs.enthought.com/mayavi/mayavi/) 4.6.2 (or later)
  
## [Surface Reconstruction Benchmark Data](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view?usp=sharing)
The scans, ground truth data and reconstructions from the paper are [available for download here](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view?usp=sharing).

The linked zip archive contains 3 directories:
* `scans` contains a simulated scan of the models. The scans are generated with the [surface reconstruction benchmark](https://github.com/fwilliams/surface-reconstruction-benchmark).
* `ground_truth` contains a dense point cloud for each model sampled from the ground truth surface.
* `our_reconstructions` contains a reconstructed point cloud for each model generated with our method.
