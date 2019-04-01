# Deep Geometric Prior for Surface Reconstruction
![](https://github.com/fwilliams/deep-geometric-prior/blob/master/data/teaser.png)
This repository contains reference implementaiton for the paper [Deep Geometric Prior for Surface Reconstruction](https://arxiv.org/pdf/1811.10943.pdf).
There are several programs in this repository explained in detail below. The documentation for each program can be seen by running it with the `-h` flag. The code is also extensively commented and should be easy to follow. Please create GitHub issues or reach out to me by email if you run into any problems.

- #### `reconstruct_surface.py`:
  Compute a set of patches which represent a surface. 

  This program produces a file (defaulting to `out.pt`) as output which can be used to upsample a point cloud with `export_point_cloud.py`. You can optionally plot the reconstruction with `plot_reconstruction.py`.
   
- #### `reconstruct_single_patch.py` 
  Compute a single surface patch fitted to a point cloud.

  As with `reconstruct_surface.pyt`, this program produces a file (defaulting to `out.pt`) as output which can be used to upsample a point cloud with `export_point_cloud.py`. You can optionally plot the reconstruction with `plot_reconstruction.py`.

   
- #### `plot_reconstruction.py` 
  Plots a reconstructed point cloud produced by `reconstruct_surface.py` or `reconstruct_single_patch.py`
   
- #### `export_point_cloud.py` 
  Exports a dense point cloud from a reconstruction file produced by `reconstruct_surface.py` or `reconstruct_single_patch,py`. 
  This can be fed into a standard algorithm such as [Screened Poisson Surface Reconstruction](https://github.com/mkazhdan/PoissonRecon) to extract a triangle mesh.


## Dependencies
  
### With [`conda`](https://conda.io/en/latest/) (Recommended)
All dependencies can be automatically installed with [`conda`](https://conda.io/en/latest/) using the provided `environment.yml`
Simply run the following from the root of the repository:
  
```
conda env create -f environment.yml
```
  
### Installing Dependencies Manually (Not Recommended)
If you are not using Conda, you can manually install the following dependencies:
- Python 3.6 or later
- PyTorch 1.0
- NumPy
- SciPy
- [fml](https://github.com/fwilliams/fml) >=0.1
- [point_cloud_utils](https://github.com/fwilliams/point_cloud_utils) >= 0.52
- Plotly
  
