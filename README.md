# Deep Geometric Prior for Surface Reconstruction
The reference implementaiton for the CVPR 2019 paper [Deep Geometric Prior for Surface Reconstruction](https://arxiv.org/pdf/1811.10943.pdf).

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
conda activate deep-geometric-prior
```

Note: this code will not work on Windows due to lack of support by the [Point Cloud Utils](https://github.com/fwilliams/point_cloud_utils) dependency. 

### Installing Dependencies Manually (Not Recommended)
If you are not using Conda, you can manually install the following dependencies:
- Python 3.6 (or later)
- PyTorch 1.0 (or later)
- NumPy 1.15 (or later)
- SciPy 1.1.0 (or later)
- [FML](https://github.com/fwilliams/fml) 0.1 (or later)
- [Point Cloud Utils](https://github.com/fwilliams/point_cloud_utils) 0.12.0 (or later) 
- [Mayavi](https://docs.enthought.com/mayavi/mayavi/) 4.6.2 (or later)


## [Surface Reconstruction Benchmark Data](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view?usp=sharing)
The scans, ground truth data and reconstructions from the paper are [available for download here](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view?usp=sharing).

The linked zip archive contains 3 directories:
* `scans` contains a simulated scan of the models. The scans are generated with the [surface reconstruction benchmark](https://github.com/fwilliams/surface-reconstruction-benchmark).
* `ground_truth` contains a dense point cloud for each model sampled from the ground truth surface.
* `our_reconstructions` contains a reconstructed point cloud for each model generated with our method.

## Running the Deep Geometric Prior on the Surface Reconstruction Benchmark
* Make sure to install the project dependencies with conda or manually as described [above](https://github.com/fwilliams/deep-geometric-prior#setting-up-and-running-the-code).
* Download the [Surface Reconstruction Benchmark Data](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view?usp=sharing) (See [above](https://github.com/fwilliams/deep-geometric-prior#surface-reconstruction-benchmark-data) section for details).
* Extract the zip file which should produce a directory named `deep_geometric_prior_data`. 
* Since Deep Geometric Prior fits many neural networks over a model, it requires a lot of memory (see paper for details), thus it is best to use multiple GPUs when reconstructing a model. Suppose four GPUs are available named `cuda:0`, `cuda:1`, `cuda:2`, and `cuda:3`, then the following five commands will reconstruct the five benchmark models using those GPUs. You can change the list of devices for different configurations:
```bash
python reconstruct_surface.py deep_geometric_prior_data/scans/gargoyle.ply 0.01 1.0 20 -d cuda:0 cuda:1 cuda:2 cuda:3 -nl 25 -ng 25 -o gargoyle

python reconstruct_surface.py deep_geometric_prior_data/scans/dc.ply 0.01 1.0 20 -d cuda:0 cuda:1 cuda:2 cuda:3 -nl 25 -ng 25 -o dc

python reconstruct_surface.py deep_geometric_prior_data/scans/lord_quas.ply 0.01 1.0 10 -d cuda:0 cuda:1 cuda:2 cuda:3 -nl 25 -ng 25 -o lord_quas

python reconstruct_surface.py deep_geometric_prior_data/scans/anchor.ply 0.01 1.0 10 -d cuda:0 cuda:1 cuda:2 cuda:3 -nl 25 -ng 25 -o anchor

python reconstruct_surface.py deep_geometric_prior_data/scans/daratech.ply 0.01 1.0 10 -d cuda:0 cuda:1 cuda:2 cuda:3 -nl 25 -ng 25 -o daratech   
```

*NOTE:* You may need to change the pathss `deep_geometric_prior_data/scans/*.ply` to point to where you extracted the zip file, and you may need to change the device arguments `-d cuda:0 ...` to adapt to your system.

Each of the above commands produces a `ply` file and `pt` file (e.g. `anchor.ply`, `anchor.pt`). The PLY file contains a dense upsampled point cloud and the PT file contains metadata about the reconstruction. You can use the PT file to perform further operations using example `export_point_cloud.py`. 
