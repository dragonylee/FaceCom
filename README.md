## FaceCom: Towards High-fidelity 3D Facial Shape Completion via Optimization and Inpainting Guidance

CVPR 2024

<br>

## Set-up

1. Download code :

```
git clone https://github.com/dragonylee/FaceCom.git
```

2. Create and activate a conda environment :

```
conda create -n FaceCom python=3.10
conda activate FaceCom
```

3. Install dependencies using `pip` or `conda` :

- [pytorch](https://pytorch.org/get-started/locally/)

  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

  ```
  conda install -c fvcore -c iopath -c conda-forge fvcore iopath
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  ```
  
- [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) & trimesh & [quad_mesh_simplify](https://github.com/jannessm/quadric-mesh-simplification)

  ```
  pip install torch_geometric, trimesh, quad_mesh_simplify
  ```

  You will find that after the installation, there is only `quad_mesh_simplify-1.1.5.dist-info` under the `site-packages` folder of your Python environment. Therefore, you also need to copy the `quad_mesh_simplify` folder from the [GitHub repository](https://github.com/jannessm/quadric-mesh-simplification) to the `site-packages` folder.

<br>

## Data

We trained our network using a structured hybrid 3D face dataset, which includes [Facescape](https://facescape.nju.edu.cn/) and [HeadSpace](https://www-users.york.ac.uk/~np7/research/Headspace/) datasets (under permissions), as well as our own dataset collected from hospitals. Due to certain reasons, the data we collected cannot be made public temporarily. Therefore, the method for training the model is not disclosed for the time being (If needed, we will provide it soon).

You can download our pre-trained model `checkpoint_decoder.pt` ([Google Drive](https://drive.google.com/file/d/1oPfWRPgCXjAffPJWfZyZyZOgd5EYPrHf/view?usp=drive_link)|[百度网盘](https://pan.baidu.com/s/1SsBW08yieLTCbK9ec6EnwA?pwd=z4vc)) and put it in `data` folder.

<br>

## Usages

After downloading the pre-trained model, you need to modify the project path of the first three lines of `config/config.cfg` 

```
template = PATH_TO_THE_PROJECT/data/template.ply
data_dir = PATH_TO_THE_PROJECT/data
checkpoint_dir = PATH_TO_THE_PROJECT/data
```

to match your own environment. Then, you can thoroughly test with the scripts we provide below.

### Random Sample

Randomly generate `--number` 3D face models.

```
python scripts/face_sample.py --config_file config/config.cfg --out_dir sample_out --number 10
```

### Facial Shape Completion

**NOTE** that our method has some considerations and flaws to be aware of.

1. The unit of the face model is in millimeters.
2. The range of the facial model should preferably be smaller than the `template.ply` we provide, otherwise add  `--dis_percent 0.8` to achieve better results.
3. We use trimesh's ICP for rigid registration, but are unsure of its accuracy and robustness. You may perform precise rigid registration with `template.ply` first and set `--rr False`.

Then, you can run our script to perform shape completion on `--in_file`, 

```
python scripts/face_completion.py --config_file config/config.cfg --in_file defect.ply --out_file comp.ply --rr True
```

where `--in_file` is a file that trimesh can read, with no requirements on topology. We provide `defect.ply` for convenience.

### Mesh Fit / Non-rigid Registration

When the input is a complete facial model without any defects, the script in the "Facial Shape Completion" section will actually output a fitting result to the input. Since the topology of our method's output is consistent, it can also be used for non-rigid registration.
