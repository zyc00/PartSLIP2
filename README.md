# PartSLIP++

official implementation of "PartSLIP++: Enhancing Low-Shot 3D Part Segmentation via Multi-View Instance Segmentation and Maximum Likelihood Estimation"

## installation
### Create a conda environment and install dependencies.
```
conda env create -f environment.yml
conda activate partslip++
```

### Install PyTorch3D
We utilize [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for rendering point clouds. Please install it by the following commands or its [official guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md):
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Install GLIP
We incorporate [GLIP](https://github.com/microsoft/GLIP) and made some small changes. Please clone our [*modified version*](https://github.com/Colin97/GLIP) and install it by the following commands or its official guide:
```
git submodule update --init
cd GLIP
python setup.py build develop --user
```

### Install cut-pursuit
We utilize [cut-pursuit](https://github.com/loicland/superpoint_graph) for computing superpoints. Please install it by the following commands or its official guide:
```
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.9.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.9 -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```

### Install SegmentAnything
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Quick Start
### Download pretrained checkpoints
You can find the pre-trained checkpoints from [here](https://huggingface.co/datasets/minghua/PartSLIP/tree/main/models). For few-shot inference, please use our few-shot checkpoints for each object category.

### Generate Superpoint
To save the superpoints and other medium results, run
```
python gen_sp.py
```

### Run PartSLIP with mask input
```
python run_partslip.py
```

### Run PartSLIP++
```
python run_partslip++.py
```

### Semantic Segmentation Evaluation
```
cd eval
python semantic_eval.py
```

### Instance Segmentation Evaluation
```
cd eval
python instance_eval.py
```