# Image segmentation
Instance segmentation for the fasionpedia dataset.

## Directory structure
Structure of the project
```
dist/
    ImageSegmentation-0.0.1-py3-none-any.whl
    ImageSegmentation-0.0.1.tar.gz
notebooks/
    main.ipynb
src/
    model/
        __init__.py
        data_loader.py
        net.py
    utils/
        __init__.py
        train_utils.py
        transform_utils.py
        utils.py
    make_fashionpedia_dataset.py
    train.py
tests/
    __init__.py
.gitignore
pyproject.toml
README.md
requirements.txt
setup.cfg
```

## Usage
First clone the project as follows,
```
git clone <url> <newprojname>
cd <newprojname>
```
Then build the project by using the following command, (assuming build is already installed in your virtual environment, if not then activate your virtual environment and use `pip install build`)
```
python -m build
```
Next, install the build wheel file as follows,
```
pip install <path to wheel file>
```
Then create the dataset by running the following command (this needs to be done only once, and can be done at anytime after cloning this repo),
```
python make_fashionpedia_dataset.py -d <path to fashion pedia root dir>
```
Then to start training on a single node with multiple gpu's we can do the following,
```
torchrun --standalone --nnodes=1 --nproc_per_node=<num gpu's> <path to train.py>
```

## Requirements
I used Anaconda with python3, but used pip to install the libraries so that they worked with my multi GPU compute environment in Azure

```
conda create -n <yourenvname> python=<3.x>
conda activate <yourenvname>
pip install -r requirements.txt
```