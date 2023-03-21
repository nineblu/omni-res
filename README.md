# Omni-RES

[![Python](https://img.shields.io/badge/python-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-%237732a8)

Omni-RES is a simple and lightweight codebase for the research of Omni-supervised referring expression segmentation, which currently supporting MCN as base model. Later LAVT model also will be updated. 


## Installation
- Clone this repo
- Create a conda virtual environment and activate it
```bash
conda create -n omni_res python=3.7 -y
conda activate omni_res
```
- Install Pytorch following the [official installation instructions](https://pytorch.org/get-started/locally/)
- Install mmcv following the [installation guide](https://github.com/open-mmlab/mmcv#installation)
- Install [Spacy](https://spacy.io/) and initialize the [GloVe](https://github-releases.githubusercontent.com/84940268/9f4d5680-4fed-11e9-9dd2-988cce16be55?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210815%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210815T072922Z&X-Amz-Expires=300&X-Amz-Signature=1bd1bd4fc52057d8ac9eec7720e3dd333e63c234abead471c2df720fb8f04597&X-Amz-SignedHeaders=host&actor_id=48727989&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_vectors_web_lg-2.1.0.tar.gz&response-content-type=application%2Foctet-stream) and install other requirements as follows:
```
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```

## Data preparation

-  Follow the instructions of  [DATA_PRE_README.md](./DATA_PRE_README.md) to generate training data and testing data.
-  Download the pretrained weights of backbone (vgg, darknet, cspdarknet, DResNet, etc.).  Expect for DResNet, all pretrained backbones are trained on COCO 2014 *train+val*  set while removing the images appeared in the *val+test* sets of RefCOCO, RefCOCO+ and RefCOCOg (nearly 6500 images).  Please follow the instructions of  [DATA_PRE_README.md](./DATA_PRE_README.md) to download them.
-  Also, we provide the necessary json files for training, the downloading url is https://anonymous.4open.science/r/omni_res_data

## Training and Evaluation 

1. **Config preparation**. Prepare your own configs in [configs](./configs), you don't need to rewrite all the contents in config every time.You can import the config as a python file to use the default configs. For example, to run 10% RefCOCO with omni-box，you can use [mcn_refcoco_omni.py](./configs/mcn_refcoco_omni.py) as follows:

```python
# your own config.py
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.mcn import model

# Refine data path depend your own need
dataset.ann_path["refcoco"] = "./data/anns/refcoco.json"
dataset.image_path["refcoco"] = "./data/images/train2014"
dataset.mask_path["refcoco"] = "./data/masks/refcoco"
dataset.sup_ann_path["refcoco"] = "./data/anns/refcoco_0.1.json"
dataset.omni_ann_path["refcoco"] = "./data/anns/refcoco_0.9.json"
...
```

2. **Train the model**. To start the training, you can input command as follows:
```shell
# Training 10% RefCOCO with omni-box (MCN as base model)
python -m torch.distributed.launch --nproc_per_node 4 train_omni.py --config configs/mcn_refcoco_omni.py

# Training 100% RefCOCO in fully supervised learning (MCN as base model)
python -m torch.distributed.launch --nproc_per_node 4 train_sup.py --config configs/mcn_refcoco_sup.py

# Training RefCOCO, RefCOCO+, RefCOCOg, Visual Genome with omni-box (MCN as base model)
python -m torch.distributed.launch --nproc_per_node 4 train_vg.py --config configs/mcn_refcoco_vg.py

```
The `training logs`, `config.yaml` and `model checkpoints` will be automatically saved under `cfg.train.output_dir`.

3. **Resume training**. We support two resume training mode. You can resume from a specific checkpoint or resume from the latest checkpoint:

- Auto resume from `last.pth`:
```python
# config.py
from .common.train import train
train.auto_resume.enabled = True
```
Setting `train.auto_resume.enabled=True`, which will automatically resume from `last_checkpoint.pth` saved in `cfg.train.output_dir`.

- Resume from a specific checkpoint

```python
# config.py
from .common.train import train

# disable auto resume first
train.auto_resume.enabled = False

# modify the resume path
train.resume_path = "path/to/specific/checkpoint.pth"
```
Setting `train.resume_path` to the specific `checkpoint.pth` you want to resume from.

4. **Test the model.** 

```shell
python -m torch.distributed.launch --nproc_per_node 4 eval_engine.py --config configs/mcn_refcoco_omni.py --eval-weights /path/to/checkpoint
```


## License

This project is released under the [Apache 2.0 license](LICENSE).


