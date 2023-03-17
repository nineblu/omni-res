from omni_res.config import LazyCall
from omni_res.datasets.dataset import RefCOCODataSet
from .train import train

dataset = LazyCall(RefCOCODataSet)(
    # the dataset to be created
    # choose from ["refcoco", "refcoco+", "refcocog", "referit", "vg", "merge"]
    dataset = "refcoco",

    # path to the files
    ann_path = {
                'refcoco':'./data/anns/refcoco.json',
                'refcoco+': './data/anns/refcoco+.json',
                'refcocog': './data/anns/refcocog.json',
                'referit': './data/anns/refclef.json',
                'flickr': './data/anns/flickr.json',
                'vg': './data/anns/vg.json',
                'merge':'./data/anns/merge.json',
                'refcoco_merge': './data/anns/refcoco_merge.json',
            },
    image_path = {
                'refcoco': './data/images/coco',
                'refcoco+': './data/images/coco',
                'refcocog': './data/images/coco',
                'referit': './data/images/refclef',
                'flickr': './data/images/flickr',
                'vg':'./data/images/VG',
                'merge':'./data/images/',
                'refcoco_merge': './data/images/train2014',
            },
    mask_path = {
                'refcoco': './data/masks/refcoco',
                'refcoco+': './data/masks/refcoco+',
                'refcocog': './data/masks/refcocog',
                'referit': './data/masks/refclef',
                'refcoco_merge': './data/masks',
            },
    sup_ann_path = {
                'refcoco':'./data/anns/refcoco.json',
                'refcoco+': './data/anns/refcoco+.json',
                'refcocog': './data/anns/refcocog.json',
                'vg': './data/anns/vg.json',
                'refcoco_merge':'./data/anns/refcoco_merge.json',
    },
    omni_ann_path = {
                'refcoco':'./data/anns/refcoco.json',
                'refcoco+': './data/anns/refcoco+.json',
                'refcocog': './data/anns/refcocog.json',
                'vg': './data/anns/vg.json',
    },
    
    # original input image shape
    input_shape = [416, 416],
    flip_lr = False,

    # the max truncked length for language tokens
    max_token_length = 15,

    # use glove pretrained embeddings or not
    use_glove = True,

    # datasets splits
    split = "train",
    label = None,

    # basic transforms
    mean=train.data.mean, 
    std=train.data.std,

    # Augment
    aug=True,
)