from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.simres import model

# Refine data path depend your own need
dataset.ann_path["refcoco"] = "simrec_data/anns/refcoco.json"
dataset.image_path["refcoco"] = "simrec_data/images/train2014"
dataset.mask_path["refcoco"] = "simrec_data/masks/refcoco"
dataset.sup_ann_path["refcoco"] = "simrec_data/anns/refcoco_0.1.json"
dataset.omni_ann_path["refcoco"] = "simrec_data/anns/refcoco_0.9.json"

dataset.ann_path["refcoco+"] = "simrec_data/anns/refcoco+.json"
dataset.image_path["refcoco+"] = "simrec_data/images/train2014"
dataset.mask_path["refcoco+"] = "simrec_data/masks/refcoco+"
dataset.sup_ann_path["refcoco+"] = "simrec_data/anns/refcoco+_0.1.json"
dataset.omni_ann_path["refcoco+"] = "simrec_data/anns/refcoco+_0.9.json"

dataset.ann_path["refcocog"] = "simrec_data/anns/refcocog_umd.json"
dataset.image_path["refcocog"] = "simrec_data/images/train2014"
dataset.mask_path["refcocog"] = "simrec_data/masks/refcocog_umd"
dataset.sup_ann_path["refcocog"] = "simrec_data/anns/refcocog_0.1.json"
dataset.omni_ann_path["refcocog"] = "simrec_data/anns/refcocog_0.9.json"

dataset.dataset = 'refcoco'
dataset.aug = True
dataset.input_shape = [480, 480]
dataset.max_token_length = 20

# Refine data cfg
train.batch_size = 64
train.evaluation.eval_batch_size = 256
train.data.pin_memory = True
train.data.persistent_workers = True
train.data.num_workers = 8
train.data.mean = [0.485, 0.456, 0.406]
train.data.std = [0.229, 0.224, 0.225]
train.data.logit_thold = [0.01, 0.2, 0.5] # set threshold for filtering pseudo-labels
train.data.omni_label = "box"             # point or box
# train.data.mean = [0., 0., 0.]
# train.data.std = [1., 1., 1.]

# Refine training cfg
train.burning_in_epochs = 15
train.epochs = 60
train.omni_weight = 1
train.scheduler.name = "step"
train.scheduler.decay_epochs = []
# train.scheduler.decay_epochs=[45, 50, 55]
train.ema.enabled = False
train.multi_scale_training.enabled = False
train.sync_bn.enabled = True
train.save_period = 1
train.log_period = 100
train.output_dir = "./output_omni/"

# Refine optim
optim.lr = train.base_lr

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.pretrained_weight_path="./data/weights/darknet_coco.weights"
