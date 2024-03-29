from omni_res.config import LazyCall
from omni_res.models.simres import SimRES
from omni_res.models.backbones import DarkNet53
from omni_res.models.heads import SimREShead
from omni_res.models.language_encoders.lstm_sa import LSTM_SA
from omni_res.layers.fusion_layer import SimpleFusion, MultiScaleFusion, GaranAttention

model = LazyCall(SimRES)(
    visual_backbone=LazyCall(DarkNet53)(
        pretrained=False,
        pretrained_weight_path="./data/weights/darknet53_coco.pth",
        freeze_backbone=True,
        multi_scale_outputs=True,
    ),
    language_encoder=LazyCall(LSTM_SA)(
        depth=3,
        hidden_size=512,
        num_heads=8,
        ffn_size=2048,
        flat_glimpses=1,
        word_embed_size=300,
        dropout_rate=0.1,
        # language_encoder.pretrained_emb and language.token_size is meant to be set
        # before instantiating
        freeze_embedding=True,
        use_glove=True,
    ),
    multi_scale_manner=LazyCall(MultiScaleFusion)(
        v_planes=(256, 512, 1024),
        scaled=True
    ),
    fusion_manner=LazyCall(SimpleFusion)(
        v_planes=1024,
        q_planes=512,
        out_planes=1024,
    ),
    det_attention=LazyCall(GaranAttention)(
        d_q=512,
        d_v=512
    ),
    seg_attention=LazyCall(GaranAttention)(
        d_q=512,
        d_v=512
    ),
    head=LazyCall(SimREShead)(
        hidden_size=512, 
        anchors=[[137, 256], [248, 272], [386, 271]], 
        arch_mask=[[0, 1, 2]], 
        layer_no=0, 
        in_ch=512, 
        n_classes=0, 
        ignore_thre=0.5,
    )
)