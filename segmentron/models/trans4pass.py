# flake8: noqa
import torch.nn.functional as F
from segmentron.models.model_zoo import MODEL_REGISTRY
from segmentron.models.segbase import SegBaseModel
from segmentron.config import cfg
# --- dmlpv1
from segmentron.modules.dmlp import DMLP as DMLPv1
# --- dmlpv2
from segmentron.modules.dmlpv2 import DMLP as DMLPv2

__all__ = ['Trans4PASS']


@MODEL_REGISTRY.register(name='Trans4PASS')
class Trans4PASS(SegBaseModel):

    def __init__(self):
        super().__init__()
        vit_params = cfg.MODEL.TRANS2Seg
        c4_HxW = (cfg.TRAIN.BASE_SIZE // 32) ** 2
        vit_params['decoder_feat_HxW'] = c4_HxW
        vit_params['nclass'] = self.nclass
        vit_params['emb_chans'] = cfg.MODEL.EMB_CHANNELS
        if cfg.MODEL.DMLP == 'DMLPv1':
            print("Using DMLPv1")
            self.dede_head = DMLPv1(vit_params)
        else:
            print("Using DMLPv2")
            self.dede_head = DMLPv2(vit_params)
        self.__setattr__('decoder', ['dede_head'])

    def forward(self, x):
        size = x.size()[2:]
        add_loss = {}
        feats, add_loss = self.encoder(x)
        c1, c2, c3, c4 = feats

        outputs = list()
        x = self.dede_head(c1, c2, c3, c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        return tuple(outputs), add_loss
