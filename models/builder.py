import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.init_func import init_weight
from utils.loss_opr import FocalLoss2d
from utils.losses import BinaryFocalLoss
from .decoders.RGBDecoder import RGBDecoderHead
from .decoders.MyDecoder import DecoderHead

from engine.logger import get_logger
logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(weight = torch.from_numpy(np.array(
            [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            pass

        self.aux_head = None

        self.head = DecoderHead(in_channels=[64, 128, 320, 512], num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        self.rgb_head = RGBDecoderHead(in_channels=[64, 128, 320, 512], num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        self.rgbx_head = RGBDecoderHead(in_channels=[64, 128, 320, 512], num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)

        self.linear_pred = nn.Conv2d(cfg.decoder_embed_dim, cfg.num_classes, kernel_size=1)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

        self.class_weight = torch.from_numpy(np.array([1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4459, 23.7228])).float()

        ### Loss
        self.focal = FocalLoss2d(weight=self.class_weight)
        self.dice_bi = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.dice_bo = nn.CrossEntropyLoss(weight=self.class_weight_boundary)


    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        orisize = rgb.shape
        x_fuse, x_rgb, x_e = self.backbone(rgb, modal_x)

        out,bi,bo = self.head.forward(x_fuse)
        out1 = self.rgb_head.forward(x_rgb)
        out2 = self.rgbx_head.forward(x_e)

        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        bi = F.interpolate(bi, size=orisize[2:], mode='bilinear', align_corners=False)
        bo = F.interpolate(bo, size=orisize[2:], mode='bilinear', align_corners=False)

        out1 = F.interpolate(self.linear_pred(out1), size=orisize[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.linear_pred(out2), size=orisize[2:], mode='bilinear', align_corners=False)

        return out, out1, out2, bi, bo

    def forward(self, rgb, modal_x, label=None, binary = None, boundary = None):
        out, out1, out2, bi, bo = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss1 = self.criterion(out, label.long())+self.focal(out,label.long())
            loss_bi = self.dice_bi(bi, binary)
            loss_bo = self.dice_bo(bo, boundary)
            loss2 = self.criterion(out1, label.long())
            loss3 = self.criterion(out2, label.long())
            return loss1 + 0.5*loss2 + 0.3*loss3 + 0.2*loss_bi + 0.2*loss_bo

        return out