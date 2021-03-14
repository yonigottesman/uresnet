import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as torchvision

from config import Config


class Hook():
    def __init__(self, module):
        self.stored = None
        self.handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.stored = output

    def remove(self):
        self.handle.remove()


class FeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[20, 70, 10]):
        super().__init__()
        self.base_loss = F.l1_loss

        self.m_feat = torchvision.models.vgg16_bn(True).to(
            Config.DEVICE).features.eval()
        self.m_feat = self.m_feat.requires_grad_(False)

        blocks = [
            i - 1 for i, o in enumerate(self.m_feat.children())
            if isinstance(o, nn.MaxPool2d)
        ]
        loss_features = [self.m_feat[i] for i in blocks[2:5]]

        self.hooks = [Hook(module) for module in loss_features]
        self.wgts = layer_wgts

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.stored.clone() if clone else o.stored) for o in self.hooks]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        return sum(self.feat_losses)

    def __del__(self):
        [hook.remove() for hook in self.hooks]
