import pytorch_lightning as pl
import torch
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TreeDataset import FoliageDataset

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


lr_clip = 1e-5
bnm_clip = 1e-2

class FoliageFilter(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        c_in = 0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=self.hparams.use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=self.hparams.use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=self.hparams.use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=self.hparams.use_xyz,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 13, kernel_size=1),
        )

    def forward(self, pointcloud):
        l_xyz, l_features = [pointcloud], [None]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, sample_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        return self.fc_layer(l_features[0])

    def training_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            onehot = torch.argmax(logits, dim=1)
            acc = (onehot == labels).float().mean()
            tp = torch.sum((onehot==1)&(labels==1))
            pp = onehot.sum()
            ap = labels.sum()
            precision = tp.float() / pp.float()
            recall = tp.float() / ap.float()
            f1_score = 2 * precision * recall / (precision + recall)

        log = dict(train_loss=loss, train_acc=acc, train_precsion=precision, train_recall=recall, train_f1=f1_score)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
        pc, labels = batch

        logits = self.forward(pc)
        loss = F.cross_entropy(logits, labels)
        onehot = torch.argmax(logits, dim=1)
        acc = (onehot == labels).float().mean()
        tp = torch.sum((onehot == 1) & (labels == 1))
        pp = onehot.sum()
        ap = labels.sum()
        precision = tp.float() / pp.float()
        recall = tp.float() / ap.float()
        f1_score = 2 * precision * recall / (precision + recall)

        return dict(val_loss=loss, val_acc=acc, val_precsion=precision, val_recall=recall, val_f1=f1_score)

    def validation_epoch_end(self, outputs):
        reduced_outputs = {}
        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams.lr_decay
            ** (
                int(
                    self.global_step
                    * self.hparams.batch_size
                    / self.hparams.decay_step
                )
            ),
            lr_clip / self.hparams.lr,
        )
        bn_lbmd = lambda _: max(
            self.hparams.bn_momentum
            * self.hparams.bnm_decay
            ** (
                int(
                    self.global_step
                    * self.hparams.batch_size
                    / self.hparams.decay_step
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self, ds_path, mode):
        dset = FoliageDataset(ds_path)
        return DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=mode == "train",
            num_workers=4,
            pin_memory=True,
            drop_last=mode == "train",
        )

    def train_dataloader(self):
        return self._build_dataloader(self.hparams['train_data'], mode="train")

    def val_dataloader(self):
        return self._build_dataloader(self.hparams['val_data'], mode="val")
