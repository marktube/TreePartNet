import torch
import math
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule,  PointnetSAModuleMSG, build_shared_mlp
from torch.utils.data import DataLoader
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from TreeDataset import TreeDataset

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

class ScaledDot(nn.Module):
    '''
    Scaled Dot product
    :parameter
        input: BxCxN
        output: input*weight*transpose(input)
    '''
    def __init__(self, d_model):
        super(ScaledDot, self).__init__()
        self.d_model = d_model
        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256)
        )
        #self.linear = nn.Linear(256,256)

    def forward(self, input):
        projection = self.fc_lyaer(input)
        projection = torch.nn.functional.normalize(projection,p=2,dim=1)
        dot_product = torch.matmul(projection.permute(0,2,1),projection)
        return torch.abs(dot_product)

class FocalLoss(nn.Module):
    '''
    Focal Loss
        FL=alpha*(1-p)^gamma*log(p) where p is the probability of ground truth class
    Parameters:
        alpha(1D tensor): weight for positive
        gamma(1D tensor):
    '''
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha=torch.tensor(alpha)
        self.gamma=gamma
        self.reduce=reduce

    def forward(self, input, target):
        BCE_Loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_Loss)
        Focal_Loss = torch.pow((1-pt), self.gamma) * F.binary_cross_entropy_with_logits(
            input, target, pos_weight=self.alpha, reduction='none')

        if self.reduce=='none':
            return Focal_Loss
        elif self.reduce=='sum':
            return torch.sum(Focal_Loss)
        else:
            return torch.mean(Focal_Loss)

class TreePartNet(pl.LightningModule):
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams: hyper parameters
        '''
        super(TreePartNet,self).__init__()
        self.hparams = hparams
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=hparams['lc_count'],
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[hparams['input_channels'], 16, 16, 16],
                      [hparams['input_channels'], 32, 32, 32]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_0 = 16 + 32

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128],
                      [c_out_0, 64, 64, 128]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_1 = 128 + 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[64, 256, 64]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_0+c_out_1, 256, 64]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 2, kernel_size=1),
        )

        self.sharedMLP_layer = build_shared_mlp([64, 64, 32, 1])
        self.dot = ScaledDot(64)
        self.scale = nn.Parameter(torch.tensor(10.0, dtype=torch.float), requires_grad=True)
        #init_threshold = math.cos(math.pi / 9) * self.hparams['ED_weight']  # 20 degree
        #self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float), requires_grad=True)


    def forward(self, xyz):
        num_point = xyz.shape[1]

        # PointNet SA Module
        l_xyz, l_features, l_s_idx = [xyz], [None], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_s_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_s_idx.append(li_s_idx)

        # PointNet FP Module
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # Semantic Label Prediction
        sem_pred = self.fc_layer(l_features[0])

        # Local Context Label Prediction
        point_feat = torch.unsqueeze(l_features[0],dim=-2)
        point_feat = point_feat.repeat(1,1,self.hparams['lc_count'],1)
        lc_feat = torch.unsqueeze(l_features[-2],dim=-1)
        lc_feat = lc_feat.repeat(1,1,1,num_point)
        per_point_feat = point_feat - lc_feat

        lc_pred = self.sharedMLP_layer(per_point_feat)
        lc_pred = lc_pred.squeeze(dim=1)

        # Tree Edge Prediction
        dot = self.dot(l_features[-2])
        batch_idx = torch.tensor(range(xyz.shape[0]))
        batch_idx = batch_idx.unsqueeze(-1)
        batch_idx = batch_idx.repeat(1, self.hparams['lc_count'])
        s_xyz = xyz[batch_idx.cuda().long(), l_s_idx[-2].long()]
        s_xyz = torch.unsqueeze(s_xyz, dim=-2)
        s_xyz = s_xyz.repeat(1, 1, self.hparams['lc_count'], 1)
        dis = s_xyz - s_xyz.permute(0, 2, 1, 3)
        dis = dis ** 2
        dis = torch.sum(dis, dim=-1)
        dis = torch.sqrt(dis)

        fnode_pred = dot-self.scale*dis

        # FPS Sample Points
        lc_idx = l_s_idx[0]
        lc_idx = lc_idx[:,0:self.hparams['lc_count']]

        return sem_pred,lc_pred,fnode_pred,lc_idx


    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams['lr_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            lr_clip / self.hparams['lr'],
        )
        bn_lbmd = lambda _: max(
            self.hparams['bn_momentum']
            * self.hparams['bnm_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self,ds_path,shuff=True):
        dataset = TreeDataset(ds_path)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=4, shuffle=shuff)
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=True)

    def training_step(self, batch, batch_idx):
        pxyz,isfork,lcl,fn = batch
        pred_isfork, pred_lcl, fnode_pred, lc_idx = self(pxyz)
        critirion = torch.nn.CrossEntropyLoss()
        sem_loss = critirion(pred_isfork, isfork)
        lc_loss = critirion(pred_lcl, lcl)
        critirion2 = FocalLoss(alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduce='mean')
        fn_loss = critirion2(fnode_pred, fn)
        total_loss = sem_loss + lc_loss + fn_loss

        with torch.no_grad():
            sem_acc = (torch.argmax(pred_isfork, dim=1) == isfork).float().mean()
            lc_acc = (torch.argmax(pred_lcl, dim=1) == lcl).float().mean()
            # fnode prediction
            o = (fnode_pred > 0)
            o = o.int()
            # fn_acc = (o == fn).float().mean()
            tp = torch.sum((fn == 1) & (o == 1))
            pp = o.sum()
            ap = fn.sum()
            fn_recall = tp.float() / ap.float()
            fn_precision = tp.float() / pp.float()
            f1_score = 2 * fn_recall * fn_precision / (fn_recall + fn_precision)

        tensorboard_logs = {'train_loss': total_loss, 'semantic_loss': sem_loss, 'local_context_loss': lc_loss,
                            'fnode_loss': fn_loss, 'fnode_f1_score': f1_score,
                            'fnode_recall': fn_recall, 'fnode_precision': fn_precision,
                            'semantic_acc': sem_acc, 'local_context_acc': lc_acc,
                            }
        return {'loss': total_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=False)

    def validation_step(self, batch, batch_idx):
        pxyz, isfork, lcl, fn = batch
        pred_isfork, pred_lcl, fnode_pred, lc_idx = self(pxyz)
        critirion = torch.nn.CrossEntropyLoss()
        sem_loss = critirion(pred_isfork, isfork)
        lc_loss = critirion(pred_lcl, lcl)
        critirion2 = FocalLoss(alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduce='mean')
        fn_loss = critirion2(fnode_pred, fn)
        total_loss = sem_loss + lc_loss + fn_loss

        sem_acc = (torch.argmax(pred_isfork, dim=1) == isfork).float().mean()
        lc_acc = (torch.argmax(pred_lcl, dim=1) == lcl).float().mean()
        o = (fnode_pred > 0)
        o = o.int()
        # fn_acc = (o == fn).float().mean()
        tp = torch.sum((fn == 1) & (o == 1))
        pp = o.sum()
        ap = fn.sum()
        fn_recall = tp.float() / ap.float()
        fn_precision = tp.float() / pp.float()
        f1_score = 2 * fn_recall * fn_precision / (fn_recall + fn_precision)

        return {'val_loss': total_loss, 'val_semantic_acc': sem_acc, 'val_local_context_acc': lc_acc, 'val_local_context_loss': lc_loss,
                'val_fn_f1_score': f1_score, 'val_fn_recall': fn_recall, 'val_fnode_loss': fn_loss, 'val_fn_precision': fn_precision}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sem_acc = torch.stack([x['val_semantic_acc'] for x in outputs]).mean()
        lc_acc = torch.stack([x['val_local_context_acc'] for x in outputs]).mean()
        lc_loss = torch.stack([x['val_local_context_loss'] for x in outputs]).mean()
        f1_score = torch.stack([x['val_fn_f1_score'] for x in outputs]).mean()
        fn_recall = torch.stack([x['val_fn_recall'] for x in outputs]).mean()
        fn_precision = torch.stack([x['val_fn_precision'] for x in outputs]).mean()
        fn_loss = torch.stack([x['val_fnode_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_semantic_acc': sem_acc, 'val_local_context_acc': lc_acc,
                            'val_fn_f1_score': f1_score, 'val_fn_recall': fn_recall,'val_local_context_loss': lc_loss,
                            'val_fnode_loss': fn_loss, 'val_fn_precision': fn_precision}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

if __name__ == "__main__":
    #device = torch.device('cuda:0')
    net = TreePartNet(output_lc=256)
    #print(net)
    train_dataloader = net.train_dataloader()
    net = net.cuda()
    i = 0
    for points,forks,prim,fn in train_dataloader:
        i = i + 1
        '''transform_p = points.float()
        out,sem,idxs=net(transform_p.cuda())
        print(out.size())
        print(sem.size())
        print(idxs.size())'''
        pxyz, isfork, lcl, fnode = points,forks,prim,fn
        pred_isfork, pred_lcl, fn_pred, idx = net(pxyz.float().cuda())
        critirion = torch.nn.CrossEntropyLoss()
        sem_loss = critirion(pred_isfork, isfork.cuda())
        lc_loss = critirion(pred_lcl, lcl.cuda())
        total_loss = sem_loss + lc_loss
        print(total_loss.cpu())
        if i == 1:
            break
