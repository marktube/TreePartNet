import torch
import sys
import os
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
from TreePartNet import TreePartNet
from TreeDataset import TreeDataset
from torch.utils.data import DataLoader

model = TreePartNet.load_from_checkpoint("ckpt/epoch=167.ckpt")
model = model.cuda()
print(model.hparams)
print(model.state_dict()['scale'])
#print(model.state_dict()['threshold'])
model.eval()
ds = TreeDataset('data/Neural_Decomposition/tree_labeled_test.hdf5')
preds_if = []
preds_lcl = []
preds_fnode = []
fps_idx = []

for i in range(len(ds)):
    pxyz, forks, prim, fn = ds[i]
    pxyz=torch.unsqueeze(pxyz,dim=0)
    logits_if, logits_lcl, logits_fnode, idx = model(pxyz.cuda())

    pred_if = torch.argmax(logits_if, dim=1)
    pred_if = torch.squeeze(pred_if)
    preds_if.append(pred_if.cpu().numpy())

    pred_lcl = torch.argmax(logits_lcl, dim=1)
    pred_lcl = torch.squeeze(pred_lcl)
    preds_lcl.append(pred_lcl.cpu().numpy())

    pred_fnode = logits_fnode>0
    pred_fnode = pred_fnode.int()
    pred_fnode = torch.squeeze(pred_fnode)
    preds_fnode.append(pred_fnode.cpu().numpy())

    idx = torch.squeeze(idx)
    fps_idx.append(idx.cpu().numpy())


with h5py.File('./tree_'+'test_pred'+'.hdf5', 'w') as f:
    fread = h5py.File('data/Neural_Decomposition/tree_labeled_test.hdf5', 'r')
    point_sets = fread['points'][:]
    normals = fread['normals'][:]
    isforks = fread['isforks'][:]
    primitives = fread['primitive_id'][:]
    #fps_id = fread['samples'][:]
    codebook = fread['codebook'][:]
    fns = fread['names'][:]
    fread.close()
    f['points'] = point_sets
    f['normals'] = normals
    f['isforks'] = isforks
    f['primitive_id'] = primitives
    f['samples'] = fps_idx
    f['codebook'] = codebook
    f['pred_fnode'] = preds_fnode
    f['pred_isfork'] = preds_if
    f['pred_lc'] = preds_lcl
    f['names'] = fns

