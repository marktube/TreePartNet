import torch
import sys
import os
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
from PreProcess import FoliageFilter


model = FoliageFilter.load_from_checkpoint("./fckpt/epoch=28-val_loss=0.17-val_acc=0.951.ckpt")
print(model.hparams)
model = torch.nn.DataParallel(model)
model = model.cuda()
model.eval()

f = h5py.File('data/Foliage_Segmentation/tree_labeled_test.hdf5','r')
ds = f['points'][:]
fns = f['names'][:]
#centroid = f['centroid'][:]
#scales = f['scale'][:]
f.close()
preds_if = []

for i in range(len(ds)):
    pxyz = torch.from_numpy(ds[i]).float()
    pxyz = torch.unsqueeze(pxyz,dim=0)
    logits_if = model(pxyz.cuda())

    pred_if = torch.argmax(logits_if, dim=1)
    pred_if = torch.squeeze(pred_if)
    preds_if.append(pred_if.cpu().numpy())

with h5py.File('./foliage_seg.hdf5', 'w') as f:
    f['points'] = ds
    f['pred_isfoliage'] = preds_if
    f['names'] = fns
    #f['centroid'] = centroid
    #f['scale'] = scales

