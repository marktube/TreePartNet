import torch
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
from TreePartNet import TreePartNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    hparams = {'batch_size': 2,
               'lc_count' : 256,
               'input_channels' : 0,
               'use_xyz' : True,
               'lr': 0.05,
               'weight_decay': 0.0,
               'lr_decay': 0.5,
               'decay_step': 3e5,
               'bn_momentum': 0.5,
               'bnm_decay': 0.5,
               'FL_alpha': 253/192,
               'FL_gamma': 2,
               'train_data': './data/Neural_Decomposition/tree_labeled_train.hdf5',
               'val_data': './data/Neural_Decomposition/tree_labeled_val.hdf5'
               }
    model = TreePartNet(hparams)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(),'ckpt'),
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    trainer = pl.Trainer(
        gpus=[0,1,2,4,5,6,7],
        distributed_backend='ddp',
        checkpoint_callback=checkpoint_callback,
        max_epochs=500,
    )
    trainer.fit(model)
    #trainer.save_checkpoint('tpn.ckpt')
