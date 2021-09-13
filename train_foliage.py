import torch
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'models'))
from PreProcess import FoliageFilter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    hparams = {'batch_size': 4,
               'lc_count': 256,
               'input_channels': 0,
               'use_xyz': True,
               'lr': 1e-3,
               'weight_decay': 0.0,
               'lr_decay': 0.5,
               'decay_step': 3e5,
               'bn_momentum': 0.5,
               'bnm_decay': 0.5,
               'train_data': './data/Foliage_Segmentation/tree_labeled_train.hdf5',
               'val_data': './data/Foliage_Segmentation/tree_labeled_val.hdf5'
               }
    model = FoliageFilter(hparams=hparams)
    #early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'fckpt', "{epoch}-{val_loss:.2f}-{val_acc:.3f}"),
        save_top_k=3,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    trainer = pl.Trainer(
        gpus=8,
        distributed_backend='ddp',
        checkpoint_callback=checkpoint_callback,
        max_epochs=200,
    )
    trainer.fit(model)
