# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

import os
import torch
import shutil
import cv2 as cv
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from typing import List
from .customMetrics import prepare_metrics

class prepare_learning:
    """create a learning object for change detection:
    model: a change detection deep net that inherits properties of 'torch.nn.Module',
    loss: a loss function that inherits properties of 'torch.nn.modules.loss._Loss', default is None ('torch.nn.CrossEntropyLoss'),
    optim: an optimizer that inherits properties of 'torch.optim.Optimizer', default is None ('torch.optim.Adam'),
    schedular: a decay learning rate that inherits properties of 'torch.optim.lr_scheduler._LRScheduler', default is None,
    num_epoch: an integer demonstrates training epochs over dataloaders, default is 25,
    device: a string demonstrates using 'cpu' or 'cuda' for training change detection model, default is 'cpu',
    metrics: a list of strings for estimating change detection accuracy (available options are \
        'accuracy', 'kappa', 'fscore', 'similarity', 'precision', 'recall', default is ('accuracy', 'kappa', 'fscore', 'similarity'))
    verbose: a boolean indicator for showing training/validation results, default is True,
    save_dir: a string defining saving directory for the best change detection models during training, default is None (main directory)"""
    
    def __init__(
        self,
        model,
        loss = None,
        optim = None,
        scheduler = None,
        num_epoch: int = 25,
        device: str = 'cpu',
        metrics: List[str] = ('accuracy', 'kappa', 'fscore', 'similarity'),
        verbose: bool = True,
        save_dir: str = None
    ):
        super(prepare_learning, self).__init__()

        mdl = model.to(device)
        loss = nn.CrossEntropyLoss().to(device) if loss is None else loss.to(device)
        optim = torch.optim.Adam(mdl.parameters()) if optim is None else optim
        save_dir = os.path.join(os.path.abspath(''), 'res') if save_dir is None else save_dir
        if ~(os.path.isdir(save_dir)): os.makedirs(name=save_dir, exist_ok=True)

        self.mdl = mdl
        self.loss = loss
        self.optim = optim
        self.scheduler = scheduler
        self.num_epoch = num_epoch
        self.device = device
        self.metrics = metrics
        self.verbose = verbose
        self.save_dir = save_dir

    def train_epoch(self, train_dl, **kwargs):
        self.mdl.train()
        
        train_log = []
        with tqdm(train_dl, desc='train', unit='batch', disable=not(self.verbose)) as tepoch:
            for base, target, mask, _ in tepoch:
                base, target, mask = base.to(self.device), target.to(self.device), mask.long().to(self.device)
                
                pred = self.mdl(base, target)
                loss = self.loss(pred, mask)

                x1 = {'loss': '{:.2f}'.format(loss.item())}
                m  = prepare_metrics()(inputs=pred, targets=mask, **kwargs)
                x2 = dict([(i, '{:.2f}'.format(m[i])) for i in m if i in set(self.metrics)])
                tepoch.set_postfix({**x1, **x2})

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_log.append({**x1, **x2})
        return pd.DataFrame(train_log).apply(pd.to_numeric)

    def valid_epoch(self, valid_dl, **kwargs):
        self.mdl.eval()
        
        valid_log = []
        with tqdm(valid_dl, desc='valid', unit='batch', disable=not(self.verbose)) as tepoch:
            for base, target, mask, _ in tepoch:
                base, target, mask = base.to(self.device), target.to(self.device), mask.long().to(self.device)

                with torch.no_grad():
                    pred = self.mdl(base, target)
                    loss = self.loss(pred, mask)

                x1 = {'loss': '{:.2f}'.format(loss.item())}
                m  = prepare_metrics()(inputs=pred, targets=mask, **kwargs)
                x2 = dict([(i, '{:.2f}'.format(m[i])) for i in m if i in set(self.metrics)])
                tepoch.set_postfix({**x1, **x2})

                valid_log.append({**x1, **x2})
        return pd.DataFrame(valid_log).apply(pd.to_numeric)

    def test_epoch(self, test_dl, **kwargs):
        save_dir = os.path.join(self.save_dir, 'vis')
        if os.path.isdir(save_dir): shutil.rmtree(path=save_dir)
        os.mkdir(path=save_dir)

        if test_dl.dataset.label_mask:
            self.mdl.eval()

            test_log = []
            with tqdm(test_dl, desc='test', unit='batch', disable=not(self.verbose)) as tepoch:
                for base, target, mask, fname in tepoch:
                    base, target, mask = base.to(self.device), target.to(self.device), mask.long().to(self.device)

                    with torch.no_grad():
                        pred = self.mdl(base, target)
                        loss = self.loss(pred, mask)

                    x1 = {'loss': '{:.2f}'.format(loss.item())}
                    m  = prepare_metrics()(inputs=pred, targets=mask, **kwargs)
                    x2 = dict([(i, '{:.2f}'.format(m[i])) for i in m if i in set(self.metrics)])
                    tepoch.set_postfix({**x1, **x2})

                    test_log.append({**x1, **x2})
                    
                    pred = pred.log_softmax(dim=1).exp().argmax(dim=1)
                    for lbl, nme in zip(pred, fname):
                        cv.imwrite(os.path.join(save_dir, os.path.basename(nme)), 255 * lbl.numpy())
            return pd.DataFrame(test_log).apply(pd.to_numeric)

        else:
            self.mdl.eval()
            
            with tqdm(test_dl, desc='test', unit='batch', disable=not(self.verbose)) as tepoch:
                for base, target, fname in tepoch:
                    base, target = base.to(self.device), target.to(self.device)

                    with torch.no_grad():
                        pred = self.mdl(base, target)
                    
                    pred = pred.log_softmax(dim=1).exp().argmax(dim=1)
                    for lbl, nme in zip(pred, fname):
                        cv.imwrite(os.path.join(save_dir, os.path.basename(nme)), 255 * lbl.numpy())
            return None

    def train(self, data_loader, score='similarity', **kwargs):
        save_dir = os.path.join(self.save_dir, 'mdl')
        if os.path.isdir(save_dir): shutil.rmtree(path=save_dir)
        os.mkdir(path=save_dir)

        max_score = 0
        train_logs, valid_logs = pd.DataFrame(), pd.DataFrame()
        train_logs_mean, valid_logs_mean = pd.DataFrame(), pd.DataFrame()
        for epoch in range(1, self.num_epoch + 1):
            if self.verbose: print('\nEpoch: {}'.format(epoch))

            train_log = self.train_epoch(data_loader['train'], **kwargs)
            train_logs = pd.concat([train_logs, train_log])
            train_logs_mean = pd.concat([train_logs_mean, train_log.mean()])
            
            valid_log = self.valid_epoch(data_loader['valid'], **kwargs)
            valid_logs = pd.concat([valid_logs, valid_log])
            valid_logs_mean = pd.concat([valid_logs_mean, valid_log.mean()])
            
            if self.scheduler: self.scheduler.step()

            if max_score < valid_log.mean()[score]:
                max_score = valid_log.mean()[score]
                torch.save(self.mdl, os.path.join(save_dir, 'change_detection_model_epoch_%d.pth' %epoch))
        return (train_logs, train_logs_mean), (valid_logs, valid_logs_mean)
    
    def predict(self, data_loader, **kwargs):
        test_logs = self.test_epoch(data_loader, **kwargs)
        return test_logs