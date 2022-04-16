# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

from sklearn.metrics import *
from .functional import prepare_kwargs

class prepare_metrics:
    """prepare a custom metric function for change detection"""

    def __init__(self):
        super(prepare_metrics, self).__init__()

    def __call__(self, inputs, targets, **kwargs):
        inputs = inputs.log_softmax(dim=1).exp().argmax(dim=1).view(-1)
        targets = targets.long().view(-1)
        return {
            'accuracy'  : accuracy_score    (targets, inputs, **prepare_kwargs(kwargs, accuracy_score   )), 
            'kappa'     : cohen_kappa_score (targets, inputs, **prepare_kwargs(kwargs, cohen_kappa_score)), 
            'fscore'    : f1_score          (targets, inputs, **prepare_kwargs(kwargs, f1_score         )), 
            'similarity': jaccard_score     (targets, inputs, **prepare_kwargs(kwargs, jaccard_score    )), 
            'precision' : precision_score   (targets, inputs, **prepare_kwargs(kwargs, precision_score  )), 
            'recall'    : recall_score      (targets, inputs, **prepare_kwargs(kwargs, recall_score     ))
        }