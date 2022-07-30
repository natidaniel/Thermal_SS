# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes, ignore_indices):
        self.n_classes = n_classes
        self.ignore_indices = ignore_indices
        self.legal_classes = [i for i in range(n_classes) if i not in ignore_indices]
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix
        if len(self.ignore_indices) > 0:
            hist = np.delete(hist,self.ignore_indices, 0)
            hist = np.delete(hist,self.ignore_indices, 1)

        acc = np.diag(hist).sum() / (hist.sum()+EPS)
        acc_cls = np.diag(hist) / (hist.sum(axis=1)+EPS)
        # acc_cls = np.nanmean(acc_cls)
        
        acc_cls = np.nan_to_num(acc_cls)
        acc_cls = np.mean(acc_cls)
        
        class_num_gt_pixels = hist.sum(axis=1)
        class_num_prediction_pixels = hist.sum(axis=0)
        
        class_precision = hist/(np.atleast_2d(class_num_gt_pixels).T+EPS)
        #class_precision = dict(zip(self.legal_classes, class_precision))
        class_recall = hist/(np.atleast_2d(class_num_prediction_pixels)+EPS)
        #class_recall = dict(zip(self.legal_classes, class_recall))

        iu = (np.diag(hist)+EPS) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) +EPS)
        # mean_iu = np.nanmean(iu)
        iu = np.nan_to_num(iu)
        mean_iu = np.mean(iu)

        freq = hist.sum(axis=1) / (hist.sum()+EPS)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        # cls_iu = dict(zip(range(self.n_classes), iu))
        n_valid_classes = self.n_classes
        for num in self.ignore_indices:
            n_valid_classes -= 1

        '''
        if self.has_invalid_class == True:
            n_valid_classes = self.n_classes - 1
        '''
        cls_iu = dict(zip(self.legal_classes, iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
                "Class precision : \t": class_precision,
                "Class recall : \t": class_recall,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count