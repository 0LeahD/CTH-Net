import numpy as np
from medpy.metric.binary import hd, hd95

__all__ = ['SegmentationMetric']

class SegmentationMetric (object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros ((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = np.diag (self.confusionMatrix).sum () / self.confusionMatrix.sum ()
        return acc

    def classPixelAccuracy(self):
        classAcc = np.diag (self.confusionMatrix) / self.confusionMatrix.sum (axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy ()
        meanAcc = np.nanmean (classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        intersection = np.diag (self.confusionMatrix)
        union = np.sum (self.confusionMatrix, axis=1) + np.sum (self.confusionMatrix, axis=0) - np.diag (
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean (IoU)
        return mIoU

    def recall(self):
        recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        recall = recall.tolist()
        del recall[0]
        recall = np.array(recall)
        recall = np.nanmean(recall)

        return recall

    def f_socre(self):
        cpa = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        Recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        dice = (2*cpa*Recall) / (cpa + Recall)
        dice = np.nanmean(dice)

        return dice

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount (label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape (self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum (self.confusionMatrix, axis=1) / np.sum (self.confusionMatrix)
        iu = np.diag (self.confusionMatrix) / (
                np.sum (self.confusionMatrix, axis=1) + np.sum (self.confusionMatrix, axis=0) -
                np.diag (self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum ()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix (imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros ((self.numClass, self.numClass))

    def Hausdorff_distance(self,imgPredict, imgLabel):
        hausdorff_distance = hd (imgPredict, imgLabel)
        hausdorff_distance95 = hd95 (imgPredict, imgLabel)
        return hausdorff_distance, hausdorff_distance95


