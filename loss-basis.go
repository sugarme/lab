package lab

import (
	"github.com/sugarme/gotch/ts"
)

// CrossEntropyLoss calculates cross entropy loss.
func CrossEntropyLoss(logits, target *ts.Tensor) *ts.Tensor {
	return logits.CrossEntropyForLogits(target)
}

// BCELoss calculates a binary cross entropy loss.
//
// - logits: tensor of shape [B, C, H, W] corresponding the raw output of the model.
// - target: ground truth tensor of shape [B, 1, H, W]
// - posWeight: scalar representing the weight attributed to positive class.
// This is especially useful for an imbalanced dataset
func BCELoss(logits, target *ts.Tensor) *ts.Tensor {
	weight := ts.NewTensor()
	posWeight := ts.NewTensor()
	// NOTE: reduction: none = 0; mean = 1; sum = 2. Default=mean
	reduction := int64(1)
	loss := logits.MustSqueeze(false).MustBinaryCrossEntropyWithLogits(target, weight, posWeight, reduction, true)
	return loss
}

// CELoss computes the weighted multi-class cross-entropy loss.
//
// - logits: tensor of shape [B, 1, H, W] corresponding the raw output of the model.
// - target: ground truth tensor of shape [B, 1, H, W]
// - weights: tensor of shape [C,] attributed to each class.
func CELoss(logits, target, weights *ts.Tensor) *ts.Tensor {
	// NOTE:
	// - reduction: none = 0; mean = 1; sum = 2. Default=mean
	// - labelSmoothing: a float in [0.0, 1.0]. Specifies the amount of smoothing when
	// computing the loss, where 0.0 means no smoothing. The targets become a mixture
	// of the original ground truth and a uniform distribution. See more: https://arxiv.org/abs/1512.00567
	// - ignoredIndex: Specifies a target value that is ignored and does not contribute to the input gradient.
	reduction := int64(1)
	labelSmoothing := 0.0
	ignoredIndex := int64(-100)
	loss := logits.MustSqueeze(false).MustCrossEntropyLoss(target, weights, reduction, ignoredIndex, labelSmoothing, true)
	return loss
}
