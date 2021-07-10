package lab

import (
	ts "github.com/sugarme/gotch/tensor"
)

// CrossEntropyLoss calculates cross entropy loss.
func CrossEntropyLoss(logits, target *ts.Tensor) *ts.Tensor{
	return logits.CrossEntropyForLogits(target)
}

// BCELoss calculates a binary cross entropy loss.
//
// - logits: tensor of shape [B, C, H, W] corresponding the raw output of the model.
// - target: ground truth tensor of shape [B, 1, H, W]
// - posWeight: scalar representing the weight attributed to positive class.
// This is especially useful for an imbalanced dataset
func BCELoss(logits, target *ts.Tensor) *ts.Tensor{
	weight := ts.NewTensor()
	posWeight := ts.NewTensor()
	// NOTE: reduction: none = 0; mean = 1; sum = 2. Default=mean
	reduction := int64(1)
	loss := logits.MustSqueeze(false).MustBinaryCrossEntropyWithLogits(target, weight, posWeight, reduction, true)
	return loss
}

// TODO. update when gotch for libtorch 1.9 being available.
// due to missing API: `atg_cross_entropy_loss`  
// // CELoss computes the weighted multi-class cross-entropy loss.
// //
// // - logits: tensor of shape [B, 1, H, W] corresponding the raw output of the model.
// // - target: ground truth tensor of shape [B, 1, H, W]
// // - weights: tensor of shape [C,] attributed to each class.
// func CELoss(logits, target, weights *ts.Tensor) *ts.Tensor{
	// // NOTE: reduction: none = 0; mean = 1; sum = 2. Default=mean
	// reduction := int64(1)
	// loss := logits.MustSqueeze(false).MustCrossEntropy(target, weights, reduction, true)
	// return loss
// }
