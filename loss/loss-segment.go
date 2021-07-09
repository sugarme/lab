// package loss - common image segmentation loss functions.
package loss

import (
	ts "github.com/sugarme/gotch/tensor"
)

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

// DiceLoss calculates the Sorensen-Dice loss.
// 
// NOTE. brief notes on DICE
// - Jaccard's index (IoU - Intersection over Union) = (intersection/union) = (TP)/(TP + FN + FP)
// - Dice Coefficient measures of overlap between 2 masks - target and predict mask where 
//   1 is perfect overlap while 0 indicates non-overlap.
//   Dice Coefficient = (2 x Intersection)/(Union + Intersection) = (2xTP)/(2xTP + FN + FP)
// - Dice Loss = 1 - Dice Coefficient
// Ref. https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
// Ref. https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py#L168-L179
func DiceLoss(logits, target *ts.Tensor, thresholdOpt ...float64) *ts.Tensor{
	dtype := target.DType()
	smooth := 1.0
	eps := 1e-7

	// Apply activations to get [0..1] class probabilities
  // Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
  // extreme values 0 and 1
	bs := logits.MustSize()[0]
	yPred := logits.MustLogSigmoid(false).MustExp(true).MustView([]int64{bs, 1, -1}, true)
	yTrue := target.MustView([]int64{bs, 1, -1}, false) 

	// output * target
	intersection := yPred.MustMul(yTrue, false).MustSum(dtype, true)

	// output + target
	cardinality := yPred.MustAdd(yTrue, false).MustSum(dtype, true)

	yPred.MustDrop()
	yTrue.MustDrop()

	// Dice Coefficient = (2 x intersection + smooth)/(cardinality + smooth)
	numerator := intersection.MustMul1(ts.FloatScalar(2), true).MustAdd1(ts.FloatScalar(smooth), true)
	denominator := cardinality.MustAdd1(ts.FloatScalar(smooth), true).MustClampMin(ts.FloatScalar(eps), true)
	score := numerator.MustDiv(denominator, true)
	denominator.MustDrop()

	// Dice loss = 1 - Dice Coefficient
	loss := score.MustMul1(ts.FloatScalar(-1), true).MustAdd1(ts.FloatScalar(1), true)
	return loss
}

