// package loss - common image segmentation loss functions.
package loss

import (
	"github.com/sugarme/gotch"
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
func DiceLoss1(logits, target *ts.Tensor, thresholdOpt ...float64) *ts.Tensor{
	threshold := 0.5
	if len(thresholdOpt) > 0 {
		threshold = thresholdOpt[0]
	}
	// Flatten
	// iflat := probability.MustView([]int64{-1}, false)
	iflat := logits.MustSigmoid(false).MustView([]int64{-1}, true)
	// iflat := logits.MustView([]int64{-1}, false)
	tflat := target.MustView([]int64{-1}, false)
	p := iflat.MustGt(ts.FloatScalar(threshold), true)
	t := tflat.MustGt(ts.FloatScalar(threshold), true)

	// p*t
	pt := p.MustMul(t, false)
	ptSum := pt.MustSum(gotch.Double, true)

	// sum(p) + sum(t)
	pSum := p.MustSum(gotch.Double, true)
	tSum := t.MustSum(gotch.Double, true)

	// (2 * sum(p*t))/(sum(p) + sum(t) + epsilon)
	// dice := (2 * overlap) / (union + 0.001)
	epsilon := 0.001 // to make sure not dividing by 0
	upperDiv := ptSum.MustMul1(ts.FloatScalar(2), true)
	lowerDiv := pSum.MustAdd(tSum, true).MustAdd1(ts.FloatScalar(epsilon), true)
	tSum.MustDrop()
	dice := upperDiv.MustDiv(lowerDiv, true)
	lowerDiv.MustDrop()

	return dice
}

// Ref. https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
func DiceLoss2(logits, target *ts.Tensor, thresholdOpt ...float64) *ts.Tensor{
	dtype := target.DType()
	smooth := 1.0
	xflat := logits.MustContiguous(false).MustView([]int64{-1}, true)
	yflat := target.MustContiguous(false).MustView([]int64{-1}, true)
	intersection := xflat.MustMul(yflat, false).MustSum(dtype, true)

	Asum := xflat.MustMul(xflat, false).MustSum(dtype, true)
	Bsum := yflat.MustMul(yflat, false).MustSum(dtype, true)

	numerator := intersection.MustMul1(ts.FloatScalar(2.0), true).MustAdd1(ts.FloatScalar(smooth), true)
	denominator := Asum.MustAdd(Bsum, true).MustAdd1(ts.FloatScalar(smooth), true)
	div := numerator.MustDiv(denominator, true)
	dice := div.MustMul1(ts.FloatScalar(-1), true).MustAdd1(ts.FloatScalar(1), true)

	denominator.MustDrop()
	xflat.MustDrop()
	yflat.MustDrop()

	return dice
}

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

	// (2 x intersection + smooth)/(cardinality + smooth)
	numerator := intersection.MustMul1(ts.FloatScalar(2), true).MustAdd1(ts.FloatScalar(smooth), true)
	denominator := cardinality.MustAdd1(ts.FloatScalar(smooth), true).MustClampMin(ts.FloatScalar(eps), true)
	score := numerator.MustDiv(denominator, true)
	denominator.MustDrop()

	loss := score.MustMul1(ts.FloatScalar(-1), true).MustAdd1(ts.FloatScalar(1), true)
	return loss
}

func SoftDice(x, y *ts.Tensor, thresholdOpt ...float64) *ts.Tensor{
	dims := []int64{-2, -1} // Calculate on last 2 dims (image H x W)
	smooth := 1.0

	// x*y
	xy := x.MustMul(y, false)

	// sum(x*y)
	tp := xy.MustSum1(dims, false, gotch.Double, true)

	// (1 - y)
	y1 := y.MustAdd1(ts.FloatScalar(-1), false)

	// x*(1-y)
	xy1 := y1.MustMul(x, true)

	// sum(x*(1-y))
	fp := xy1.MustSum1(dims, false, gotch.Double, true)

	// (1 - x)
	x1 := x.MustAdd1(ts.FloatScalar(-1), false)

	// y*(1-x)
	x1y := x1.MustMul(y, true)

	// sum(y*(1-x))
	fn := x1y.MustSum1(dims, false, gotch.Double, true)

	// 2 * sum(x*y) + smooth
	numerator := tp.MustMul1(ts.FloatScalar(2.0), false).MustAdd1(ts.FloatScalar(smooth), true)

	// (2 * sum(x*y) + smooth) + sum(x*(1-y)) + sum(y*(1-x))
	denominator := numerator.MustAdd(fp, false).MustAdd(fn, false)

	dc := numerator.MustDiv(denominator, true)

	tp.MustDrop()
	fp.MustDrop()
	fn.MustDrop()
	denominator.MustDrop()

	mean := dc.MustMean(gotch.Double, true)

	retVal := mean.MustMul1(ts.FloatScalar(-1), true).MustAdd1(ts.FloatScalar(1), true)
	return retVal
}
