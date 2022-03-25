package lab

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

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
func DiceLoss(logits, target *ts.Tensor, opts ...MetricOption) *ts.Tensor {
	// Check dim
	if logits.MustSize()[0] != target.MustSize()[0] {
		err := fmt.Errorf("Expected same Dim 0 of inputs. Got %v and %v\n", logits.MustSize(), target.MustSize())
		// return nil, err
		log.Fatal(err)
	}

	dtype := target.DType()
	options := defaultMetricOptions()
	for _, o := range opts {
		o(options)
	}

	if options.Classes != nil && options.Mode == "BinaryMode" {
		err := fmt.Errorf("JaccardLoss: Masking classes is not supported with 'BinaryMode'\n")
		// return nil, err
		log.Fatal(err)
	}

	var output *ts.Tensor
	if options.FromLogits {
		if options.Mode == "MultiClassMode" {
			output = logits.MustLogSoftmax(1, dtype, false).MustExp(true)
		} else {
			// Apply activations to get [0..1] class probabilities
			// Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
			// extreme values 0 and 1
			output = logits.MustLogSigmoid(false).MustExp(true)
		}
	} else {
		output = logits.MustShallowClone()
	}

	bs := target.MustSize()[0]
	numClasses := output.MustSize()[1]
	dims := []int64{0, 2}

	var (
		yTrue *ts.Tensor
		yPred *ts.Tensor
	)
	switch options.Mode {
	case "BinaryMode":
		yTrue = target.MustView([]int64{bs, 1, -1}, false)
		yPred = output.MustView([]int64{bs, 1, -1}, true)

	case "MultiClassMode":
		// y_true = y_true.view(bs, -1)
		yT := target.MustView([]int64{bs, -1}, false)
		// y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
		// y_true = y_true.permute(0, 2, 1)  # N, C, H*W
		yTrue = yT.MustOneHot(numClasses, true).MustPermute([]int64{0, 2, 1}, true)

		// y_pred = y_pred.view(bs, num_classes, -1)
		yPred = output.MustView([]int64{bs, numClasses, -1}, true)

	case "MultiLabelMode":
		yTrue = target.MustView([]int64{bs, numClasses, -1}, false)
		yPred = output.MustView([]int64{bs, numClasses, -1}, true)
	}

	// scores shape = [classes] if `MultiClassMode` or `MultiLabelMode` or [1] if `BinaryMode`
	scores, err := softDiceScore(yPred, yTrue, options.Smooth, options.Eps, dims)
	if err != nil {
		// return nil, err
		log.Fatal(err)
	}

	var loss *ts.Tensor
	eps := ts.FloatScalar(options.Eps)
	switch options.LogLoss {
	case true:
		// loss = -torch.log(scores.clamp_min(self.eps))
		loss = scores.MustClampMin(eps, true).MustLog(true).MustMulScalar(ts.FloatScalar(-1), true)
	case false:
		// loss = 1.0 - scores
		loss = scores.MustMulScalar(ts.FloatScalar(-1), true).MustAddScalar(ts.FloatScalar(1), true)
	}

	// IoU loss is defined for non-empty classes
	// So we zero contribution of channel that does not have true pixels
	// NOTE: A better workaround would be to use loss term `mean(y_pred)`
	// for this case, however it will be a modified jaccard loss

	// mask = y_true.sum(dims) > 0
	yTrueSum := yTrue.MustSumDimIntlist(dims, false, dtype, false)
	mask := yTrueSum.MustGt(ts.FloatScalar(0), true).MustTotype(gotch.Float, true)
	// loss *= mask.float()
	loss1 := loss.MustMul(mask, true)
	mask.MustDrop()

	// if self.classes is not None:
	// loss = loss[self.classes]
	// return loss.mean()
	var res *ts.Tensor
	if options.Classes != nil {
		idx := ts.MustOfSlice(options.Classes)
		l := loss1.MustIndexSelect(0, idx, true)
		idx.MustDrop()
		res = l.MustMean(dtype, true)
	} else {
		res = loss1.MustMean(dtype, true)
	}

	yTrue.MustDrop()
	yPred.MustDrop()

	return res
}
