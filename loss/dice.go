package loss

import (
	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// criterionBinaryCrossEntropy calculates loss from input logits and mask
// using Binary Cross Entropy with Logits algorithm.
func CriterionBinaryCrossEntropy(logits, mask *ts.Tensor) *ts.Tensor {
	// fmt.Printf("logit size: %v\n", logits.MustSize())
	// fmt.Printf("mask size: %v\n", mask.MustSize())
	// fmt.Printf("logit: %v\n", logit.Float64Values())
	// fmt.Printf("mask: %v\n", mask.Float64Values())
	// panic("stop")

	loss := logits.MustSqueeze(false).MustBinaryCrossEntropyWithLogits(mask, ts.NewTensor(), ts.NewTensor(), 1, true).MustView([]int64{-1}, true)
	return loss
	
	// logitR := logit.MustReshape([]int64{-1}, false)
	// maskR := mask.MustReshape([]int64{-1}, false)
//
	// // NOTE: reduction: none = 0; mean = 1; sum = 2. Default=mean
	// // ref. https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.binary_cross_entropy_with_logits
	// retVal := logitR.MustBinaryCrossEntropyWithLogits(maskR, ts.NewTensor(), ts.NewTensor(), 1, true).MustView([]int64{-1}, true)
	// maskR.MustDrop()
	// return retVal
}


// DiceLoss measures overlap between 2
// Ref. https://github.com/pytorch/pytorch/issues/1249
// http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
// https://www.jeremyjordan.me/semantic-segmentation/#:~:text=Another%20popular%20loss%20function%20for,denotes%20perfect%20and%20complete%20overlap.
func DiceScore(probability, mask *ts.Tensor, thresholdOpt ...float64) float64 {
	threshold := 0.5
	if len(thresholdOpt) > 0 {
		threshold = thresholdOpt[0]
	}
	// Flatten
	iflat := probability.MustView([]int64{-1}, false)
	tflat := mask.MustView([]int64{-1}, false)
	p := iflat.MustGt(ts.FloatScalar(threshold), true)
	t := tflat.MustGt(ts.FloatScalar(threshold), true)

	// p*t
	pt := p.MustMul(t, false)
	ptSum := pt.MustSum(gotch.Double, true)
	overlap := ptSum.Float64Values()[0]

	// sum(p) + sum(t)
	pSum := p.MustSum(gotch.Double, true)
	tSum := t.MustSum(gotch.Double, true)
	union := pSum.Float64Values()[0] + tSum.Float64Values()[0]

	// (2 * sum(p*t))/(sum(p) + sum(t) + epsilon)
	dice := (2 * overlap) / (union + 0.001)

	pSum.MustDrop()
	tSum.MustDrop()
	ptSum.MustDrop()

	return dice
}

// SoftDiceLoss calculates ratio between the overlap of the  positive instances
// between 2 sets, and their mutual combined values.
// https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
// Ref. https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08
// Ref. https://www.kaggle.com/finlay/pytorch-fcn-resnet50-in-20-minute
// Other name: Jaccard loss; Intersection over union loss
// (overlap(x, y))/(|x| + |y| - overlap(x, y))
func SoftDiceLoss(x, y *ts.Tensor) *ts.Tensor { // x prediction, y ground truth
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

// Ref. https://www.kaggle.com/finlay/pytorch-fcn-resnet50-in-20-minute
func LossFunc(logit, mask *ts.Tensor) *ts.Tensor {
	bce := CriterionBinaryCrossEntropy(logit, mask).MustMul1(ts.FloatScalar(0.8), true)
	prob := logit.MustSigmoid(false)
	dice := SoftDiceLoss(prob, mask).MustMul1(ts.FloatScalar(0.2), true)
	prob.MustDrop()

	retVal := bce.MustAdd(dice, true)
	dice.MustDrop()

	return retVal
}

// Accuracy calculates true positive and true negative.
// Default threshold = 0.5.
func Accuracy(input, target *ts.Tensor, thresholdOpt ...float64) (tp, tn float64) {
	threshold := 0.5
	if len(thresholdOpt) > 0 {
		threshold = thresholdOpt[0]
	}

	iflat := input.MustView([]int64{-1}, false)
	tflat := target.MustView([]int64{-1}, false)

	p := iflat.MustGt(ts.FloatScalar(threshold), true)
	t := tflat.MustGt(ts.FloatScalar(threshold), true)

	// p*t
	pt := p.MustMul(t, false)

	// sum(p*t)
	overlap := pt.MustSum(gotch.Double, true)

	// sum(t)
	tSum := t.MustSum(gotch.Double, false)

	// sum(p*t)/sum(t)
	tp = overlap.Float64Values()[0] / tSum.Float64Values()[0]
	overlap.MustDrop()
	tSum.MustDrop()

	// 1-p
	p1 := p.MustAdd1(ts.FloatScalar(-1), true)
	// 1-t
	t1 := t.MustAdd1(ts.FloatScalar(-1), true)

	// (1-p)*(1-t)
	p1t1 := p1.MustMul(t1, true)

	// sum((1-p)*(1-t))
	numerator := p1t1.MustSum(gotch.Double, true)

	// sum(1-t)
	denominator := t1.MustSum(gotch.Double, true)

	// sum((1-p)*(1-t))/sum(1-t)
	tn = numerator.Float64Values()[0] / denominator.Float64Values()[0]
	numerator.MustDrop()
	denominator.MustDrop()

	return tp, tn
}

