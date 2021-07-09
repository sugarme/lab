// package metric - common image segmentation metrics.
package metric

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// JaccardIndex compute the intersection over union.
// ref: https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/metrics.py#L12
func JaccardIndex1(pred, target *ts.Tensor, nclasses int64) float64 {
	hist := FastHist(pred, target, nclasses)
	intersect := hist.MustDiag(0, false)
	eps := 1e-10
	a := hist.MustSum1([]int64{1}, false, gotch.Double, false)
	b := hist.MustSum1([]int64{0}, false, gotch.Double, false)

	denominator := a.MustAdd(b, true).MustSub(intersect, true).MustAdd1(ts.FloatScalar(eps), true)
	b.MustDrop()

	jaccard := intersect.MustDiv(denominator, true)

	denominator.MustDrop()

	mean := jaccard.MustMean(gotch.Double, true)
	retVal := mean.Float64Values()[0]
	jaccard.MustDrop()
	mean.MustDrop()

	return retVal
}

// DiceCoeff calculates Intersection over Union.
// Ref. https://github.com/milesial/Pytorch-UNet/blob/master/dice_loss.py
// NOTE: Dice Coefficient for individual examples.
func DiceCoeff(prob, mask *ts.Tensor) float64 {
	eps := 0.0001
	p := prob.MustView([]int64{-1}, false)
	m := mask.MustView([]int64{-1}, false)

	// 2 * intersection + eps
	numerator := p.MustDot(m, false).MustMul1(ts.FloatScalar(2.0), true).MustAdd1(ts.FloatScalar(eps), true)
	p.MustDrop()
	m.MustDrop()

	// union
	pSum := prob.MustSum(gotch.Double, false)
	mSum := mask.MustSum(gotch.Double, false)
	denominator := pSum.MustAdd(mSum, true).MustAdd1(ts.FloatScalar(eps), true)
	mSum.MustDrop()

	dice := numerator.MustDiv(denominator, true)
	denominator.MustDrop()

	retVal := dice.Float64Values()[0]
	dice.MustDrop()

	return retVal
}

// DiceCoeffBatch calculates dice coefficient by batch
func DiceCoeffBatch(prob, target *ts.Tensor, thresholdOpt ...float64) float64 {
	var threshold float64 = 0.5
	if len(thresholdOpt) > 0 {
		threshold = thresholdOpt[0]
	}

	pred := prob.MustGreater(ts.FloatScalar(threshold), false).MustTotype(gotch.Float, true)
	var diceCum float64 = 0
	bs := int(pred.MustSize()[0])
	for i := 0; i < bs; i++ {
		x := pred.Idx(ts.NewSelect(int64(i))) // select first element of axis 0
		y := target.Idx(ts.NewSelect(int64(i)))
		dice := DiceCoeff(x, y)
		diceCum += dice
		x.MustDrop()
		y.MustDrop()
	}

	pred.MustDrop()

	return diceCum / float64(bs)
}

// FastHist computes confusion mastrix.
// ref: https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/metrics.py#L12
func FastHist(pred, target *ts.Tensor, nclasses int64) *ts.Tensor {
	t1 := target.MustGreaterEqual(ts.FloatScalar(0.0), false)
	t2 := target.MustLess(ts.IntScalar(nclasses), false)
	fmt.Printf("t1: \n%v", t1)
	fmt.Printf("t2: \n%v", t2)
	mask := t1.MustLogicalAnd(t2, false)
	t2.MustDrop()

	idxTs := []ts.Tensor{*mask}
	targetIdx := target.MustIndex(idxTs, false).MustMul1(ts.FloatScalar(float64(nclasses)), true)
	predIdx := pred.MustIndex(idxTs, false)
	x := targetIdx.MustAdd(predIdx, true).MustTotype(gotch.Int, true)
	mask.MustDrop()
	predIdx.MustDrop()

	hist := x.MustBincount(ts.NewTensor(), nclasses*nclasses, true).MustReshape([]int64{nclasses, nclasses}, true)

	return hist
}


// IoU calculates intersection over union.
// Ref: https://discuss.pytorch.org/t/understanding-different-metrics-implementations-iou/85817
func IoU(logits, mask *ts.Tensor) float64 {
	eps := 1e-6
	pred := logits.MustSqueeze1(1, false)
	intersection := pred.MustLogicalAnd(mask, false).MustSum1([]int64{1, 2}, false, gotch.Double, true) // Will be zero if pred = 0 or mask = 0
	union := pred.MustLogicalOr(mask, true).MustSum1([]int64{1, 2}, false, gotch.Double, true)          // will be zero if both pred = 0 and mask=0

	numerator := intersection.MustAdd1(ts.FloatScalar(eps), true)
	denominator := union.MustAdd1(ts.FloatScalar(eps), true)

	iou := numerator.MustDiv(denominator, true)
	denominator.MustDrop()
	/*
	 *   //torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
	 *   x := iou.MustSub1(ts.FloatScalar(0.5), true).MustMul1(ts.IntScalar(20), true)
	 *   xclamp := x.MustClamp(ts.IntScalar(0), ts.IntScalar(10), true).MustCeil(true)
	 *   out := xclamp.MustDiv1(ts.IntScalar(10), true)
	 *   retVal := out.Float64Values()[0]
	 *   out.MustDrop()
	 *  */

	iouMean := iou.MustMean(gotch.Double, true)
	retVal := iouMean.Float64Values()[0]
	iouMean.MustDrop()

	return retVal
}

