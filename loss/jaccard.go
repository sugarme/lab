package loss

import (
	"fmt"
	"log"
	"reflect"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

func softJaccardScore(output, target *ts.Tensor, smoothVal float64, epsVal float64, dims []int64) (*ts.Tensor, error){
	// Check size
	if !reflect.DeepEqual(output.MustSize(), target.MustSize()){
		err := fmt.Errorf("softJaccardScore - Expected output and target have the same shape. Got output %v and target %v\n", output.MustSize(), target.MustSize())
		return nil, err
	}

	var(
		intersection *ts.Tensor
		cardinality *ts.Tensor
	)

	dtype := target.DType()

	// Intersection: yTrue x yPred = 0 if pred = 0 or mask = 0
	// Cardinality: yTrue + yPred = 0 if BOTH pred = 0 and mask = 0
	switch{
	case dims != nil:
		intersection = output.MustMul(target, false).MustSum1(dims, false, dtype, true)
		cardinality = output.MustAdd(target, false).MustSum1(dims, false, dtype, true)
	default:
		intersection = output.MustMul(target, false).MustSum(dtype, true)
		cardinality = output.MustAdd(target, false).MustSum(dtype, true)
	}

	// union = cardinality - intersection
	union := cardinality.MustSub(intersection, true)

	smooth := ts.FloatScalar(smoothVal)
	eps := ts.FloatScalar(epsVal)
	numerator := intersection.MustAdd1(smooth, true)
	denominator := union.MustAdd1(smooth, true).MustClampMin(eps, true)

	score := numerator.MustDiv(denominator, true)
	denominator.MustDrop()
	smooth.MustDrop()
	eps.MustDrop()

	return score, nil
}

// JaccardLoss (JaccardIndex, IoU) calculates the ratio of intersection over union.
//
// Ref. https://en.wikipedia.org/wiki/Jaccard_index
func JaccardLoss(logits, target *ts.Tensor, opts ...Option) *ts.Tensor{

	// Check dim
	if logits.MustSize()[0] != target.MustSize()[0]{
		err := fmt.Errorf("Expected same Dim 0 of inputs. Got %v and %v\n", logits.MustSize(), target.MustSize())
		// return nil, err
		log.Fatal(err)
	}

	dtype := target.DType()
	options := defaultOptions()
	for _, o := range opts{
		o(options)
	}

	if options.Classes != nil && options.Mode == "BinaryMode"{
		err := fmt.Errorf("JaccardLoss: Masking classes is not supported with 'BinaryMode'\n")
		// return nil, err
		log.Fatal(err)
	}

	var output *ts.Tensor
	if options.FromLogits{
		if options.Mode == "MultiClassMode"{
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
	switch options.Mode{
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
	scores, err := softJaccardScore(yPred, yTrue, options.Smooth, options.Eps, dims)
	if err != nil{
		// return nil, err
		log.Fatal(err)
	}

	var loss *ts.Tensor
	eps := ts.FloatScalar(options.Eps)
	switch options.LogLoss{
	case true:
		// loss = -torch.log(scores.clamp_min(self.eps))
		loss = scores.MustClampMin(eps, true).MustLog(true).MustMul1(ts.FloatScalar(-1), true)
	case false:
		// loss = 1.0 - scores
		loss = scores.MustMul1(ts.FloatScalar(-1), true).MustAdd1(ts.FloatScalar(1), true)
	}

	// IoU loss is defined for non-empty classes
	// So we zero contribution of channel that does not have true pixels
	// NOTE: A better workaround would be to use loss term `mean(y_pred)`
	// for this case, however it will be a modified jaccard loss

	// mask = y_true.sum(dims) > 0
	yTrueSum := yTrue.MustSum1(dims, false, dtype, false)
	mask := yTrueSum.MustGt(ts.FloatScalar(0), true).MustTotype(gotch.Float, true)
  // loss *= mask.float()
	loss1 := loss.MustMul(mask, true)
	mask.MustDrop()

	// if self.classes is not None:
			// loss = loss[self.classes]
  // return loss.mean()
	var res *ts.Tensor
	if options.Classes != nil{
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

