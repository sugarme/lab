package lab

import (
	"fmt"
	"log"
	"reflect"

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


// JaccardIndex measures the ratio of intersection over union.
//
// Jaccard's index (IoU - Intersection over Union) = (intersection/union) = (TP)/(TP + FN + FP)
func JaccardIndex(logits, target *ts.Tensor, opts ...MetricOption) float64{
	// Check dim
	if logits.MustSize()[0] != target.MustSize()[0]{
		err := fmt.Errorf("Expected same Dim 0 of inputs. Got %v and %v\n", logits.MustSize(), target.MustSize())
		// return nil, err
		log.Fatal(err)
	}

	dtype := target.DType()
	options := defaultMetricOptions()
	for _, o := range opts{
		o(options)
	}

	if options.Classes != nil && options.Mode == "BinaryMode"{
		err := fmt.Errorf("JaccardIndex: Masking classes is not supported with 'BinaryMode'\n")
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

	jaccard := scores.MustMean(dtype, true)

	retVal := jaccard.Float64Values()[0]
	jaccard.MustDrop()

	return retVal
}

// JaccardIndexMetric
type JaccardIndexMetric struct{}

func (m *JaccardIndexMetric) Calculate(logits, target *ts.Tensor, opts ...MetricOption) float64{
	return JaccardIndex(logits, target, opts...)
}

func (m *JaccardIndexMetric) Name() string{
	return "jaccard_index"
}

func NewJaccardIndexMetric() Metric{
	return &JaccardIndexMetric{}
}
