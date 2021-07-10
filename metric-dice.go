package lab

import (
	"fmt"
	"log"
	"reflect"

	ts "github.com/sugarme/gotch/tensor"
)

func softDiceScore(output, target *ts.Tensor, smoothVal float64, epsVal float64, dims []int64) (*ts.Tensor, error){
	// Check size
	if !reflect.DeepEqual(output.MustSize(), target.MustSize()){
		err := fmt.Errorf("softDiceScore - Expected output and target have the same shape. Got output %v and target %v\n", output.MustSize(), target.MustSize())
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

	smooth := ts.FloatScalar(smoothVal)
	eps := ts.FloatScalar(epsVal)

	// dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
	numerator := intersection.MustMul1(ts.FloatScalar(2), true).MustAdd1(smooth, true)
	denominator := cardinality.MustAdd1(smooth, true).MustClampMin(eps, true)

	score := numerator.MustDiv(denominator, true)
	denominator.MustDrop()
	smooth.MustDrop()
	eps.MustDrop()

	return score, nil
}


// DiceCoefficient measures the overlap between 2 masks - target and predict mask where
// 1 is perfect overlap while 0 indicates non-overlap.
//
// Dice Coefficient = (2 x Intersection)/(Union + Intersection) = (2xTP)/(2xTP + FN + FP)
func DiceCoefficient(logits, target *ts.Tensor, opts ...MetricOption) float64{
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
	scores, err := softDiceScore(yPred, yTrue, options.Smooth, options.Eps, dims)
	if err != nil{
		// return nil, err
		log.Fatal(err)
	}

	coeff := scores.MustMean(dtype, true)
	
	retVal := coeff.Float64Values()[0]
	coeff.MustDrop()

	return retVal
}


// DiceCoefficientMetric
type DiceCoefficientMetric struct{}

func(m *DiceCoefficientMetric) Calculate(logits, target *ts.Tensor, opts ...MetricOption) float64{
	return DiceCoefficient(logits, target, opts...)
}

func (m *DiceCoefficientMetric) Name() string{
	return "dice_coefficient"
}

func NewDiceCoefficientMetric() Metric{
	return &DiceCoefficientMetric{}
}
