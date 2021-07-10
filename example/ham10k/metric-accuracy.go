package main

import (
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/lab"
)


type SkinAccuracy struct{}

// logits shape: [n_classes]
// target shape: [1] - class index
func(m *SkinAccuracy) Calculate(logits, target *ts.Tensor, opts ...lab.MetricOption) float64{
	yTrue := target.Float64Values()[0]
	pred := logits.MustSigmoid(false).MustArgmax([]int64{0}, true, true)
	yPred := pred.Float64Values()[0]
	pred.MustDrop()
	if yTrue == yPred{
		return 1.0
	} 

	return 0
}

func NewSkinAccuracy(logger *lab.Logger, classes int) lab.Metric{
	return &SkinAccuracy{}
}

func(m *SkinAccuracy) Name() string{
	return "skin_accuracy"
}

