package main

import (
	"fmt"
	"log"

	"github.com/pa-m/sklearn/metrics"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/lab"
	"gonum.org/v1/gonum/mat"
)


type SkinAccuracy struct{}

// logits shape: [batch_size, n_classes]
// target shape: [batch_size] 
func(m *SkinAccuracy) Calculate(logits, target *ts.Tensor, opts ...lab.MetricOption) float64{
	yTrueDims := target.MustSize()
	yPredDims := logits.MustSize()

	if yTrueDims[0] != yPredDims[0]{
		err := fmt.Errorf("Expected dim 0 of logits and target equal. Got logits %+v - target %+v", yPredDims, yTrueDims)
		log.Fatal(err)
	}

	yTrue := target.Float64Values()
	pred := logits.MustArgmax([]int64{1}, true, false)
	yPred := pred.Float64Values()
	pred.MustDrop()

	n := len(yTrue)
	t := mat.NewDense(n, 1, yTrue) 
	p := mat.NewDense(n, 1, yPred) 

	acc := metrics.AccuracyScore(t, p, true, nil)
	return acc
}

func NewSkinAccuracy() lab.Metric{
	return &SkinAccuracy{}
}

func(m *SkinAccuracy) Name() string{
	return "skin_accuracy"
}

