package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/lab"
)


type SkinAccuracy struct{}

// Calculate implements metric interface. 
// 
// logits shape: [batch_size, n_classes]
// target shape: [batch_size] 
func(m *SkinAccuracy) Calculate(logits, target *ts.Tensor, opts ...lab.MetricOption) float64{
	yTrueDims := target.MustSize()
	yPredDims := logits.MustSize()

	if yTrueDims[0] != yPredDims[0]{
		err := fmt.Errorf("Expected dim 0 of logits and target equal. Got logits %+v - target %+v", yPredDims, yTrueDims)
		log.Fatal(err)
	}

	pred := logits.MustArgmax([]int64{1}, false, false)
	// fmt.Printf("taget shape: %v - argmax shape: %v\n", target.MustSize(), pred.MustSize())
	// fmt.Printf("target:%v\n", target.Float64Values())
	// fmt.Printf("pred: %v\n", pred.Float64Values())
	// fmt.Printf("eq: %v\n", pred.MustEq1(target, false).Float64Values())

	sumTs := pred.MustEqTensor(target, true).MustSum(gotch.Float, true)
	sum := sumTs.Float64Values()[0]
	sumTs.MustDrop()
	n := yTrueDims[0]
	avg := sum/float64(n)

	// fmt.Printf("average acc: %0.4f\n", avg)
	return avg

	// yPred := pred.Float64Values()
	// pred.MustDrop()
//
	// n := len(yTrue)
	// t := mat.NewDense(n, 1, yTrue)
	// p := mat.NewDense(n, 1, yPred)
//
	// acc := metrics.AccuracyScore(t, p, true, nil)
	// return acc
}

func NewSkinAccuracy() lab.Metric{
	return &SkinAccuracy{}
}

func(m *SkinAccuracy) Name() string{
	return "skin_accuracy"
}


type ValidLoss struct{
	lossFunc lab.LossFunc
}

func(m *ValidLoss) Calculate(logits, target *ts.Tensor, opts ...lab.MetricOption) float64{
	lossTs := m.lossFunc(logits, target)
	loss := lossTs.Float64Values()[0]
	lossTs.MustDrop()
	return loss
}

func(m *ValidLoss) Name() string{
	return "valid_loss"
}

func NewValidLoss(lossFunc lab.LossFunc) lab.Metric{
	return &ValidLoss{lossFunc}
}
