package main

import (
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/lab"
)


type DiceCoeffBatch struct{}
// Calculate implements Metric interface.
func(m *DiceCoeffBatch) Calculate(logits, target *ts.Tensor, opts ...lab.MetricOption) float64{
	return lab.DiceCoefficient(logits, target, opts...)
}

func(m *DiceCoeffBatch) Name() string{
	return "dice_coeff_batch"
}

func NewDiceCoeffBatch() lab.Metric{
	return &DiceCoeffBatch{}
}

