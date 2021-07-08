package main

import (
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/lab"
	"github.com/sugarme/lab/metric"
)


type DiceCoeffBatch struct{}
// Calculate implements Metric interface.
func(m *DiceCoeffBatch) Calculate(logits, target *ts.Tensor, threshold float64) float64{
	prob := logits.MustSigmoid(false)	
	dice := metric.DiceCoeffBatch(prob, target, threshold)
	prob.MustDrop()
	return dice
}

func(m *DiceCoeffBatch) Name() string{
	return "dice_coeff_batch"
}

func NewDiceCoeffBatch() lab.Metric{
	return &DiceCoeffBatch{}
}

