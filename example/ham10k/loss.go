package main

import (
	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/lab"
)

type lossFnOptions struct{
	ClassWeights []float64
	Reduction int64 // 0: "None", 1: "mean", 2: "sum"
	IgnoreIndex int64
}
type LossFnOption func(*lossFnOptions)
func WithLossFnWeights(vals []float64) LossFnOption{
	return func(o *lossFnOptions){
		o.ClassWeights = vals
	}
}
func WithLossFnReduction(val int64) LossFnOption{
	return func(o *lossFnOptions){
		o.Reduction = val
	}
}
func WithLossFnIgnoreIndex(val int64) LossFnOption{
	return func(o *lossFnOptions){
		o.IgnoreIndex = val
	}
}

func defaultLossFnOptions() *lossFnOptions{
	return &lossFnOptions{
		ClassWeights: nil,
		Reduction: 1, // "mean"
		IgnoreIndex: -100,
	}
}

// CrossEntropyLoss calculates cross entropy loss.
// Ref. https://github.com/pytorch/pytorch/blob/15be189f0de4addf4f68d18022500f67617ab05d/torch/nn/functional.py#L2012
func CustomCrossEntropyLoss(opts ...LossFnOption) lab.LossFunc{
	return func(logits, target *ts.Tensor) *ts.Tensor{
		options := defaultLossFnOptions()
		for _, o := range opts{
			o(options)
		}
		
		var ws *ts.Tensor
		device := logits.MustDevice()
		dtype := logits.DType()
		if len(options.ClassWeights) > 0{
			ws = ts.MustOfSlice(options.ClassWeights).MustTotype(dtype, true).MustTo(device, true)
		} else {
			ws = ts.NewTensor()
		}
		reduction := options.Reduction
		ignoreIndex := options.IgnoreIndex

		logSm := logits.MustLogSoftmax(-1, gotch.Float, false)
		loss := logSm.MustNllLoss(target, ws, reduction, ignoreIndex, true)
		ws.MustDrop()

		return loss
	}
}

/*

// TODO. Fixed this in gotch!!! Some leak here
func (ts *Tensor) CrossEntropyForLogits(targets *Tensor) (retVal *Tensor) {
	weight := NewTensor()
	reduction := int64(1) // Mean of loss
	ignoreIndex := int64(-100)

	logSm := ts.MustLogSoftmax(-1, gotch.Float, false)
	return logSm.MustNllLoss(targets, weight, reduction, ignoreIndex, true) // mem leak. Need delete logSm then return!!!
}

*/
