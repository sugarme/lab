package main

import (
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/lab"
)

// Reduce LR by 0.1 every 10 epochs
func CustomScheduler(opt *nn.Optimizer) *lab.Scheduler{
		stepSize := 10
		gamma := 0.1
		s := nn.NewStepLR(opt, stepSize, gamma).Build()
		update := "on_epoch"
		name := "CustomLR"

		return lab.NewScheduler(s, name, update)
}