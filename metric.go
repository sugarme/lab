package lab

import (
	"github.com/sugarme/gotch/ts"
)

type Metric interface {
	Calculate(logits, target *ts.Tensor, opts ...MetricOption) float64
	// Calculate(yTrue, yPred *ts.Tensor, opts ...MetricOption) float64
	Name() string
}
