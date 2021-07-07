package lab

import (
	"fmt"
	"log"

	ts "github.com/sugarme/gotch/tensor"
)

type Metric interface{
	Calculate(logits, target *ts.Tensor, thresholds float64) float64
	Name() string
}

type AUC struct{}

func(m *AUC) Calculate(logits, target *ts.Tensor, thresholds float64) float64{
	// yPred.shape = (N, C)
  // AUC for melanoma (class 0)
	// return metrics.ROCAUCScore()
	// TODO.
	panic("Not implemented")
}
func(m *AUC) Name() string{
	return "auc"
}

type F1 struct{}
func(m *F1) Calculate(logits, target *ts.Tensor, thresholds float64) float64{
	// TODO.
	panic("Not implemented")
}
func(m *F1) Name() string{
	return "f1"
}

type Accuracy struct{}
func(m *Accuracy) Calculate(logits, target *ts.Tensor, threshold float64) float64{
	// TODO.
	panic("Not implemented")
}
func(m *Accuracy) Name() string{
	return "f1"
}

func NewMetric(name string) Metric{
	var m Metric
	switch name{
	case "auc":
		m = &AUC{}
	case "f1":
		m = &F1{}
	case "accuracy":
		m = &Accuracy{}
	default:
		err := fmt.Errorf("Unsupported metric %q\n", name)
		log.Fatal(err)
	}

	return m
}

