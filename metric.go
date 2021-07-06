package lab

import (
	"fmt"
	"log"

	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
)

type Metric interface{
	Calculate(yTrue []float64, yPred [][]float64, thresholds []float64) float64
	Name() string
}

type AUC struct{}

func(m *AUC) Calculate(yTrue []float64, yPred [][]float64, thresholds []float64) float64{
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
func(m *F1) Calculate(yTrue []float64, yPred [][]float64, thresholds []float64) float64{
	// TODO.
	panic("Not implemented")
}
func(m *F1) Name() string{
	return "f1"
}

type Accuracy struct{}
func(m *Accuracy) Calculate(yTrue []float64, yPred [][]float64, thresholds []float64) float64{
	// TODO.
	panic("Not implemented")
}
func(m *Accuracy) Name() string{
	return "f1"
}

type MelF1 struct{}
// Calculate implements Metric interface.
//
// yPred shape = (N, C)
// AUC for melanoma + nevi (class 0 + 1)
func(m *MelF1) Calculate(yTrue []float64, yPred [][]float64, thresholds []float64) float64{
	n := len(yTrue) // number of batches

	// NOTE: assuming batch size = 1!!!
	var yTrueMel []float64 = make([]float64, n) // [number_of_batches][batch_size]float64
	var yPredMel []float64 = make([]float64, n) // [number_of_batches][batch_size x number_of_classes]float64
	for i := 0; i < n; i++{
		rowT := yTrue[i] // ground truth for 1 batch. If batch size = 1, it's only 1 value
		yTrueMel[i] = rowT

		rP := yPred[i] // prediction for 1 batch. If batch size = 1, it's equal to number of classes
		var rowP float64
		if rP[0] + rP[1] >= 0.5{
			rowP = 1.0
		} else{
			rowP = 0.0
		}

		yPredMel[i] = rowP
	} 

	t := mat.NewDense(n, 1, yTrueMel)
	p := mat.NewDense(n, 1, yPredMel)
	var sampleWeight []float64
	return metrics.F1Score(t, p, "weighted", sampleWeight)
}

func(m *MelF1) Name() string{
	return "mel_f1"
}



// Flatens slice while keeping order.
func flattenSlice(vals [][]float64) []float64{
	var output []float64 = make([]float64, len(vals))
	idx := 0
	for i := 0; i < len(vals); i++{
		v := vals[i]
		for n := 0; n < len(v); n++{
			output[idx] = v[n]
		}
	}
	return output
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
	case "mel_f1":
		m = &MelF1{}
	default:
		err := fmt.Errorf("Unsupported metric %q\n", name)
		log.Fatal(err)
	}

	return m
}

