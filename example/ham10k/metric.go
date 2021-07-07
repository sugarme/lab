package main

import (
	"fmt"

	"github.com/pa-m/sklearn/metrics"
	"github.com/sugarme/lab"
	"gonum.org/v1/gonum/mat"
)


type DiceCoeffBatch struct{
	F1s map[int]float64 // f1 for each class
	Classes int // number of classes
	log *lab.Logger
}
// Calculate implements Metric interface.
//
// yPred: prediction pvalues for each class. [number of samples][number of classes]
// yTrue: ground truth (class id). [number of samples] 
func(m *DiceCoeffBatch) Calculate(yTrue []float64, yPred [][]float64, thresholds []float64) float64{
	classes := m.Classes
	n := len(yTrue) // number of samples

	f1s := make(map[int]float64, classes)
	// 1. Calculate f1 for each class
	for c := 0; c < classes; c++{
		var yTrueSkin []float64 = make([]float64, n)
		var yPredSkin []float64 = make([]float64, n)
		for i := 0; i < n; i++{
			// ground truth
			var rowT float64
			if yTrue[i] == float64(c){
				rowT = 1.0
			} else {
				rowT = 0.0
			}
			yTrueSkin[i] = rowT

			// predict
			var rowP float64
			if maxIdx(yPred[i]) == c{
				rowP = 1.0
			} else {
				rowP = 0.0
			}
			yPredSkin[i] = rowP
		}	
		t := mat.NewDense(n, 1, yTrueSkin)
		p := mat.NewDense(n, 1, yPredSkin)
		var sampleWeight []float64
		f1 := metrics.F1Score(t, p, "weighted", sampleWeight)
		f1s[c] = f1
		acc := metrics.AccuracyScore(t, p, true, nil)
		recall := metrics.RecallScore(t, p, "weighted", sampleWeight) // sensitivity
		msg := fmt.Sprintf("Class %2d\tF1 %0.4f\tAccuracy %0.4f\tRecall %0.4f\n", c, f1, acc, recall) 
		m.log.Printf(msg)
		m.log.SendSlack(msg)
	}

	// 2. Average of all classes' f1
	var cum float64 = 0.0
	for _, f1 := range f1s{
		cum += f1
	}

	// 3. Keep in a field
	m.F1s = f1s

	return cum/float64(len(f1s))
}

func(m *DiceCoeffBatch) Name() string{
	return "dice_coeff_batch"
}

func NewDiceCoeffBatch(logger *lab.Logger, classes int) lab.Metric{
	return &DiceCoeffBatch{
		F1s: make(map[int]float64, 0),
		Classes: classes,
		log: logger,
	}
}

func maxIdx(data []float64) int{
	var maxIdx int = -1
	max := -999.0
	for idx, val := range data{
		if val > max {
			maxIdx = idx
			max = val
		}
	}

	return maxIdx
}

