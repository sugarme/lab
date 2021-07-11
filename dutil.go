package lab

import (
	"sort"

	"gonum.org/v1/gonum/stat"
)

// ClassWeights calculate weights for each class using Median Frequency Balancing method.
//
// - data: map of class name and its frequency
// - Return data map of class name and its weight.
// Ref. https://arxiv.org/abs/1411.4734
func ClassWeights(data map[string]int) map[string]float64{
	var counts []float64
	classWeights := make(map[string]float64, len(data))
	for _, v := range data{
		counts = append(counts, float64(v))
	}

	sort.Float64s(counts)
	median := stat.Quantile(0.5, stat.Empirical, counts, nil)
	for name, count := range data{
		w := median/float64(count)
		classWeights[name] = w
	}

	return classWeights
}
