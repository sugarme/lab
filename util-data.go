package lab

import (
	"sort"
)

// ClassWeights calculate weights for each class using Median Frequency Balancing method.
//
// - data: map of class name and its frequency
// - Return data map of class name and its weight.
// Ref. https://arxiv.org/abs/1411.4734
func ClassWeights(data map[string]int) map[string]float64 {
	var counts []float64
	classWeights := make(map[string]float64, len(data))
	for _, v := range data {
		counts = append(counts, float64(v))
	}

	sort.Float64s(counts)
	median := Median(counts)
	for name, count := range data {
		w := median / float64(count)
		classWeights[name] = w
	}

	return classWeights
}

func SliceInterface2Float64(vals []interface{}) []float64 {
	var retVal []float64
	for _, v := range vals {
		retVal = append(retVal, v.(float64))
	}

	return retVal
}

func SliceInterface2Int64(vals []interface{}) []int64 {
	var retVal []int64
	for _, v := range vals {
		val := v.(int)
		retVal = append(retVal, int64(val))
	}

	return retVal
}
