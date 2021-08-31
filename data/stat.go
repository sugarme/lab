package data

import (
	"log"
	"math"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// StdDev calculates standard deviation.
//
// -unbiased: If unbiased is True, Bessel’s correction will be used. Otherwise,
// the sample deviation is calculated, without any correction.
func StdDev(vals []float64, unbiased bool) float64 {
	if len(vals) == 0 {
		return math.NaN()
	}
	x := ts.MustOfSlice(vals)
	std := x.MustStd(unbiased, true)
	retVal := std.Float64Values()[0]
	std.MustDrop()

	// round to 6 digits
	return math.Floor(retVal*1000000) / 1000000
}

// Mean calculates mean of input values.
func Mean(vals []float64) float64 {
	if len(vals) == 0 {
		return math.NaN()
	}
	x := ts.MustOfSlice(vals)
	mean, err := x.Mean(gotch.Float, true)
	if err != nil {
		log.Fatal(err)
	}
	retVal := mean.Float64Values()[0]
	mean.MustDrop()

	return retVal
}

// Quantile computes the q-th quantiles of input values.
func Quantile(vals []float64, q float64) float64 {
	if len(vals) == 0 {
		return math.NaN()
	}
	x := ts.MustOfSlice(vals)
	quantile := x.MustQuantileScalar(q, []int64{0}, false, true)
	retVal := quantile.Float64Values()[0]
	quantile.MustDrop()

	return math.Floor(retVal)
}
