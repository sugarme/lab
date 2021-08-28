package lab

import (
	"math"
	"math/rand"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// The below code taken from: https://github.com/gonum/gonum/blob/master/stat/distuv/poisson.go

// Poisson implements the Poisson distribution, a discrete probability distribution
// that expresses the probability of a given number of events occurring in a fixed
// interval.
// The poisson distribution has density function:
//  f(k) = λ^k / k! e^(-λ)
// For more information, see https://en.wikipedia.org/wiki/Poisson_distribution.
type Poisson struct {
	// Lambda is the average number of events in an interval.
	// Lambda must be greater than 0.
	Lambda float64

	Src rand.Source
}

// Rand returns a random sample drawn from the distribution.
func (p Poisson) Rand() float64 {
	// NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING (ISBN 0-521-43108-5)
	// p. 294
	// <http://www.aip.de/groups/soe/local/numres/bookcpdf/c7-3.pdf>

	rnd := rand.ExpFloat64
	var rng *rand.Rand
	if p.Src != nil {
		rng = rand.New(p.Src)
		rnd = rng.ExpFloat64
	}

	if p.Lambda < 10.0 {
		// Use direct method.
		var em float64
		t := 0.0
		for {
			t += rnd()
			if t >= p.Lambda {
				break
			}
			em++
		}
		return em
	}
	// Generate using:
	//  W. Hörmann. "The transformed rejection method for generating Poisson
	//  random variables." Insurance: Mathematics and Economics
	//  12.1 (1993): 39-45.

	// Algorithm PTRS
	rnd = rand.Float64
	if rng != nil {
		rnd = rng.Float64
	}
	b := 0.931 + 2.53*math.Sqrt(p.Lambda)
	a := -0.059 + 0.02483*b
	invalpha := 1.1239 + 1.1328/(b-3.4)
	vr := 0.9277 - 3.6224/(b-2)
	for {
		U := rnd() - 0.5
		V := rnd()
		us := 0.5 - math.Abs(U)
		k := math.Floor((2*a/us+b)*U + p.Lambda + 0.43)
		if us >= 0.07 && V <= vr {
			return k
		}
		if k <= 0 || (us < 0.013 && V > us) {
			continue
		}
		lg, _ := math.Lgamma(k + 1)
		if math.Log(V*invalpha/(a/(us*us)+b)) <= k*math.Log(p.Lambda)-p.Lambda-lg {
			return k
		}
	}
}

// Mean calculates mean of input slice.
func Mean(data []float64) float64 {
	vals := ts.MustOfSlice(data)
	mean := vals.MustMean(gotch.Float, true)
	retVal := mean.Float64Values()[0]
	mean.MustDrop()
	return retVal
}

// Median calculate median of input slice.
func Median(data []float64) float64 {
	vals := ts.MustOfSlice(data)
	median := vals.MustMedian(true)
	retVal := median.Float64Values()[0]
	median.MustDrop()
	return retVal
}
