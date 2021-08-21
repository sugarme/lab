package lab

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	ts "github.com/sugarme/gotch/tensor"
)

func TestConfusionMatrix(t *testing.T) {
	var nclasses int64 = 3
	target := ts.MustOfSlice([]int64{1})
	logits := ts.MustOfSlice([]int64{1})

	mat, err := ConfusionMatrix(logits, target, nclasses)
	if err != nil {
		t.Fatal(err)
	}

	got := mat.Int64Values()
	want := []int64{
		0, 0, 0,
		0, 1, 0,
		0, 0, 0,
	}

	fmt.Printf("%v\n", mat)
	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want %v\n", want)
		t.Errorf("Got %v\n", got)
	}
}

func TestF1Meter_Calculate(t *testing.T) {
	var nclasses int = 3
	target := ts.MustOfSlice([]int64{0, 1, 2, 0, 1, 2})
	logits := ts.MustOfSlice([]int64{0, 2, 1, 0, 0, 1})

	f1meter := NewF1Meter(nclasses)
	f1 := f1meter.Calculate(logits, target)
	// Round 2 digits
	// got := math.Floor(f1*100) / 100
	got := math.Round(f1*100) / 100
	want := 0.27
	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}

func TestPrecisionMeter_Calculate(t *testing.T) {
	var nclasses int = 3
	target := ts.MustOfSlice([]int64{0, 1, 2, 0, 1, 2})
	logits := ts.MustOfSlice([]int64{0, 2, 1, 0, 0, 1})

	meter := NewPrecisionMeter(nclasses)
	val := meter.Calculate(logits, target)
	// Round 2 digits
	got := math.Round(val*100) / 100
	want := 0.22
	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}

func TestRecallMeter_Calculate(t *testing.T) {
	var nclasses int = 3
	target := ts.MustOfSlice([]int64{0, 1, 2, 0, 1, 2})
	logits := ts.MustOfSlice([]int64{0, 2, 1, 0, 0, 1})

	meter := NewRecallMeter(nclasses)
	val := meter.Calculate(logits, target)
	// Round 2 digits
	got := math.Round(val*100) / 100
	want := 0.33
	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}

func TestAccuracyMeter_Calculate(t *testing.T) {
	var nclasses int = 3
	target := ts.MustOfSlice([]int64{0, 1, 2, 0, 1, 2})
	logits := ts.MustOfSlice([]int64{0, 2, 1, 0, 0, 1})

	meter := NewAccuracyMeter(nclasses)
	val := meter.Calculate(logits, target)
	// Round 2 digits
	got := math.Round(val*100) / 100
	want := 0.33
	if !reflect.DeepEqual(want, got) {
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}

func TestMultiClassMeter_PrintStats(t *testing.T) {
	var nclasses int64 = 3
	yTrue := ts.MustOfSlice([]int64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2})
	yPred := ts.MustOfSlice([]int64{0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2})

	m, err := NewMultiClassMeter(yTrue, yPred, nclasses)
	if err != nil {
		t.Fatal(err)
	}

	err = m.PrintStats()
	if err != nil {
		t.Fatal(err)
	}

	/*
			Output:
		                      precision     recall   f1-score    support
		                   0      0.308      0.667      0.421          6
		                   1      0.667      0.200      0.308         10
		                   2      0.667      0.667      0.667          9

		            accuracy                            0.480         25
		           macro avg      0.547      0.511      0.465         25
		        weighted avg      0.581      0.480      0.464         25

	*/

}

func TestMultiClassMeter_ConfusionMatrix(t *testing.T) {
	var nclasses int64 = 3
	yTrue := ts.MustOfSlice([]int64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2})
	yPred := ts.MustOfSlice([]int64{0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2})

	m, err := NewMultiClassMeter(yTrue, yPred, nclasses)
	if err != nil {
		t.Fatal(err)
	}

	m.ConfusionMatrix()

	/*
				Output:

		                              0          1          2      total

		                   0          4          1          1          6
		                   1          6          2          2         10
		                   2          3          0          6          9

		               total         13          3          9         25
	*/

}
