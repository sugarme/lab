package lab

import (
	"fmt"
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
	// var nclasses int64 = 3
	// target := ts.MustOfSlice([]int64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2})
	// logits := ts.MustOfSlice([]int64{0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2})
	// Confusion matrix
	// 4  1  1
	// 6  2  2
	// 3  0  6

	var nclasses int64 = 4
	target := ts.MustOfSlice([]int64{0, 1, 2, 3})
	logits := ts.MustOfSlice([]int64{0, 2, 1, 3})

	precision, recall, f1, accuracy, err := ClassificationMetrics(logits, target, nclasses)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Printf("Precision: \t%v\n", precision)
	fmt.Printf("Recall: \t%v\n", recall)
	fmt.Printf("F1: \t\t%v\n", f1)
	fmt.Printf("Accuracy: \t%v\n", accuracy)

	t.Fatal("stop")
}

func TestMultiClassMeter(t *testing.T) {
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

	t.Fatal("stop")

}
