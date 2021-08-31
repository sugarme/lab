package data

import (
	"bytes"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// compareFloats compares floating point values up to the number of digits specified.
// Returns true if both values are equal with the given precision
func compareFloats(lvalue, rvalue float64, digits int) bool {
	if math.IsNaN(lvalue) || math.IsNaN(rvalue) {
		return math.IsNaN(lvalue) && math.IsNaN(rvalue)
	}
	d := math.Pow(10.0, float64(digits))
	lv := int(lvalue * d)
	rv := int(rvalue * d)
	return lv == rv
}

func TestDataFrame_NewDataframe(t *testing.T) {
	series := []Series{
		Strings([]int{1, 2, 3, 4, 5}),
		NewSeries([]int{1, 2, 3, 4, 5}, String, "0"),
		Ints([]int{1, 2, 3, 4, 5}),
		NewSeries([]int{1, 2, 3, 4, 5}, String, "0"),
		NewSeries([]int{1, 2, 3, 4, 5}, Float, "1"),
		NewSeries([]int{1, 2, 3, 4, 5}, Bool, "1"),
	}
	d := NewDataframe(series...)

	// Check that the names are renamed properly
	received := d.Names()
	expected := []string{"X0", "0_0", "X1", "0_1", "1_0", "1_1"}
	if !reflect.DeepEqual(received, expected) {
		t.Errorf(
			"Expected:\n%v\nReceived:\n%v",
			expected, received,
		)
	}
}

func TestDataFrame_Copy(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a"}, String, "COL.1"),
		NewSeries([]int{1, 2}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0}, Float, "COL.3"),
	)
	b := a.Copy()

	// Check that there are no shared memory addresses between DataFrames
	//if err := checkAddrDf(a, b); err != nil {
	//t.Error(err)
	//}
	// Check that the types are the same between both DataFrames
	if !reflect.DeepEqual(a.Types(), b.Types()) {
		t.Errorf("Different types:\nA:%v\nB:%v", a.Types(), b.Types())
	}
	// Check that the values are the same between both DataFrames
	if !reflect.DeepEqual(a.Records(), b.Records()) {
		t.Errorf("Different values:\nA:%v\nB:%v", a.Records(), b.Records())
	}
}

func TestDataFrame_Subset(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		indexes interface{}
		expDf   DataFrame
	}{
		{
			[]int{1, 2},
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "COL.1"),
				NewSeries([]int{2, 4}, Int, "COL.2"),
				NewSeries([]float64{4.0, 5.3}, Float, "COL.3"),
			),
		},
		{
			[]bool{false, true, true, false, false},
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "COL.1"),
				NewSeries([]int{2, 4}, Int, "COL.2"),
				NewSeries([]float64{4.0, 5.3}, Float, "COL.3"),
			),
		},
		{
			Ints([]int{1, 2}),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "COL.1"),
				NewSeries([]int{2, 4}, Int, "COL.2"),
				NewSeries([]float64{4.0, 5.3}, Float, "COL.3"),
			),
		},
		{
			[]int{0, 0, 1, 1, 2, 2, 3, 4},
			NewDataframe(
				NewSeries([]string{"b", "b", "a", "a", "b", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 1, 2, 2, 4, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 3.0, 4.0, 4.0, 5.3, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
	}

	for i, tc := range table {
		b := a.Subset(tc.indexes)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_Select(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		indexes interface{}
		expDf   DataFrame
	}{
		{
			Bools([]bool{false, true, true}),
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]bool{false, true, true},
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			Ints([]int{1, 2}),
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]int{1, 2},
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]int{1},
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
			),
		},
		{
			1,
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
			),
		},
		{
			[]int{1, 2, 0},
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
			),
		},
		{
			[]int{0, 0},
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
			),
		},
		{
			"COL.3",
			NewDataframe(
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]string{"COL.3"},
			NewDataframe(
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]string{"COL.3", "COL.1"},
			NewDataframe(
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
			),
		},
		{
			Strings([]string{"COL.3", "COL.1"}),
			NewDataframe(
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
			),
		},
	}

	for i, tc := range table {
		b := a.Select(tc.indexes)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_Drop(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		indexes interface{}
		expDf   DataFrame
	}{
		{
			Bools([]bool{false, true, true}),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
			),
		},
		{
			[]bool{false, true, true},
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
			),
		},
		{
			Ints([]int{1, 2}),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
			),
		},
		{
			[]int{1, 2},
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
			),
		},
		{
			[]int{1},
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			1,
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]int{0, 0},
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			"COL.3",
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
			),
		},
		{
			[]string{"COL.3"},
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
			),
		},
		{
			[]string{"COL.3", "COL.1"},
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
			),
		},
		{
			Strings([]string{"COL.3", "COL.1"}),
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
			),
		},
	}

	for i, tc := range table {
		b := a.Drop(tc.indexes)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_Rename(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		newname string
		oldname string
		expDf   DataFrame
	}{
		{
			"NEWCOL.1",
			"COL.1",
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "NEWCOL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			"NEWCOL.3",
			"COL.3",
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "NEWCOL.3"),
			),
		},
		{
			"NEWCOL.2",
			"COL.2",
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "NEWCOL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
	}
	for i, tc := range table {
		b := a.Rename(tc.newname, tc.oldname)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_CBind(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		dfb   DataFrame
		expDf DataFrame
	}{
		{
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.4"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.5"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.6"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.4"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.5"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.6"),
			),
		},
		{
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.4"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.4"),
			),
		},
		{
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.4"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.6"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.4"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.6"),
			)},
	}
	for i, tc := range table {
		b := a.CBind(tc.dfb)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_RBind(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		dfb   DataFrame
		expDf DataFrame
	}{
		{
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d", "b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4, 1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2, 3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d", "1", "2", "4", "5", "4"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4, 1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2, 3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
	}
	for i, tc := range table {
		b := a.RBind(tc.dfb)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_Concat(t *testing.T) {
	type NA struct{}

	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		dfa   DataFrame
		dfb   DataFrame
		expDf DataFrame
	}{
		{
			a,
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d", "b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4, 1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2, 3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			a,
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d", "1", "2", "4", "5", "4"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4, 1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2, 3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},

		{
			a,
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d", "b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2").Concat(NewSeries([]NA{NA{}, NA{}, NA{}, NA{}, NA{}}, Int, "")),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2, 3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			a,
			NewDataframe(
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]string{"a", "b", "c", "d", "e"}, String, "COL.4"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d", "1", "2", "4", "5", "4"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4, 1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2, 3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]NA{NA{}, NA{}, NA{}, NA{}, NA{}}, String, "COL.4").Concat(NewSeries([]string{"a", "b", "c", "d", "e"}, String, "COL.4")),
			),
		},
		{
			a,
			NewDataframe(
				NewSeries([]string{"a", "b", "c", "d", "e"}, String, "COL.0"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d", "1", "2", "4", "5", "4"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4, 1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2, 3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]NA{NA{}, NA{}, NA{}, NA{}, NA{}}, String, "COL.0").Concat(NewSeries([]string{"a", "b", "c", "d", "e"}, String, "COL.0")),
			),
		},
		{
			DataFrame{},
			a,
			a,
		},
	}
	for i, tc := range table {
		b := tc.dfa.Concat(tc.dfb)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}
func TestDataFrame_Records(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"a", "b", "c"}, String, "COL.1"),
		NewSeries([]int{1, 2, 3}, Int, "COL.2"),
		NewSeries([]float64{3, 2, 1}, Float, "COL.3"))
	expected := [][]string{
		{"COL.1", "COL.2", "COL.3"},
		{"a", "1", "3.000000"},
		{"b", "2", "2.000000"},
		{"c", "3", "1.000000"},
	}
	received := a.Records()
	if !reflect.DeepEqual(expected, received) {
		t.Error(
			"Error when saving records.\n",
			"Expected: ", expected, "\n",
			"Received: ", received,
		)
	}
}

func TestDataFrame_Mutate(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		s     Series
		expDf DataFrame
	}{
		{
			NewSeries([]string{"A", "B", "A", "A", "A"}, String, "COL.1"),
			NewDataframe(
				NewSeries([]string{"A", "B", "A", "A", "A"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			NewSeries([]string{"A", "B", "A", "A", "A"}, String, "COL.2"),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]string{"A", "B", "A", "A", "A"}, String, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			NewSeries([]string{"A", "B", "A", "A", "A"}, String, "COL.4"),
			NewDataframe(
				NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
				NewSeries([]string{"A", "B", "A", "A", "A"}, String, "COL.4"),
			),
		},
	}
	for i, tc := range table {
		b := a.Mutate(tc.s)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_Filter_Or(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		filters []F
		expDf   DataFrame
	}{
		{
			[]F{{0, "COL.2", GreaterEq, 4}},
			NewDataframe(
				NewSeries([]string{"b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]F{
				{0, "COL.2", Greater, 4},
				{0, "COL.2", Eq, 1},
			},
			NewDataframe(
				NewSeries([]string{"b", "c"}, String, "COL.1"),
				NewSeries([]int{1, 5}, Int, "COL.2"),
				NewSeries([]float64{3.0, 3.2}, Float, "COL.3"),
			),
		},
		{
			[]F{
				{0, "COL.2", Greater, 4},
				{0, "COL.2", Eq, 1},
				{0, "COL.1", Eq, "d"},
			},
			NewDataframe(
				NewSeries([]string{"b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]F{
				{1, "", Greater, 4},
				{1, "", Eq, 1},
				{0, "", Eq, "d"},
			},
			NewDataframe(
				NewSeries([]string{"b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{1, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{3.0, 3.2, 1.2}, Float, "COL.3"),
			),
		},
	}
	for i, tc := range table {
		b := a.Filter(tc.filters...)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}

		b2 := a.FilterAggregation(Or, tc.filters...)

		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(b.Types(), b2.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nB:%v\nB2:%v", i, b.Types(), b2.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(b.Names(), b2.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nB:%v\nB2:%v", i, b.Names(), b2.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(b.Records(), b2.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nB:%v\nB2:%v", i, b.Records(), b2.Records())
		}
	}
}

func TestDataFrame_Filter_And(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "c", "d"}, String, "COL.1"),
		NewSeries([]int{1, 2, 4, 5, 4}, Int, "COL.2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "COL.3"),
	)
	table := []struct {
		filters []F
		expDf   DataFrame
	}{
		{
			[]F{{Colname: "COL.2", Comparator: GreaterEq, Comparando: 4}},
			NewDataframe(
				NewSeries([]string{"b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		{
			[]F{{Colidx: 1, Comparator: GreaterEq, Comparando: 4}},
			NewDataframe(
				NewSeries([]string{"b", "c", "d"}, String, "COL.1"),
				NewSeries([]int{4, 5, 4}, Int, "COL.2"),
				NewSeries([]float64{5.3, 3.2, 1.2}, Float, "COL.3"),
			),
		},
		// should not have any rows
		{
			[]F{
				{Colname: "COL.2", Comparator: Greater, Comparando: 4},
				{Colname: "COL.2", Comparator: Eq, Comparando: 1},
			},
			NewDataframe(
				NewSeries([]string{}, String, "COL.1"),
				NewSeries([]int{}, Int, "COL.2"),
				NewSeries([]float64{}, Float, "COL.3"),
			),
		},
		{
			[]F{
				{Colidx: 1, Comparator: Greater, Comparando: 4},
				{Colidx: 1, Comparator: Eq, Comparando: 1},
			},
			NewDataframe(
				NewSeries([]string{}, String, "COL.1"),
				NewSeries([]int{}, Int, "COL.2"),
				NewSeries([]float64{}, Float, "COL.3"),
			),
		},
		{
			[]F{
				{Colname: "COL.2", Comparator: Less, Comparando: 4},
				{Colname: "COL.1", Comparator: Eq, Comparando: "b"},
			},
			NewDataframe(
				NewSeries([]string{"b"}, String, "COL.1"),
				NewSeries([]int{1}, Int, "COL.2"),
				NewSeries([]float64{3.0}, Float, "COL.3"),
			),
		},
		{
			[]F{
				{Colidx: 1, Comparator: Less, Comparando: 4},
				{Colidx: 0, Comparator: Eq, Comparando: "b"},
			},
			NewDataframe(
				NewSeries([]string{"b"}, String, "COL.1"),
				NewSeries([]int{1}, Int, "COL.2"),
				NewSeries([]float64{3.0}, Float, "COL.3"),
			),
		},
	}
	for i, tc := range table {
		b := a.FilterAggregation(And, tc.filters...)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestLoadRecords(t *testing.T) {
	table := []struct {
		df    DataFrame
		expDf DataFrame
		err   bool
	}{
		{ // Test: 0
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]float64{0, 0.5}, Float, "D"),
			),
			false,
		},
		{ // Test: 1
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
				HasHeader(true),
				DetectTypes(false),
				DefaultType(String),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]int{1, 2}, String, "B"),
				NewSeries([]bool{true, true}, String, "C"),
				NewSeries([]string{"0", "0.5"}, String, "D"),
			),
			false,
		},
		{ // Test: 2
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
				HasHeader(false),
				DetectTypes(false),
				DefaultType(String),
			),
			NewDataframe(
				NewSeries([]string{"A", "a", "b"}, String, "X0"),
				NewSeries([]string{"B", "1", "2"}, String, "X1"),
				NewSeries([]string{"C", "true", "true"}, String, "X2"),
				NewSeries([]string{"D", "0", "0.5"}, String, "X3"),
			),
			false,
		},
		{ // Test: 3
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
				HasHeader(true),
				DetectTypes(false),
				DefaultType(String),
				WithTypes(map[string]Type{
					"B": Float,
					"C": String,
				}),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]float64{1, 2}, Float, "B"),
				NewSeries([]bool{true, true}, String, "C"),
				NewSeries([]string{"0", "0.5"}, String, "D"),
			),
			false,
		},
		{ // Test: 4
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
				HasHeader(true),
				DetectTypes(true),
				DefaultType(String),
				WithTypes(map[string]Type{
					"B": Float,
				}),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]float64{1, 2}, Float, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]string{"0", "0.5"}, Float, "D"),
			),
			false,
		},
		{ // Test: 5
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
				HasHeader(true),
				Names("MyA", "MyB", "MyC", "MyD"),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "MyA"),
				NewSeries([]int{1, 2}, Int, "MyB"),
				NewSeries([]bool{true, true}, Bool, "MyC"),
				NewSeries([]string{"0", "0.5"}, Float, "MyD"),
			),
			false,
		},
		{ // Test: 6
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
				HasHeader(false),
				Names("MyA", "MyB", "MyC", "MyD"),
			),
			NewDataframe(
				NewSeries([]string{"A", "a", "b"}, String, "MyA"),
				NewSeries([]string{"B", "1", "2"}, String, "MyB"),
				NewSeries([]string{"C", "true", "true"}, String, "MyC"),
				NewSeries([]string{"D", "0", "0.5"}, String, "MyD"),
			),
			false,
		},
		{ // Test: 7
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
				HasHeader(false),
				Names("MyA", "MyB", "MyC"),
			),
			DataFrame{},
			true,
		},
		{ // Test: 8
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"b", "2", "true", "0.5"},
				},
				HasHeader(false),
				Names("MyA", "MyB", "MyC", "MyD", "MyE"),
			),
			DataFrame{},
			true,
		},
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"1", "1", "true", "0"},
					{"a", "2", "true", "0.5"},
				},
			),
			NewDataframe(
				NewSeries([]string{"1", "a"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]float64{0, 0.5}, Float, "D"),
			),
			false,
		},
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0"},
					{"1", "2", "true", "0.5"},
				},
			),
			NewDataframe(
				NewSeries([]string{"a", "1"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]float64{0, 0.5}, Float, "D"),
			),
			false,
		},
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0.5"},
					{"1", "2", "true", "1"},
				},
			),
			NewDataframe(
				NewSeries([]string{"a", "1"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]float64{0.5, 1}, Float, "D"),
			),
			false,
		},
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "trueee", "0.5"},
					{"1", "2", "true", "1"},
				},
			),
			NewDataframe(
				NewSeries([]string{"a", "1"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]string{"trueee", "true"}, String, "C"),
				NewSeries([]float64{0.5, 1}, Float, "D"),
			),
			false,
		},
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0.5"},
					{"1", "2", "trueee", "1"},
				},
			),
			NewDataframe(
				NewSeries([]string{"a", "1"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]string{"true", "trueee"}, String, "C"),
				NewSeries([]float64{0.5, 1}, Float, "D"),
			),
			false,
		},
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0.5"},
					{"1", "2", "true", "a"},
				},
			),
			NewDataframe(
				NewSeries([]string{"a", "1"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]string{"0.5", "a"}, String, "D"),
			),
			false,
		},
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "1", "true", "0.5"},
					{"1", "2", "0.5", "a"},
				},
			),
			NewDataframe(
				NewSeries([]string{"a", "1"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]string{"true", "NaN"}, Bool, "C"),
				NewSeries([]string{"0.5", "a"}, String, "D"),
			),
			false,
		},
	}

	for i, tc := range table {
		if tc.err != (tc.df.Err != nil) {
			t.Errorf("Test: %d\nError: %v", i, tc.df.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), tc.df.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), tc.df.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), tc.df.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), tc.df.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), tc.df.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), tc.df.Records())
		}
	}
}

func TestLoadMaps(t *testing.T) {
	table := []struct {
		df    DataFrame
		expDf DataFrame
	}{
		{ // Test: 0
			LoadMaps(
				[]map[string]interface{}{
					{
						"A": "a",
						"B": 1,
						"C": true,
						"D": 0,
					},
					{
						"A": "b",
						"B": 2,
						"C": true,
						"D": 0.5,
					},
				},
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]float64{0, 0.5}, Float, "D"),
			),
		},
		{ // Test: 1
			LoadMaps(
				[]map[string]interface{}{
					{
						"A": "a",
						"B": 1,
						"C": true,
						"D": 0,
					},
					{
						"A": "b",
						"B": 2,
						"C": true,
						"D": 0.5,
					},
				},
				HasHeader(true),
				DetectTypes(false),
				DefaultType(String),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]int{1, 2}, String, "B"),
				NewSeries([]bool{true, true}, String, "C"),
				NewSeries([]string{"0", "0.5"}, String, "D"),
			),
		},
		{ // Test: 2
			LoadMaps(
				[]map[string]interface{}{
					{
						"A": "a",
						"B": 1,
						"C": true,
						"D": 0,
					},
					{
						"A": "b",
						"B": 2,
						"C": true,
						"D": 0.5,
					},
				},
				HasHeader(false),
				DetectTypes(false),
				DefaultType(String),
			),
			NewDataframe(
				NewSeries([]string{"A", "a", "b"}, String, "X0"),
				NewSeries([]string{"B", "1", "2"}, String, "X1"),
				NewSeries([]string{"C", "true", "true"}, String, "X2"),
				NewSeries([]string{"D", "0", "0.5"}, String, "X3"),
			),
		},
		{ // Test: 3
			LoadMaps(
				[]map[string]interface{}{
					{
						"A": "a",
						"B": 1,
						"C": true,
						"D": 0,
					},
					{
						"A": "b",
						"B": 2,
						"C": true,
						"D": 0.5,
					},
				},
				HasHeader(true),
				DetectTypes(false),
				DefaultType(String),
				WithTypes(map[string]Type{
					"B": Float,
					"C": String,
				}),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]float64{1, 2}, Float, "B"),
				NewSeries([]bool{true, true}, String, "C"),
				NewSeries([]string{"0", "0.5"}, String, "D"),
			),
		},
		{ // Test: 4
			LoadMaps(
				[]map[string]interface{}{
					{
						"A": "a",
						"B": 1,
						"C": true,
						"D": 0,
					},
					{
						"A": "b",
						"B": 2,
						"C": true,
						"D": 0.5,
					},
				},
				HasHeader(true),
				DetectTypes(true),
				DefaultType(String),
				WithTypes(map[string]Type{
					"B": Float,
				}),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]float64{1, 2}, Float, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]string{"0", "0.5"}, Float, "D"),
			),
		},
	}

	for i, tc := range table {
		if tc.df.Err != nil {
			t.Errorf("Test: %d\nError: %v", i, tc.df.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), tc.df.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), tc.df.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), tc.df.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), tc.df.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), tc.df.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), tc.df.Records())
		}
	}
}

func TestReadCSV(t *testing.T) {
	// Load the data from a CSV string and try to infer the type of the
	// columns
	csvStr := `
Country,Date,Age,Amount,Id
"United States",2012-02-01,50,112.1,01234
"United States",2012-02-01,32,321.31,54320
"United Kingdom",2012-02-01,17,18.2,12345
"United States",2012-02-01,32,321.31,54320
"United Kingdom",2012-02-01,NA,18.2,12345
"United States",2012-02-01,32,321.31,54320
"United States",2012-02-01,32,321.31,54320
Spain,2012-02-01,66,555.42,00241
`
	a := ReadCSV(strings.NewReader(csvStr))
	if a.Err != nil {
		t.Errorf("Expected success, got error: %v", a.Err)
	}
}

func TestReadJSON(t *testing.T) {
	table := []struct {
		jsonStr string
		expDf   DataFrame
	}{
		{
			`[{"COL.1":null,"COL.2":1,"COL.3":3},{"COL.1":5,"COL.2":2,"COL.3":2},{"COL.1":6,"COL.2":3,"COL.3":20180428}]`,
			LoadRecords(
				[][]string{
					{"COL.1", "COL.2", "COL.3"},
					{"NaN", "1", "3"},
					{"5", "2", "2"},
					{"6", "3", "20180428"},
				},
				DetectTypes(false),
				DefaultType(Int),
			),
		},
		{
			`[{"COL.2":1,"COL.3":3},{"COL.1":5,"COL.2":2,"COL.3":2},{"COL.1":6,"COL.2":3,"COL.3":1}]`,
			LoadRecords(
				[][]string{
					{"COL.1", "COL.2", "COL.3"},
					{"NaN", "1", "3"},
					{"5", "2", "2"},
					{"6", "3", "1"},
				},
				DetectTypes(false),
				DefaultType(Int),
			),
		},
	}
	for i, tc := range table {
		c := ReadJSON(strings.NewReader(tc.jsonStr))

		if c.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, c.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), c.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), c.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), c.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), c.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), c.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), c.Records())
		}
	}
}

func TestReadHTML(t *testing.T) {
	table := []struct {
		htmlStr string
		expDf   []DataFrame
	}{
		{
			"",
			[]DataFrame{},
		},
		{
			`<html>
			<body>
			<table>
			<tr><td>COL.1</td></tr>
			<tr><td>100</td></tr>
			</table>
			</body>
			</html>`,
			[]DataFrame{
				LoadRecords(
					[][]string{
						{"COL.1"},
						{"100"},
					}),
			},
		},
		{
			`<html>
			<body>
			<table>
			<tr><td rowspan='2'>COL.1</td><td rowspan='2'>COL.2</td><td>COL.3</td></tr>
			<tr><td>100</td></tr>
			</table>
			</body>
			</html>`,
			[]DataFrame{
				LoadRecords(
					[][]string{
						{"COL.1", "COL.2", "COL.3"},
						{"COL.1", "COL.2", "100"},
					}),
			},
		},
	}

	for i, tc := range table {
		cs := ReadHTML(strings.NewReader(tc.htmlStr))
		if tc.htmlStr != "" && len(cs) == 0 {
			t.Errorf("Test: %d, got zero dataframes: %#v", i, cs)
		}
		for j, c := range cs {
			if len(cs) != len(tc.expDf) {
				t.Errorf("Test: %d\n got len(%d), want len(%d)", i, len(cs), len(tc.expDf))
			}
			if c.Err != nil {
				t.Errorf("Test: %d\nError:%v", i, c.Err)
			}
			// Check that the types are the same between both DataFrames
			if !reflect.DeepEqual(tc.expDf[j].Types(), c.Types()) {
				t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf[j].Types(), c.Types())
			}
			// Check that the colnames are the same between both DataFrames
			if !reflect.DeepEqual(tc.expDf[j].Names(), c.Names()) {
				t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf[j].Names(), c.Names())
			}
			// Check that the values are the same between both DataFrames
			if !reflect.DeepEqual(tc.expDf[j].Records(), c.Records()) {
				t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf[j].Records(), c.Records())
			}
		}
	}
}

func TestDataFrame_SetNames(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"a", "b", "c"}, String, "COL.1"),
		NewSeries([]int{1, 2, 3}, Int, "COL.2"),
		NewSeries([]float64{3, 2, 1}, Float, "COL.3"),
	)

	err := a.SetNames("wot", "tho", "tree")
	if err != nil {
		t.Error("Expected success, got error")
	}
	err = a.SetNames("yaaa")
	if err == nil {
		t.Error("Expected error, got success")
	}
}

func TestDataFrame_InnerJoin(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"1", "a", "5.1", "true"},
			{"2", "b", "6.0", "true"},
			{"3", "c", "6.0", "false"},
			{"1", "d", "7.1", "false"},
		},
	)
	b := LoadRecords(
		[][]string{
			{"A", "F", "D"},
			{"1", "1", "true"},
			{"4", "2", "false"},
			{"2", "8", "false"},
			{"5", "9", "false"},
		},
	)
	table := []struct {
		keys  []string
		expDf DataFrame
	}{
		{
			[]string{"A", "D"},
			LoadRecords(
				[][]string{
					{"A", "D", "B", "C", "F"},
					{"1", "true", "a", "5.1", "1"},
				},
			),
		},
		{
			[]string{"A"},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D_0", "F", "D_1"},
					{"1", "a", "5.1", "true", "1", "true"},
					{"2", "b", "6.0", "true", "8", "false"},
					{"1", "d", "7.1", "false", "1", "true"},
				},
			),
		},
		{
			[]string{"D"},
			LoadRecords(
				[][]string{
					{"D", "A_0", "B", "C", "A_1", "F"},
					{"true", "1", "a", "5.1", "1", "1"},
					{"true", "2", "b", "6.0", "1", "1"},
					{"false", "3", "c", "6.0", "4", "2"},
					{"false", "3", "c", "6.0", "2", "8"},
					{"false", "3", "c", "6.0", "5", "9"},
					{"false", "1", "d", "7.1", "4", "2"},
					{"false", "1", "d", "7.1", "2", "8"},
					{"false", "1", "d", "7.1", "5", "9"},
				},
			),
		},
	}
	for i, tc := range table {
		c := a.InnerJoin(b, tc.keys...)

		if err := c.Err; err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), c.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), c.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), c.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), c.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), c.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), c.Records())
		}
	}
}

func TestDataFrame_LeftJoin(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"1", "4", "5.1", "1"},
			{"2", "4", "6.0", "1"},
			{"3", "3", "6.0", "0"},
			{"1", "2", "7.1", "0"},
		},
		DetectTypes(false),
		DefaultType(Float),
	)
	b := LoadRecords(
		[][]string{
			{"A", "F", "D"},
			{"1", "1", "1"},
			{"4", "2", "0"},
			{"2", "8", "0"},
			{"5", "9", "0"},
		},
		DetectTypes(false),
		DefaultType(Float),
	)
	table := []struct {
		keys  []string
		expDf DataFrame
	}{
		{
			[]string{"A", "D"},
			LoadRecords(
				[][]string{
					{"A", "D", "B", "C", "F"},
					{"1", "1", "4", "5.1", "1"},
					{"2", "1", "4", "6.0", "NaN"},
					{"3", "0", "3", "6.0", "NaN"},
					{"1", "0", "2", "7.1", "NaN"},
				},
				DetectTypes(false),
				DefaultType(Float),
			),
		},
		{
			[]string{"A"},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D_0", "F", "D_1"},
					{"1", "4", "5.1", "1", "1", "1"},
					{"2", "4", "6.0", "1", "8", "0"},
					{"3", "3", "6.0", "0", "NaN", "NaN"},
					{"1", "2", "7.1", "0", "1", "1"},
				},
				DetectTypes(false),
				DefaultType(Float),
			),
		},
	}
	for i, tc := range table {
		c := a.LeftJoin(b, tc.keys...)

		if err := c.Err; err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), c.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), c.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), c.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), c.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), c.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), c.Records())
		}
	}
}

func TestDataFrame_RightJoin(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "F", "D"},
			{"1", "1", "1"},
			{"4", "2", "0"},
			{"2", "8", "0"},
			{"5", "9", "0"},
		},
		DetectTypes(false),
		DefaultType(Float),
	)
	b := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"1", "4", "5.1", "1"},
			{"2", "4", "6.0", "1"},
			{"3", "3", "6.0", "0"},
			{"1", "2", "7.1", "0"},
		},
		DetectTypes(false),
		DefaultType(Float),
	)
	table := []struct {
		keys  []string
		expDf DataFrame
	}{
		{
			[]string{"A", "D"},
			LoadRecords(
				[][]string{
					{"A", "D", "F", "B", "C"},
					{"1", "1", "1", "4", "5.1"},
					{"2", "1", "NaN", "4", "6.0"},
					{"3", "0", "NaN", "3", "6.0"},
					{"1", "0", "NaN", "2", "7.1"},
				},
				DetectTypes(false),
				DefaultType(Float),
			),
		},
		{
			[]string{"A"},
			LoadRecords(
				[][]string{
					{"A", "F", "D_0", "B", "C", "D_1"},
					{"1", "1", "1", "4", "5.1", "1"},
					{"2", "8", "0", "4", "6.0", "1"},
					{"1", "1", "1", "2", "7.1", "0"},
					{"3", "NaN", "NaN", "3", "6.0", "0"},
				},
				DetectTypes(false),
				DefaultType(Float),
			),
		},
	}
	for i, tc := range table {
		c := a.RightJoin(b, tc.keys...)

		if err := c.Err; err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), c.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), c.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), c.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), c.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), c.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), c.Records())
		}
	}
}

func TestDataFrame_OuterJoin(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"1", "4", "5.1", "1"},
			{"2", "4", "6.0", "1"},
			{"3", "3", "6.0", "0"},
			{"1", "2", "7.1", "0"},
		},
		DetectTypes(false),
		DefaultType(Float),
	)
	b := LoadRecords(
		[][]string{
			{"A", "F", "D"},
			{"1", "1", "1"},
			{"4", "2", "0"},
			{"2", "8", "0"},
			{"5", "9", "0"},
		},
		DetectTypes(false),
		DefaultType(Float),
	)
	table := []struct {
		keys  []string
		expDf DataFrame
	}{
		{
			[]string{"A", "D"},
			LoadRecords(
				[][]string{
					{"A", "D", "B", "C", "F"},
					{"1", "1", "4", "5.1", "1"},
					{"2", "1", "4", "6.0", "NaN"},
					{"3", "0", "3", "6.0", "NaN"},
					{"1", "0", "2", "7.1", "NaN"},
					{"4", "0", "NaN", "NaN", "2"},
					{"2", "0", "NaN", "NaN", "8"},
					{"5", "0", "NaN", "NaN", "9"},
				},
				DetectTypes(false),
				DefaultType(Float),
			),
		},
		{
			[]string{"A"},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D_0", "F", "D_1"},
					{"1", "4", "5.1", "1", "1", "1"},
					{"2", "4", "6.0", "1", "8", "0"},
					{"3", "3", "6.0", "0", "NaN", "NaN"},
					{"1", "2", "7.1", "0", "1", "1"},
					{"4", "NaN", "NaN", "NaN", "2", "0"},
					{"5", "NaN", "NaN", "NaN", "9", "0"},
				},
				DetectTypes(false),
				DefaultType(Float),
			),
		},
	}
	for i, tc := range table {
		c := a.OuterJoin(b, tc.keys...)

		if err := c.Err; err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), c.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), c.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), c.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), c.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), c.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), c.Records())
		}
	}
}

func TestDataFrame_CrossJoin(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"1", "a", "5.1", "true"},
			{"2", "b", "6.0", "true"},
			{"3", "c", "6.0", "false"},
			{"1", "d", "7.1", "false"},
		},
	)
	b := LoadRecords(
		[][]string{
			{"A", "F", "D"},
			{"1", "1", "true"},
			{"4", "2", "false"},
			{"2", "8", "false"},
			{"5", "9", "false"},
		},
	)
	c := a.CrossJoin(b)
	expectedCSV := `
A_0,B,C,D_0,A_1,F,D_1
1,a,5.1,true,1,1,true
1,a,5.1,true,4,2,false
1,a,5.1,true,2,8,false
1,a,5.1,true,5,9,false
2,b,6.0,true,1,1,true
2,b,6.0,true,4,2,false
2,b,6.0,true,2,8,false
2,b,6.0,true,5,9,false
3,c,6.0,false,1,1,true
3,c,6.0,false,4,2,false
3,c,6.0,false,2,8,false
3,c,6.0,false,5,9,false
1,d,7.1,false,1,1,true
1,d,7.1,false,4,2,false
1,d,7.1,false,2,8,false
1,d,7.1,false,5,9,false
`
	expected := ReadCSV(
		strings.NewReader(expectedCSV),
		WithTypes(map[string]Type{
			"A.1": String,
		}))
	if err := c.Err; err != nil {
		t.Errorf("Error:%v", err)
	}
	// Check that the types are the same between both DataFrames
	if !reflect.DeepEqual(expected.Types(), c.Types()) {
		t.Errorf("Different types:\nA:%v\nB:%v", expected.Types(), c.Types())
	}
	// Check that the colnames are the same between both DataFrames
	if !reflect.DeepEqual(expected.Names(), c.Names()) {
		t.Errorf("Different colnames:\nA:%v\nB:%v", expected.Names(), c.Names())
	}
	// Check that the values are the same between both DataFrames
	if !reflect.DeepEqual(expected.Records(), c.Records()) {
		t.Errorf("Different values:\nA:%v\nB:%v", expected.Records(), c.Records())
	}
}

func TestDataFrame_Maps(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"a", "b", "c"}, String, "COL.1"),
		NewSeries([]string{"", "2", "3"}, Int, "COL.2"),
		NewSeries([]string{"", "", "3"}, Int, "COL.3"),
	)
	m := a.Maps()
	expected := []map[string]interface{}{
		{
			"COL.1": "a",
			"COL.2": nil,
			"COL.3": nil,
		},
		{
			"COL.1": "b",
			"COL.2": 2,
			"COL.3": nil,
		},
		{
			"COL.1": "c",
			"COL.2": 3,
			"COL.3": 3,
		},
	}
	if !reflect.DeepEqual(expected, m) {
		t.Errorf("Different values:\nA:%v\nB:%v", expected, m)
	}
}

func TestDataFrame_WriteCSV(t *testing.T) {
	table := []struct {
		df       DataFrame
		options  []WriteOption
		expected string
	}{
		{ // Test: 0
			LoadRecords(
				[][]string{
					{"COL.1", "COL.2", "COL.3"},
					{"NaN", "1", "3"},
					{"b", "2", "2"},
					{"c", "3", "1"},
				},
			),
			nil,
			`COL.1,COL.2,COL.3
NaN,1,3
b,2,2
c,3,1
`,
		},
		{ // Test: 1
			LoadRecords(
				[][]string{
					{"COL.1", "COL.2", "COL.3"},
					{"NaN", "1", "3"},
					{"b", "2", "2"},
					{"c", "3", "1"},
				},
			),
			nil,
			`COL.1,COL.2,COL.3
NaN,1,3
b,2,2
c,3,1
`,
		},
		{ // Test: 2
			LoadRecords(
				[][]string{
					{"COL.1", "COL.2", "COL.3"},
					{"NaN", "1", "3"},
					{"b", "2", "2"},
					{"c", "3", "1"},
				},
			),
			[]WriteOption{WriteHeader(false)},
			`NaN,1,3
b,2,2
c,3,1
`,
		},
	}

	for i, tc := range table {
		buf := new(bytes.Buffer)
		err := tc.df.WriteCSV(buf, tc.options...)
		if err != nil {
			t.Errorf("Test: %d\nError: %v", i, err)
		}
		if tc.expected != buf.String() {
			t.Errorf("Test: %d\nExpected: %v\nreceived: %v", i, tc.expected, buf.String())
		}
	}
}

func TestDataFrame_WriteJSON(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"COL.1", "COL.2", "COL.3"},
			{"NaN", "1", "3"},
			{"5", "2", "2"},
			{"6", "3", "1"},
		},
		DetectTypes(false),
		DefaultType(Int),
	)
	buf := new(bytes.Buffer)
	err := a.WriteJSON(buf)
	if err != nil {
		t.Errorf("Expected success, got error: %v", err)
	}
	expected := `[{"COL.1":null,"COL.2":1,"COL.3":3},{"COL.1":5,"COL.2":2,"COL.3":2},{"COL.1":6,"COL.2":3,"COL.3":1}]
`
	if expected != buf.String() {
		t.Errorf("\nexpected: %v\nreceived: %v", expected, buf.String())
	}
}

func TestDataFrame_Col(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"COL.1", "COL.2", "COL.3"},
			{"NaN", "1", "3"},
			{"5", "2", "2"},
			{"6", "3", "1"},
		},
		DetectTypes(false),
		DefaultType(Int),
	)
	b := a.Col("COL.2")
	expected := NewSeries([]int{1, 2, 3}, Int, "COL.2")
	if !reflect.DeepEqual(b.Records(), expected.Records()) {
		t.Errorf("\nexpected: %v\nreceived: %v", expected, b)
	}
}

func TestDataFrame_Set(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"a", "4", "5.1", "true"},
			{"b", "4", "6.0", "true"},
			{"c", "3", "6.0", "false"},
			{"a", "2", "7.1", "false"},
		},
	)
	table := []struct {
		indexes   Indexes
		newvalues DataFrame
		expDf     DataFrame
	}{
		{
			Ints([]int{0, 2}),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"k", "4", "6.0", "true"},
				},
			),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"b", "4", "6.0", "true"},
					{"k", "4", "6.0", "true"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			Ints(0),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
				},
			),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			Bools([]bool{true, false, false, false}),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
				},
			),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			Bools([]bool{false, true, true, false}),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"k", "4", "6.0", "true"},
				},
			),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "4", "5.1", "true"},
					{"k", "5", "7.0", "true"},
					{"k", "4", "6.0", "true"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			[]int{0, 2},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"k", "4", "6.0", "true"},
				},
			),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"b", "4", "6.0", "true"},
					{"k", "4", "6.0", "true"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			0,
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
				},
			),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			[]bool{true, false, false, false},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
				},
			),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			[]bool{false, true, true, false},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"k", "5", "7.0", "true"},
					{"k", "4", "6.0", "true"},
				},
			),
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "4", "5.1", "true"},
					{"k", "5", "7.0", "true"},
					{"k", "4", "6.0", "true"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
	}
	for i, tc := range table {
		a := a.Copy()
		b := a.Set(tc.indexes, tc.newvalues)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_Arrange(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"a", "4", "5.1", "true"},
			{"b", "4", "6.0", "true"},
			{"c", "3", "6.0", "false"},
			{"a", "2", "7.1", "false"},
		},
	)
	table := []struct {
		colnames []Order
		expDf    DataFrame
	}{
		{
			[]Order{Sort("A")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "4", "5.1", "true"},
					{"a", "2", "7.1", "false"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
				},
			),
		},
		{
			[]Order{Sort("B")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "2", "7.1", "false"},
					{"c", "3", "6.0", "false"},
					{"a", "4", "5.1", "true"},
					{"b", "4", "6.0", "true"},
				},
			),
		},
		{
			[]Order{Sort("A"), Sort("B")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "2", "7.1", "false"},
					{"a", "4", "5.1", "true"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
				},
			),
		},
		{
			[]Order{Sort("B"), Sort("A")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "2", "7.1", "false"},
					{"c", "3", "6.0", "false"},
					{"a", "4", "5.1", "true"},
					{"b", "4", "6.0", "true"},
				},
			),
		},
		{
			[]Order{RevSort("A")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"c", "3", "6.0", "false"},
					{"b", "4", "6.0", "true"},
					{"a", "4", "5.1", "true"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			[]Order{RevSort("B")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "4", "5.1", "true"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			[]Order{Sort("A"), RevSort("B")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "4", "5.1", "true"},
					{"a", "2", "7.1", "false"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
				},
			),
		},
		{
			[]Order{Sort("B"), RevSort("A")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "2", "7.1", "false"},
					{"c", "3", "6.0", "false"},
					{"b", "4", "6.0", "true"},
					{"a", "4", "5.1", "true"},
				},
			),
		},
		{
			[]Order{RevSort("B"), RevSort("A")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"b", "4", "6.0", "true"},
					{"a", "4", "5.1", "true"},
					{"c", "3", "6.0", "false"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
		{
			[]Order{RevSort("A"), RevSort("B")},
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"c", "3", "6.0", "false"},
					{"b", "4", "6.0", "true"},
					{"a", "4", "5.1", "true"},
					{"a", "2", "7.1", "false"},
				},
			),
		},
	}
	for i, tc := range table {
		b := a.Arrange(tc.colnames...)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_Capply(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"a", "4", "5.1", "true"},
			{"b", "4", "6.0", "true"},
			{"c", "3", "6.0", "false"},
			{"a", "2", "7.1", "false"},
		},
	)
	mean := func(s Series) Series {
		floats := s.Float()
		sum := 0.0
		for _, f := range floats {
			sum += f
		}
		return Floats(sum / float64(len(floats)))
	}
	sum := func(s Series) Series {
		floats := s.Float()
		sum := 0.0
		for _, f := range floats {
			sum += f
		}
		return Floats(sum)
	}
	table := []struct {
		fun   func(Series) Series
		expDf DataFrame
	}{
		{
			mean,
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"NaN", "3.25", "6.05", "0.5"},
				},
				DefaultType(Float),
				DetectTypes(false),
			),
		},
		{
			sum,
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"NaN", "13", "24.2", "2"},
				},
				DefaultType(Float),
				DetectTypes(false),
			),
		},
	}
	for i, tc := range table {
		b := a.Capply(tc.fun)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestDataFrame_String(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "C", "D"},
			{"1", "5.1", "true"},
			{"NaN", "6.0", "true"},
			{"2", "6.0", "false"},
			{"2", "7.1", "false"},
		},
	)
	received := a.String()
	expected := `[4x3] DataFrame

    A     C        D
 0: 1     5.100000 true
 1: NaN   6.000000 true
 2: 2     6.000000 false
 3: 2     7.100000 false
    <int> <float>  <bool>
`
	if expected != received {
		t.Errorf("Different values:\nExpected: \n%v\nReceived: \n%v\n", expected, received)
	}
}

func TestDataFrame_Rapply(t *testing.T) {
	a := LoadRecords(
		[][]string{
			{"A", "B", "C", "D"},
			{"1", "4", "5.1", "1"},
			{"1", "4", "6.0", "1"},
			{"2", "3", "6.0", "0"},
			{"2", "2", "7.1", "0"},
		},
	)
	mean := func(s Series) Series {
		floats := s.Float()
		sum := 0.0
		for _, f := range floats {
			sum += f
		}
		ret := Floats(sum / float64(len(floats)))
		return ret
	}
	sum := func(s Series) Series {
		floats := s.Float()
		sum := 0.0
		for _, f := range floats {
			sum += f
		}
		return Floats(sum)
	}
	table := []struct {
		fun   func(Series) Series
		expDf DataFrame
	}{
		{
			mean,
			LoadRecords(
				[][]string{
					{"X0"},
					{"2.775"},
					{"3"},
					{"2.75"},
					{"2.775"},
				},
				DefaultType(Float),
				DetectTypes(false),
			),
		},
		{
			sum,
			LoadRecords(
				[][]string{
					{"X0"},
					{"11.1"},
					{"12"},
					{"11"},
					{"11.1"},
				},
				DefaultType(Float),
				DetectTypes(false),
			),
		},
	}
	for i, tc := range table {
		b := a.Rapply(tc.fun)

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		//if err := checkAddrDf(a, b); err != nil {
		//t.Error(err)
		//}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

type mockMatrix struct {
	DataFrame
}

func (m mockMatrix) At(i, j int) float64 {
	return m.columns[j].Elem(i).Float()
}

func (m mockMatrix) T() Matrix {
	return m
}

func TestLoadMatrix(t *testing.T) {
	table := []struct {
		b     DataFrame
		expDf DataFrame
	}{
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"4", "1", "true", "0"},
					{"3", "2", "true", "0.5"},
				},
			),
			NewDataframe(
				NewSeries([]string{"4", "3"}, Float, "X0"),
				NewSeries([]int{1, 2}, Float, "X1"),
				NewSeries([]bool{true, true}, Float, "X2"),
				NewSeries([]float64{0, 0.5}, Float, "X3"),
			),
		},
	}
	for i, tc := range table {
		b := LoadMatrix(mockMatrix{tc.b})

		if b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, b.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), b.Records()) {
			t.Errorf("Test: %d\nDifferent values:\nA:%v\nB:%v", i, tc.expDf.Records(), b.Records())
		}
	}
}

func TestLoadStructs(t *testing.T) {
	type testStruct struct {
		A string
		B int
		C bool
		D float64
	}
	type testStructTags struct {
		A string  `dataframe:"a,string"`
		B int     `dataframe:"b,string"`
		C bool    `dataframe:"c,string"`
		D float64 `dataframe:"d,string"`
		E int     `dataframe:"-"` // ignored
		f int     // ignored
	}
	data := []testStruct{
		{"a", 1, true, 0.0},
		{"b", 2, true, 0.5},
	}
	dataTags := []testStructTags{
		{"a", 1, true, 0.0, 0, 0},
		{"NA", 2, true, 0.5, 0, 0},
	}
	table := []struct {
		b     DataFrame
		expDf DataFrame
	}{
		{
			LoadStructs(dataTags),
			NewDataframe(
				NewSeries([]string{"a", "NaN"}, String, "a"),
				NewSeries([]int{1, 2}, String, "b"),
				NewSeries([]bool{true, true}, String, "c"),
				NewSeries([]string{"0.000000", "0.500000"}, String, "d"),
			),
		},
		{
			LoadStructs(data),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]int{1, 2}, Int, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]float64{0, 0.5}, Float, "D"),
			),
		},
		{
			LoadStructs(
				data,
				HasHeader(true),
				DetectTypes(false),
				DefaultType(String),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]int{1, 2}, String, "B"),
				NewSeries([]bool{true, true}, String, "C"),
				NewSeries([]string{"0.000000", "0.500000"}, String, "D"),
			),
		},
		{
			LoadStructs(
				data,
				HasHeader(false),
				DetectTypes(false),
				DefaultType(String),
			),
			NewDataframe(
				NewSeries([]string{"A", "a", "b"}, String, "X0"),
				NewSeries([]string{"B", "1", "2"}, String, "X1"),
				NewSeries([]string{"C", "true", "true"}, String, "X2"),
				NewSeries([]string{"D", "0.000000", "0.500000"}, String, "X3"),
			),
		},
		{
			LoadStructs(
				data,
				HasHeader(true),
				DetectTypes(false),
				DefaultType(String),
				WithTypes(map[string]Type{
					"B": Float,
					"C": String,
				}),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]float64{1, 2}, Float, "B"),
				NewSeries([]bool{true, true}, String, "C"),
				NewSeries([]string{"0.000000", "0.500000"}, String, "D"),
			),
		},
		{
			LoadStructs(
				data,
				HasHeader(true),
				DetectTypes(true),
				DefaultType(String),
				WithTypes(map[string]Type{
					"B": Float,
				}),
			),
			NewDataframe(
				NewSeries([]string{"a", "b"}, String, "A"),
				NewSeries([]float64{1, 2}, Float, "B"),
				NewSeries([]bool{true, true}, Bool, "C"),
				NewSeries([]string{"0", "0.5"}, Float, "D"),
			),
		},
	}
	for i, tc := range table {
		if tc.b.Err != nil {
			t.Errorf("Test: %d\nError:%v", i, tc.b.Err)
		}
		// Check that the types are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Types(), tc.b.Types()) {
			t.Errorf("Test: %d\nDifferent types:\nA:%v\nB:%v", i, tc.expDf.Types(), tc.b.Types())
		}
		// Check that the colnames are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Names(), tc.b.Names()) {
			t.Errorf("Test: %d\nDifferent colnames:\nA:%v\nB:%v", i, tc.expDf.Names(), tc.b.Names())
		}
		// Check that the values are the same between both DataFrames
		if !reflect.DeepEqual(tc.expDf.Records(), tc.b.Records()) {
			t.Errorf("Test: %d: Different values:\nA:%v\nB:%v", i, tc.expDf, tc.b)
		}
	}
}

func TestDescribe(t *testing.T) {
	table := []struct {
		df       DataFrame
		expected DataFrame
	}{
		{
			LoadRecords(
				[][]string{
					{"A", "B", "C", "D"},
					{"a", "4", "5.1", "true"},
					{"b", "4", "6.0", "true"},
					{"c", "3", "6.0", "false"},
					{"a", "2", "7.1", "false"},
				}),

			NewDataframe(
				NewSeries(
					[]string{"mean", "median", "stddev", "min", "25%", "50%", "75%", "max"},
					String,
					"column",
				),
				NewSeries(
					[]string{"-", "-", "-", "a", "-", "-", "-", "c"},
					String,
					"A",
				),
				NewSeries(
					[]float64{3.25, 3.5, 0.957427, 2.0, 2.0, 3.0, 4.0, 4.0},
					Float,
					"B",
				),
				NewSeries(
					[]float64{6.05, 6., 0.818535, 5.1, 5.0, 6.0, 6.0, 7.1},
					Float,
					"C",
				),
				NewSeries(
					[]float64{0.5, math.NaN(), 0.57735, 0.0, 0.0, 0.0, 1.0, 1.0},
					Float,
					"D",
				),
			),
		},
	}

	for testnum, test := range table {
		received := test.df.Describe()
		expected := test.expected

		equal := true
		for i, col := range received.columns {
			lcol := col.Records()
			rcol := expected.columns[i].Records()
			for j, value := range lcol {
				lvalue, lerr := strconv.ParseFloat(value, 64)
				rvalue, rerr := strconv.ParseFloat(rcol[j], 64)
				if lerr != nil || rerr != nil {
					equal = lvalue == rvalue
				} else {
					equal = compareFloats(lvalue, rvalue, 6)
				}
				if !equal {
					break
				}
			}
			if !equal {
				break
			}
		}

		if !equal {
			t.Errorf("Test:%v\nExpected:\n%v\nReceived:\n%v\n", testnum, expected, received)
		}
	}
}

const MIN = 0.000001

func IsEqual(f1, f2 float64) bool {
	if f1 > f2 {
		return math.Dim(f1, f2) < MIN
	} else {
		return math.Dim(f2, f1) < MIN
	}
}
func TestDataFrame_GroupBy(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "a", "b"}, String, "key1"),
		NewSeries([]int{1, 2, 1, 2, 2}, Int, "key2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "values"),
	)
	groups := a.GroupBy("key1", "key2")
	resultMap := make(map[string]float32, 3)
	resultMap[fmt.Sprintf("%s_%d", "a", 2)] = 4 + 3.2
	resultMap[fmt.Sprintf("%s_%d", "b", 1)] = 3 + 5.3
	resultMap[fmt.Sprintf("%s_%d", "b", 2)] = 1.2

	for k, values := range groups.groups {
		curV := 0.0
		for _, vMap := range values.Maps() {
			curV += vMap["values"].(float64)
		}
		targetV, ok := resultMap[k]
		if !ok {
			t.Errorf("GroupBy: %s not found", k)
			return
		}
		if !IsEqual(float64(targetV), curV) {
			t.Errorf("GroupBy: expect %f , but got %f", targetV, curV)
		}
	}

	b := NewDataframe(
		NewSeries([]string{"b", "a", "b", "a", "b"}, String, "key3"),
	)
	groups = b.GroupBy("key1", "key2")
	if groups.Err == nil {
		t.Errorf("GroupBy: COLUMNS NOT FOUND")
	}
}

func TestDataFrame_Aggregation(t *testing.T) {
	a := NewDataframe(
		NewSeries([]string{"b", "a", "b", "a", "b"}, String, "key1"),
		NewSeries([]int{1, 2, 1, 2, 2}, Int, "key2"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "values"),
		NewSeries([]float64{3.0, 4.0, 5.3, 3.2, 1.2}, Float, "values2"),
	)
	groups := a.GroupBy("key1", "key2")
	df := groups.Aggregation([]AggregationType{Aggregation_MAX, Aggregation_MIN, Aggregation_COUNT, Aggregation_SUM}, []string{"values", "values2", "values2", "values2"})
	resultMap := make(map[string]float32, 3)
	resultMap[fmt.Sprintf("%s_%d", "a", 2)] = 4
	resultMap[fmt.Sprintf("%s_%d", "b", 1)] = 5.3
	resultMap[fmt.Sprintf("%s_%d", "b", 2)] = 1.2
	for _, m := range df.Maps() {
		key := fmt.Sprintf("%s_%d", m["key1"], m["key2"])
		if !IsEqual(m["values_MAX"].(float64), float64(resultMap[key])) {
			t.Errorf("Aggregation: expect %f , but got %f", float64(resultMap[key]), m["values"].(float64))
		}
	}
}
