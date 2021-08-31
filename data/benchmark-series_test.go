package data_test

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/sugarme/lab/data"
)

func generateInts(n int) (data []int) {
	for i := 0; i < n; i++ {
		data = append(data, rand.Int())
	}
	return
}

func generateFloats(n int) (data []float64) {
	for i := 0; i < n; i++ {
		data = append(data, rand.Float64())
	}
	return
}

func generateStrings(n int) (data []string) {
	for i := 0; i < n; i++ {
		data = append(data, strconv.Itoa(rand.Int()))
	}
	return
}

func generateBools(n int) (data []bool) {
	for i := 0; i < n; i++ {
		r := rand.Intn(2)
		b := false
		if r == 1 {
			b = true
		}
		data = append(data, b)
	}
	return
}

// func generateIntsN(n, k int) (data []int) {
// for i := 0; i < n; i++ {
// data = append(data, rand.Intn(k))
// }
// return
// }

func BenchmarkSeries_New(b *testing.B) {
	rand.Seed(100)
	table := []struct {
		name       string
		data       interface{}
		seriesType data.Type
	}{
		{
			"[]bool(100000)_Int",
			generateBools(100000),
			data.Int,
		},
		{
			"[]bool(100000)_String",
			generateBools(100000),
			data.String,
		},
		{
			"[]bool(100000)_Bool",
			generateBools(100000),
			data.Bool,
		},
		{
			"[]bool(100000)_Float",
			generateBools(100000),
			data.Float,
		},
		{
			"[]string(100000)_Int",
			generateStrings(100000),
			data.Int,
		},
		{
			"[]string(100000)_String",
			generateStrings(100000),
			data.String,
		},
		{
			"[]string(100000)_Bool",
			generateStrings(100000),
			data.Bool,
		},
		{
			"[]string(100000)_Float",
			generateStrings(100000),
			data.Float,
		},
		{
			"[]float64(100000)_Int",
			generateFloats(100000),
			data.Int,
		},
		{
			"[]float64(100000)_String",
			generateFloats(100000),
			data.String,
		},
		{
			"[]float64(100000)_Bool",
			generateFloats(100000),
			data.Bool,
		},
		{
			"[]float64(100000)_Float",
			generateFloats(100000),
			data.Float,
		},
		{
			"[]int(100000)_Int",
			generateInts(100000),
			data.Int,
		},
		{
			"[]int(100000)_String",
			generateInts(100000),
			data.String,
		},
		{
			"[]int(100000)_Bool",
			generateInts(100000),
			data.Bool,
		},
		{
			"[]int(100000)_Float",
			generateInts(100000),
			data.Float,
		},
	}
	for _, test := range table {
		b.Run(test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				data.NewSeries(test.data, test.seriesType, test.name)
			}
		})
	}
}

func BenchmarkSeries_Copy(b *testing.B) {
	rand.Seed(100)
	table := []struct {
		name   string
		series data.Series
	}{
		{
			"[]int(100000)_Int",
			data.Ints(generateInts(100000)),
		},
		{
			"[]int(100000)_String",
			data.Strings(generateInts(100000)),
		},
		{
			"[]int(100000)_Bool",
			data.Bools(generateInts(100000)),
		},
		{
			"[]int(100000)_Float",
			data.Floats(generateInts(100000)),
		},
	}
	for _, test := range table {
		b.Run(test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				test.series.Copy()
			}
		})
	}
}

func BenchmarkSeries_Subset(b *testing.B) {
	rand.Seed(100)
	table := []struct {
		name    string
		indexes interface{}
		series  data.Series
	}{
		{
			"[]int(100000)_Int",
			generateIntsN(10000, 2),
			data.Ints(generateInts(100000)),
		},
		{
			"[]int(100000)_String",
			generateIntsN(10000, 2),
			data.Strings(generateInts(100000)),
		},
		{
			"[]int(100000)_Bool",
			generateIntsN(10000, 2),
			data.Bools(generateInts(100000)),
		},
		{
			"[]int(100000)_Float",
			generateIntsN(10000, 2),
			data.Floats(generateInts(100000)),
		},
	}
	for _, test := range table {
		b.Run(test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				test.series.Subset(test.indexes)
			}
		})
	}
}

func BenchmarkSeries_Set(b *testing.B) {
	rand.Seed(100)
	table := []struct {
		name      string
		indexes   interface{}
		newValues data.Series
		series    data.Series
	}{
		{
			"[]int(100000)_Int",
			generateIntsN(10000, 2),
			data.Ints(generateIntsN(10000, 2)),
			data.Ints(generateInts(100000)),
		},
		{
			"[]int(100000)_String",
			generateIntsN(10000, 2),
			data.Strings(generateIntsN(10000, 2)),
			data.Strings(generateInts(100000)),
		},
		{
			"[]int(100000)_Bool",
			generateIntsN(10000, 2),
			data.Bools(generateIntsN(10000, 2)),
			data.Bools(generateInts(100000)),
		},
		{
			"[]int(100000)_Float",
			generateIntsN(10000, 2),
			data.Floats(generateIntsN(10000, 2)),
			data.Floats(generateInts(100000)),
		},
	}
	for _, test := range table {
		s := test.series.Copy()
		b.Run(test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s.Set(test.indexes, test.newValues)
			}
		})
	}
}
