package data_test

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/sugarme/lab/data"
)

func generateSeries(n, rep int) (dat []data.Series) {
	rand.Seed(100)
	for j := 0; j < rep; j++ {
		var is []int
		var bs []bool
		var fs []float64
		var ss []string
		for i := 0; i < n; i++ {
			is = append(is, rand.Int())
		}
		for i := 0; i < n; i++ {
			fs = append(fs, rand.Float64())
		}
		for i := 0; i < n; i++ {
			ss = append(ss, strconv.Itoa(rand.Int()))
		}
		for i := 0; i < n; i++ {
			r := rand.Intn(2)
			b := false
			if r == 1 {
				b = true
			}
			bs = append(bs, b)
		}
		dat = append(dat, data.Ints(is))
		dat = append(dat, data.Bools(bs))
		dat = append(dat, data.Floats(fs))
		dat = append(dat, data.Strings(ss))
	}
	return
}

func generateIntsN(n, k int) (data []int) {
	for i := 0; i < n; i++ {
		data = append(data, rand.Intn(k))
	}
	return
}

func BenchmarkNew(b *testing.B) {
	table := []struct {
		name string
		data []data.Series
	}{
		{
			"100000x4",
			generateSeries(100000, 1),
		},
		{
			"100000x40",
			generateSeries(100000, 10),
		},
		{
			"100000x400",
			generateSeries(100000, 100),
		},
		{
			"1000x40",
			generateSeries(1000, 10),
		},
		{
			"1000x4000",
			generateSeries(1000, 1000),
		},
		{
			"1000x40000",
			generateSeries(1000, 10000),
		},
	}
	for _, test := range table {
		b.Run(test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				data.NewDataframe(test.data...)
			}
		})
	}
}

func BenchmarkDataFrame_Arrange(b *testing.B) {
	dat := data.NewDataframe(generateSeries(100000, 5)...)
	table := []struct {
		name string
		dat  data.DataFrame
		key  []data.Order
	}{
		{
			"100000x20_1",
			dat,
			[]data.Order{data.Sort("X0")},
		},
		{
			"100000x20_2",
			dat,
			[]data.Order{
				data.Sort("X0"),
				data.Sort("X1"),
			},
		},
		{
			"100000x20_3",
			dat,
			[]data.Order{
				data.Sort("X0"),
				data.Sort("X1"),
				data.Sort("X2"),
			},
		},
	}
	for _, test := range table {
		b.Run(test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				test.dat.Arrange(test.key...)
			}
		})
	}
}

func BenchmarkDataFrame_Subset(b *testing.B) {
	b.ReportAllocs()
	data1000x20 := data.NewDataframe(generateSeries(1000, 5)...)
	data1000x200 := data.NewDataframe(generateSeries(1000, 50)...)
	data1000x2000 := data.NewDataframe(generateSeries(1000, 500)...)
	data100000x20 := data.NewDataframe(generateSeries(100000, 5)...)
	data1000000x20 := data.NewDataframe(generateSeries(1000000, 5)...)
	idx10 := generateIntsN(10, 10)
	idx100 := generateIntsN(100, 100)
	idx1000 := generateIntsN(1000, 1000)
	idx10000 := generateIntsN(10000, 10000)
	idx100000 := generateIntsN(100000, 100000)
	idx1000000 := generateIntsN(1000000, 1000000)
	table := []struct {
		name    string
		data    data.DataFrame
		indexes interface{}
	}{
		{
			"1000000x20_100",
			data1000000x20,
			idx100,
		},
		{
			"1000000x20_1000",
			data1000000x20,
			idx1000,
		},
		{
			"1000000x20_10000",
			data1000000x20,
			idx10000,
		},
		{
			"1000000x20_100000",
			data1000000x20,
			idx100000,
		},
		{
			"1000000x20_1000000",
			data1000000x20,
			idx1000000,
		},
		{
			"100000x20_100",
			data100000x20,
			idx100,
		},
		{
			"100000x20_1000",
			data100000x20,
			idx1000,
		},
		{
			"100000x20_10000",
			data100000x20,
			idx10000,
		},
		{
			"100000x20_100000",
			data100000x20,
			idx100000,
		},
		{
			"1000x20_10",
			data1000x20,
			idx10,
		},
		{
			"1000x20_100",
			data1000x20,
			idx100,
		},
		{
			"1000x20_1000",
			data1000x20,
			idx1000,
		},
		{
			"1000x200_10",
			data1000x200,
			idx10,
		},
		{
			"1000x200_100",
			data1000x200,
			idx100,
		},
		{
			"1000x200_1000",
			data1000x200,
			idx1000,
		},
		{
			"1000x2000_10",
			data1000x2000,
			idx10,
		},
		{
			"1000x2000_100",
			data1000x2000,
			idx100,
		},
		{
			"1000x2000_1000",
			data1000x2000,
			idx1000,
		},
	}
	for _, test := range table {
		b.Run(test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				test.data.Subset(test.indexes)
			}
		})
	}
}

func BenchmarkDataFrame_Elem(b *testing.B) {
	dat := data.NewDataframe(generateSeries(100000, 5)...)
	table := []struct {
		name string
		dat  data.DataFrame
	}{
		{
			"100000x20_ALL",
			dat,
		},
	}
	for _, test := range table {
		b.Run(test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for k := 0; k < 100000; k++ {
					test.dat.Elem(k, 0)
				}
			}
		})
	}
}
