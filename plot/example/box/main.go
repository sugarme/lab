package main

import (
	"image/color"
	"log"
	"math/rand"

	"github.com/sugarme/lab/plot"
)

func main() {
	boxChart()
}

//
// Box Charts
//
func boxChart() {
	pl, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(2, 1))
	if err != nil {
		log.Fatal(err)
	}

	p := plot.BoxChart{Title: "Box Chart"}
	p.XRange.Label, p.YRange.Label = "Value", "Count"

	p.NextDataSet("Sample A", plot.Style{Symbol: '.', LineColor: color.NRGBA{0xff, 0x00, 0x00, 0xff}, LineWidth: 1, LineStyle: plot.SolidLine})
	for x := 10; x <= 50; x += 5 {
		points := make([]float64, 70)
		a := rand.Float64() * 10
		v := rand.Float64()*5 + 2
		for i := 0; i < len(points); i++ {
			x := rand.NormFloat64()*v + a
			points[i] = x
		}
		p.AddSet(float64(x), points, true)
	}

	p.NextDataSet("Sample B", plot.Style{Symbol: '.', LineColor: color.NRGBA{0x00, 0xc0, 0x00, 0xff}, LineWidth: 1, LineStyle: plot.SolidLine})
	for x := 12; x <= 50; x += 10 {
		points := make([]float64, 60)
		a := rand.Float64()*15 + 30
		v := rand.Float64()*5 + 2
		for i := 0; i < len(points); i++ {
			x := rand.NormFloat64()*v + a
			points[i] = x
		}
		p.AddSet(float64(x), points, true)
	}
	pl.Plot(&p)

	p = plot.BoxChart{Title: "Categorical Box Chart"}
	p.XRange.Label, p.YRange.Label = "Population", "Count"
	p.XRange.Fixed(-1, 3, 1)
	p.XRange.Category = []string{"Rural", "Urban", "Island"}

	p.NextDataSet("", plot.Style{Symbol: '%', LineColor: color.NRGBA{0x00, 0x00, 0xcc, 0xff}, LineWidth: 1, LineStyle: plot.SolidLine})
	p.AddSet(0, bigauss(100, 0, 5, 10, 0, 0, 0, 50), true)
	p.AddSet(1, bigauss(100, 25, 5, 5, 2, 25, 0, 50), true)
	p.AddSet(2, bigauss(50, 50, 4, 8, 4, 16, 0, 50), true)
	pl.Plot(&p)

	pl.WriteToFile("box-chart")
}

// bigaussian distribution with n samples, stddev of s, offset of a, clipped to [l,u]
func bigauss(n1, n2 int, s1, a1, s2, a2, l, u float64) []float64 {
	points := make([]float64, n1+n2)
	for i := 0; i < n1; i++ {
		x := rand.NormFloat64()*s1 + a1
		for x < l || x > u {
			x = rand.NormFloat64()*s1 + a1
		}
		points[i] = x
	}
	for i := n1; i < n1+n2; i++ {
		x := rand.NormFloat64()*s2 + a2
		for x < l || x > u {
			x = rand.NormFloat64()*s2 + a2
		}
		points[i] = x
	}
	return points
}
