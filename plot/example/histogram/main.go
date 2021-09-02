package main

import (
	"log"
	"math/rand"

	"github.com/sugarme/lab/plot"
)

func main() {
	histChart("Histogram", false, false, false)
}

//
// Histograms Charts
//
func histChart(title string, stacked, counts, shifted bool) {
	pl, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(2, 2))
	if err != nil {
		log.Fatal(err)
	}

	hc := plot.HistChart{Title: title, Stacked: stacked, Counts: counts, Shifted: shifted}
	hc.XRange.Label = "Sample Value"
	if counts {
		hc.YRange.Label = "Total Count"
	} else {
		hc.YRange.Label = "Rel. Frequency [%]"
	}
	hc.Key.Hide = true
	points := gauss(150, 10, 20, 0, 50)
	hc.AddData("Sample 1", points,
		plot.Style{ /*LineColor: color.NRGBA{0xff,0x00,0x00,0xff}, LineWidth: 1, LineStyle: 1, FillColor: color.NRGBA{0xff,0x80,0x80,0xff}*/ })
	hc.Kernel = plot.BisquareKernel //  plot.GaussKernel // plot.EpanechnikovKernel // plot.RectangularKernel // plot.BisquareKernel
	pl.Plot(&hc)

	points2 := gauss(80, 4, 37, 0, 50)
	// hc.Kernel = nil
	hc.AddData("Sample 2", points2,
		plot.Style{ /*LineColor: color.NRGBA{0x00,0xff,0x00,0xff}, LineWidth: 1, LineStyle: 1, FillColor: color.NRGBA{0x80,0xff,0x80,0xff}*/ })
	hc.YRange.TicSetting.Delta = 0
	pl.Plot(&hc)

	points3 := gauss(60, 15, 0, 0, 50)
	hc.AddData("Sample 3", points3,
		plot.Style{ /*LineColor: color.NRGBA{0x00,0x00,0xff,0xff}, LineWidth: 1, LineStyle: 1, FillColor: color.NRGBA{0x80,0x80,0xff,0xff}*/ })
	hc.YRange.TicSetting.Delta = 0
	pl.Plot(&hc)

	points4 := gauss(40, 30, 15, 0, 50)
	hc.AddData("Sample 4", points4, plot.Style{ /*LineColor: color.NRGBA{0x00,0x00,0x00,0xff}, LineWidth: 1, LineStyle: 1*/ })
	hc.Kernel = nil
	hc.YRange.TicSetting.Delta = 0
	pl.Plot(&hc)

	pl.WriteToFile("histogram")
}

// gaussian distribution with n samples, stddev of s, offset of a, forced to [l,u]
func gauss(n int, s, a, l, u float64) []float64 {
	// Make output of gauss deterministic by seeding with a fixed value.
	rand.Seed(12345)
	points := make([]float64, n)
	for i := 0; i < len(points); i++ {
		x := rand.NormFloat64()*s + a
		if x < l {
			x = l
		} else if x > u {
			x = u
		}
		points[i] = x
	}
	return points
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
