package main

import (
	"log"

	"github.com/sugarme/lab/plot"
)

func main() {
	stripChart()
}

func stripChart() {
	var (
		data1 = []float64{15e-7, 30e-7, 35e-7, 50e-7, 70e-7, 75e-7, 80e-7, 32e-7, 35e-7, 70e-7, 65e-7}
		data2 = []float64{10e-7, 11e-7, 12e-7, 22e-7, 25e-7, 33e-7}
		data3 = []float64{50e-7, 55e-7, 55e-7, 60e-7, 50e-7, 65e-7, 60e-7, 65e-7, 55e-7, 50e-7}
	)
	p, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(2, 2), plot.WithPlotterType("png"))
	if err != nil {
		log.Fatal(err)
	}

	c := plot.StripChart{}

	c.AddData("Sample A", data1, plot.Style{})
	c.AddData("Sample B", data2, plot.Style{})
	c.AddData("Sample C", data3, plot.Style{})
	c.Title = "Sample Strip Chart (no Jitter)"
	c.XRange.Label = "X - Axis"
	c.Key.Pos = "icr"
	p.Plot(&c)

	c.Jitter = true
	c.Title = "Sample Strip Chart (with Jitter)"
	p.Plot(&c)

	c.Key.Hide = true
	p.Plot(&c)

	c.Jitter = false
	c.Title = "Sample Strip Chart (no Jitter)"
	p.Plot(&c)

	err = p.WriteToFile("strip-chart")
	if err != nil {
		log.Fatal(err)
	}
}
