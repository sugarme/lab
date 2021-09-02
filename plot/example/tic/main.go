package main

import (
	"fmt"
	"image/color"
	"log"
	"math"

	"github.com/sugarme/lab/plot"
)

func main() {
	scatterTics()
}

//
// Scatter plots with different tic/grid settings
//
func scatterTics() {
	var (
		data1  = []float64{15e-7, 30e-7, 35e-7, 50e-7, 70e-7, 75e-7, 80e-7, 32e-7, 35e-7, 70e-7, 65e-7}
		data10 = []float64{34567, 35432, 37888, 39991, 40566, 42123, 44678}
	)

	pl, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(3, 3))
	if err != nil {
		log.Fatal(err)
	}

	p := plot.ScatterChart{Title: "Sample Scatter Chart"}
	p.AddDataPair("Sample A", data10, data1, plot.PlotStylePoints, plot.Style{})
	p.XRange.TicSetting.Delta = 5000
	p.XRange.Label = "X - Value"
	p.YRange.Label = "Y - Value"
	pl.Plot(&p)

	p.XRange.TicSetting.Hide, p.YRange.TicSetting.Hide = true, true
	pl.Plot(&p)

	p.YRange.TicSetting.Hide = false
	p.XRange.TicSetting.Grid, p.YRange.TicSetting.Grid = plot.GridLines, plot.GridLines
	pl.Plot(&p)

	p.XRange.TicSetting.Hide, p.YRange.TicSetting.Hide = false, false
	p.XRange.TicSetting.Mirror, p.YRange.TicSetting.Mirror = 1, 2
	pl.Plot(&p)

	c := plot.ScatterChart{Title: "Own tics"}
	c.XRange.Fixed(0, 4*math.Pi, math.Pi)
	c.YRange.Fixed(-1.25, 1.25, 0.5)
	c.XRange.TicSetting.Format = func(f float64) string {
		w := int(180*f/math.Pi + 0.5)
		return fmt.Sprintf("%d°", w)
	}
	c.AddFunc("Sin(x)", func(x float64) float64 { return math.Sin(x) }, plot.PlotStyleLines,
		plot.Style{Symbol: '@', LineWidth: 2, LineColor: color.NRGBA{0x00, 0x00, 0xcc, 0xff}, LineStyle: 0})
	c.AddFunc("Cos(x)", func(x float64) float64 { return math.Cos(x) }, plot.PlotStyleLines,
		plot.Style{Symbol: '%', LineWidth: 2, LineColor: color.NRGBA{0x00, 0xcc, 0x00, 0xff}, LineStyle: 0})
	pl.Plot(&c)

	c.Title = "Tic Variants"
	c.XRange.TicSetting.Tics = 1
	c.YRange.TicSetting.Tics = 2
	pl.Plot(&c)

	c.Title = "Blocked Grid"
	c.XRange.TicSetting.Tics = 1
	c.YRange.TicSetting.Tics = 1
	c.XRange.TicSetting.Mirror, c.YRange.TicSetting.Mirror = 1, 1
	c.XRange.TicSetting.Grid = plot.GridBlocks
	c.YRange.TicSetting.Grid = plot.GridBlocks
	pl.Plot(&c)

	pl.WriteToFile("scatter-tics")
}
