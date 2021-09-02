package main

import (
	"image/color"
	"log"
	"math"

	"github.com/sugarme/lab/plot"
)

func main() {
	functionPlots()
}

//
// Function plots with fancy clippings
//
func functionPlots() {
	pl, err := plot.NewPlotter(500, 400, plot.WithPlotterLayout(2, 1))
	if err != nil {
		log.Fatal(err)
	}

	p := plot.ScatterChart{Title: "Functions"}
	p.XRange.Label, p.YRange.Label = "X - Value", "Y - Value"
	p.Key.Pos = "ibl"
	p.XRange.MinMode.Fixed, p.XRange.MaxMode.Fixed = true, true
	p.XRange.MinMode.Value, p.XRange.MaxMode.Value = -10, 10
	p.YRange.MinMode.Fixed, p.YRange.MaxMode.Fixed = true, true
	p.YRange.MinMode.Value, p.YRange.MaxMode.Value = -10, 10

	p.XRange.TicSetting.Delta = 2
	p.YRange.TicSetting.Delta = 5
	p.XRange.TicSetting.Mirror = 1
	p.YRange.TicSetting.Mirror = 1

	p.AddFunc("i+n", func(x float64) float64 {
		if x > -7 && x < -5 {
			return math.Inf(-1)
		} else if x > -1.5 && x < 1.5 {
			return math.NaN()
		} else if x > 5 && x < 7 {
			return math.Inf(1)
		}
		return -0.75 * x
	},
		plot.PlotStyleLines, plot.Style{Symbol: 'o', LineWidth: 2, LineColor: color.NRGBA{0xa0, 0x00, 0x00, 0xff}, LineStyle: 1})
	p.AddFunc("sin", func(x float64) float64 { return 13 * math.Sin(x) }, plot.PlotStyleLines,
		plot.Style{Symbol: '#', LineWidth: 1, LineColor: color.NRGBA{0x00, 0x00, 0xa0, 0xff}, LineStyle: 1})
	p.AddFunc("2x", func(x float64) float64 { return 2 * x }, plot.PlotStyleLines,
		plot.Style{Symbol: 'X', LineWidth: 1, LineColor: color.NRGBA{0x00, 0xa0, 0x00, 0xff}, LineStyle: 1})

	pl.Plot(&p)

	p = plot.ScatterChart{Title: "Functions"}
	p.Key.Hide = true
	p.XRange.MinMode.Fixed, p.XRange.MaxMode.Fixed = true, true
	p.XRange.MinMode.Value, p.XRange.MaxMode.Value = -2, 2
	p.YRange.MinMode.Fixed, p.YRange.MaxMode.Fixed = true, true
	p.YRange.MinMode.Value, p.YRange.MaxMode.Value = -2, 2
	p.XRange.TicSetting.Delta = 1
	p.YRange.TicSetting.Delta = 1
	p.XRange.TicSetting.Mirror = 1
	p.YRange.TicSetting.Mirror = 1
	p.NSamples = 5
	p.AddFunc("10x", func(x float64) float64 { return 10 * x }, plot.PlotStyleLines,
		plot.Style{Symbol: 'o', LineWidth: 2, LineColor: color.NRGBA{0x00, 0xa0, 0x00, 0xff}, LineStyle: 1})
	pl.Plot(&p)

	pl.WriteToFile("func-plot")
}
