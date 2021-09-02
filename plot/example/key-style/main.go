package main

import (
	"fmt"
	"image/color"
	"log"
	"math"

	"github.com/sugarme/lab/plot"
)

func main() {
	keyStyles()
}

func keyStyles() {
	pl, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(6, 6), plot.WithPlotterType("png"))
	if err != nil {
		log.Fatal(err)
	}

	p := plot.ScatterChart{Title: "Key Placement"}
	p.XRange.TicSetting.Mirror, p.YRange.TicSetting.Mirror = 1, 1
	p.XRange.MinMode.Fixed, p.XRange.MaxMode.Fixed = true, true
	p.XRange.MinMode.Value, p.XRange.MaxMode.Value = -5, 5
	p.XRange.Min, p.XRange.Max = -5, 5
	p.XRange.TicSetting.Delta = 2

	p.YRange.MinMode.Fixed, p.YRange.MaxMode.Fixed = true, true
	p.YRange.MinMode.Value, p.YRange.MaxMode.Value = -5, 5
	p.YRange.Min, p.YRange.Max = -5, 5
	p.YRange.TicSetting.Delta = 3

	p.AddFunc("Sin", func(x float64) float64 { return math.Sin(x) }, plot.PlotStyleLines,
		plot.Style{LineColor: color.NRGBA{0xa0, 0x00, 0x00, 0xff}, LineWidth: 1, LineStyle: 1})
	p.AddFunc("Cos", func(x float64) float64 { return math.Cos(x) }, plot.PlotStyleLines,
		plot.Style{LineColor: color.NRGBA{0x00, 0xa0, 0x00, 0xff}, LineWidth: 1, LineStyle: 1})
	p.AddFunc("Tan", func(x float64) float64 { return math.Tan(x) }, plot.PlotStyleLines,
		plot.Style{LineColor: color.NRGBA{0x00, 0x00, 0xa0, 0xff}, LineWidth: 1, LineStyle: 1})

	for _, pos := range []string{"itl", "itc", "itr", "icl", "icc", "icr", "ibl", "ibc", "ibr",
		"otl", "otc", "otr", "olt", "olc", "olb", "obl", "obc", "obr", "ort", "orc", "orb"} {
		p.Key.Pos = pos
		p.Title = "Key Placement: " + pos
		pl.Plot(&p)
	}

	p.Key.Pos = "itl"
	p.AddFunc("Log", func(x float64) float64 { return math.Log(x) }, plot.PlotStyleLines,
		plot.Style{LineColor: color.NRGBA{0xff, 0x60, 0x60, 0xff}, LineWidth: 1, LineStyle: 1})
	p.AddFunc("Exp", func(x float64) float64 { return math.Exp(x) }, plot.PlotStyleLines,
		plot.Style{LineColor: color.NRGBA{0x60, 0xff, 0x60, 0xff}, LineWidth: 1, LineStyle: 1})
	p.AddFunc("Atan", func(x float64) float64 { return math.Atan(x) }, plot.PlotStyleLines,
		plot.Style{LineColor: color.NRGBA{0x60, 0x60, 0xff, 0xff}, LineWidth: 1, LineStyle: 1})
	p.AddFunc("Y1", func(x float64) float64 { return math.Y1(x) }, plot.PlotStyleLines,
		plot.Style{LineColor: color.NRGBA{0xd0, 0xd0, 0x00, 0xff}, LineWidth: 1, LineStyle: 1})

	for _, cols := range []int{-4, -3, -2, -1, 0, 1, 2, 3, 4} {
		p.Key.Cols = cols
		p.Title = fmt.Sprintf("Key Cols: %d", cols)
		pl.Plot(&p)
	}

	pl.WriteToFile("key-styles")
}
