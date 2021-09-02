package main

import (
	"image/color"
	"log"

	"github.com/sugarme/lab/plot"
)

func main() {
	scatterChart()
}

func scatterChart() {
	p, err := plot.NewPlotter(600, 400, plot.WithPlotterType("png"))
	if err != nil {
		log.Fatal(err)
	}

	// create scatter chart
	c := new(plot.ScatterChart)
	c.Title = "Scatter + Lines"

	c.XRange.TicSetting.Mirror, c.YRange.TicSetting.Mirror = 0, 0
	c.XRange.MinMode.Fixed, c.XRange.MaxMode.Fixed = true, true
	c.XRange.MinMode.Value, c.XRange.MaxMode.Value = -10, 10
	c.XRange.Min, c.XRange.Max = -10, 10
	c.XRange.TicSetting.Delta = 2

	c.YRange.MinMode.Fixed, c.YRange.MaxMode.Fixed = true, true
	c.YRange.MinMode.Value, c.YRange.MaxMode.Value = -1, 100
	c.YRange.Min, c.YRange.Max = -1, 100
	c.YRange.TicSetting.Delta = 10

	c.Key.Cols = 3
	c.Key.Pos = "obc"

	// Line-point
	x := []float64{-4, -3.3, -1.8, -1, 0.2, 0.8, 1.8, 3.1, 4, 5.3, 6, 7, 8, 9}
	y := []float64{22, 18, -3, 0, 0.5, 2, 45, 12, 16.5, 24, 30, 55, 60, 70}
	style1 := plot.Style{Symbol: '.', SymbolColor: color.NRGBA{0x00, 0x00, 0xff, 0xff}, LineStyle: plot.SolidLine}
	c.AddDataPair("Data", x, y, plot.PlotStyleLinesPoints, style1)

	// last := len(c.Data) - 1
	// c.Data[last].Samples[6].DeltaX = 2.5
	// c.Data[last].Samples[6].OffX = 0.5
	// c.Data[last].Samples[6].DeltaY = 16
	// c.Data[last].Samples[6].OffY = 2

	// Points
	points := []plot.EPoint{
		{-4, 40, 0, 0, 0, 0},
		{-3, 45, 0, 0, 0, 0},
		{-2, 35, 0, 0, 0, 0},
	}
	pointStyle := plot.Style{Symbol: '0', SymbolColor: color.NRGBA{0xff, 0x00, 0xff, 0xff}}
	c.AddData("Points", points, plot.PlotStylePoints, pointStyle)

	// Func
	theoryFn := func(x float64) float64 {
		if x > 5.25 && x < 5.75 {
			return 75
		}
		if x > 7.25 && x < 7.75 {
			return 500
		}
		return x * x
	}
	funcStyle1 := plot.Style{Symbol: '%', LineWidth: 2, LineColor: color.NRGBA{0xa0, 0x00, 0x00, 0xff}, LineStyle: plot.DottedLine}
	c.AddFunc("Theory", theoryFn, plot.PlotStyleLines, funcStyle1)

	funcStyle2 := plot.Style{Symbol: '+', LineWidth: 1, LineColor: color.NRGBA{0x00, 0xa0, 0x00, 0xff}, LineStyle: plot.DottedLine}
	c.AddFunc("Upper Range", func(x float64) float64 { return 30 }, plot.PlotStyleLines, funcStyle2)

	funcStyle3 := plot.Style{Symbol: '@', LineWidth: 1, LineColor: color.NRGBA{0x00, 0x00, 0xa0, 0xff}, LineStyle: plot.DottedLine}
	c.AddFunc("Lower Range", func(x float64) float64 { return 7 }, plot.PlotStyleLines, funcStyle3)

	p.Plot(c)
	p.WriteToFile("scatter-chart")
}
