package main

import (
	"image/color"
	"log"

	"github.com/sugarme/lab/plot"
)

func main() {
	logAxis()
}

//
// Logarithmic axes
//
func logAxis() {
	pl, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(2, 2))
	if err != nil {
		log.Fatal(err)
	}

	lc := plot.ScatterChart{}
	lc.XRange.Label, lc.YRange.Label = "X-Value", "Y-Value"
	lx := []float64{4e-2, 3e-1, 2e0, 1e1, 8e1, 7e2, 5e3}
	ly := []float64{10, 30, 90, 270, 3 * 270, 9 * 270, 27 * 270}
	lc.AddDataPair("Measurement", lx, ly, plot.PlotStylePoints,
		plot.Style{Symbol: '#', SymbolColor: color.NRGBA{0x99, 0x66, 0xff, 0xff}, SymbolSize: 1.5})
	lc.Key.Hide = true
	lc.XRange.MinMode.Expand, lc.XRange.MaxMode.Expand = plot.ExpandToTic, plot.ExpandToTic
	lc.YRange.MinMode.Expand, lc.YRange.MaxMode.Expand = plot.ExpandToTic, plot.ExpandToTic
	lc.Title = "Lin / Lin"
	lc.XRange.Min, lc.XRange.Max = 0, 0
	lc.YRange.Min, lc.YRange.Max = 0, 0
	pl.Plot(&lc)

	lc.Title = "Lin / Log"
	lc.XRange.Log, lc.YRange.Log = false, true
	lc.XRange.Min, lc.XRange.Max, lc.XRange.TicSetting.Delta = 0, 0, 0
	lc.YRange.Min, lc.YRange.Max, lc.YRange.TicSetting.Delta = 0, 0, 0
	pl.Plot(&lc)

	lc.Title = "Log / Lin"
	lc.XRange.Log, lc.YRange.Log = true, false
	lc.XRange.Min, lc.XRange.Max, lc.XRange.TicSetting.Delta = 0, 0, 0
	lc.YRange.Min, lc.YRange.Max, lc.YRange.TicSetting.Delta = 0, 0, 0
	pl.Plot(&lc)

	lc.Title = "Log / Log"
	lc.XRange.Log, lc.YRange.Log = true, true
	lc.XRange.Min, lc.XRange.Max, lc.XRange.TicSetting.Delta = 0, 0, 0
	lc.YRange.Min, lc.YRange.Max, lc.YRange.TicSetting.Delta = 0, 0, 0
	pl.Plot(&lc)

	pl.WriteToFile("log-axis")
}
