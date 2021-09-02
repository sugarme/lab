package main

import (
	"image/color"
	"log"
	"math/rand"

	"github.com/sugarme/lab/plot"
)

func main() {
	autoscale()
}

//
// Autoscaling
//
func autoscale() {
	pl, err := plot.NewPlotter(600, 400, plot.WithPlotterLayout(2, 2))
	if err != nil {
		log.Fatal(err)
	}

	N := 200
	points := make([]plot.EPoint, N)
	for i := 0; i < N-1; i++ {
		points[i].X = rand.Float64()*10000 - 5000 // Full range is [-5000:5000]
		points[i].Y = rand.Float64()*10000 - 5000 // Full range is [-5000:5000]
		points[i].DeltaX = rand.Float64() * 400
		points[i].DeltaY = rand.Float64() * 400
	}
	points[N-1].X = -650
	points[N-1].Y = -2150
	points[N-1].DeltaX = 400
	points[N-1].DeltaY = 400
	points[N-1].OffX = 100
	points[N-1].OffY = -150

	s := plot.ScatterChart{Title: "Full Autoscaling"}
	s.Key.Hide = true
	s.XRange.TicSetting.Mirror = 1
	s.YRange.TicSetting.Mirror = 1
	s.AddData("Data", points, plot.PlotStylePoints, plot.Style{Symbol: 'o', SymbolColor: color.NRGBA{0x00, 0xee, 0x00, 0xff}})
	pl.Plot(&s)

	s = plot.ScatterChart{Title: "Xmin: -1850, Xmax clipped to [500:900]"}
	s.Key.Hide = true
	s.XRange.TicSetting.Mirror = 1
	s.YRange.TicSetting.Mirror = 1
	s.XRange.MinMode.Fixed, s.XRange.MinMode.Value = true, -1850
	s.XRange.MaxMode.Constrained = true
	s.XRange.MaxMode.Lower, s.XRange.MaxMode.Upper = 500, 900

	s.AddData("Data", points, plot.PlotStylePoints, plot.Style{Symbol: '0', SymbolColor: color.NRGBA{0xee, 0x00, 0x00, 0xff}})
	pl.Plot(&s)

	s = plot.ScatterChart{Title: "Xmin: -1850, Ymax clipped to [9000:11000]"}
	s.Key.Hide = true
	s.XRange.TicSetting.Mirror = 1
	s.YRange.TicSetting.Mirror = 1
	s.XRange.MinMode.Fixed, s.XRange.MinMode.Value = true, -1850
	s.YRange.MaxMode.Constrained = true
	s.YRange.MaxMode.Lower, s.YRange.MaxMode.Upper = 9000, 11000

	s.AddData("Data", points, plot.PlotStylePoints, plot.Style{Symbol: '0', SymbolColor: color.NRGBA{0x00, 0x00, 0xee, 0xff}})
	pl.Plot(&s)

	s = plot.ScatterChart{Title: "Tiny fraction"}
	s.Key.Hide = true
	s.XRange.TicSetting.Mirror = 1
	s.YRange.TicSetting.Mirror = 1

	s.YRange.MinMode.Constrained = true
	s.YRange.MinMode.Lower, s.YRange.MinMode.Upper = -2250, -2050
	s.YRange.MaxMode.Constrained = true
	s.YRange.MaxMode.Lower, s.YRange.MaxMode.Upper = -1950, -1700

	s.XRange.MinMode.Constrained = true
	s.XRange.MinMode.Lower, s.XRange.MinMode.Upper = -900, -800
	s.XRange.MaxMode.Constrained = true
	s.XRange.MaxMode.Lower, s.XRange.MaxMode.Upper = -850, -650

	s.AddData("Data", points, plot.PlotStylePoints, plot.Style{Symbol: '0', SymbolColor: color.NRGBA{0xee, 0xcc, 0x00, 0xff}})
	pl.Plot(&s)

	pl.WriteToFile("autoscale")
}
