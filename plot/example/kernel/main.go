package main

import (
	"image/color"
	"log"

	"github.com/sugarme/lab/plot"
)

func main() {
	kernels()
}

func kernels() {
	pl, err := plot.NewPlotter(600, 400)
	if err != nil {
		log.Fatal(err)
	}

	p := plot.ScatterChart{Title: "Kernels"}
	p.XRange.Label, p.YRange.Label = "u", "K(u)"
	p.XRange.MinMode.Fixed, p.XRange.MaxMode.Fixed = true, true
	p.XRange.MinMode.Value, p.XRange.MaxMode.Value = -2, 2
	p.YRange.MinMode.Fixed, p.YRange.MaxMode.Fixed = true, true
	p.YRange.MinMode.Value, p.YRange.MaxMode.Value = -0.1, 1.1

	p.XRange.TicSetting.Delta = 1
	p.YRange.TicSetting.Delta = 0.2
	p.XRange.TicSetting.Mirror = 1
	p.YRange.TicSetting.Mirror = 1

	p.AddFunc("Bisquare", plot.BisquareKernel,
		plot.PlotStyleLines, plot.Style{Symbol: 'o', LineWidth: 1, LineColor: color.NRGBA{0xa0, 0x00, 0x00, 0xff}, LineStyle: 1})
	p.AddFunc("Epanechnikov", plot.EpanechnikovKernel,
		plot.PlotStyleLines, plot.Style{Symbol: 'X', LineWidth: 1, LineColor: color.NRGBA{0x00, 0xa0, 0x00, 0xff}, LineStyle: 1})
	p.AddFunc("Rectangular", plot.RectangularKernel,
		plot.PlotStyleLines, plot.Style{Symbol: '=', LineWidth: 1, LineColor: color.NRGBA{0x00, 0x00, 0xa0, 0xff}, LineStyle: 1})
	p.AddFunc("Gauss", plot.GaussKernel,
		plot.PlotStyleLines, plot.Style{Symbol: '*', LineWidth: 1, LineColor: color.NRGBA{0xa0, 0x00, 0xa0, 0xff}, LineStyle: 1})

	pl.Plot(&p)

	pl.WriteToFile("kernels")
}
