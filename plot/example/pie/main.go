package main

import (
	"image/color"
	"log"

	"github.com/sugarme/lab/plot"
)

func main() {
	pieChart()
}

//
// Pie Charts
//
func pieChart() {
	pl1, err := plot.NewPlotter(500, 250, plot.WithPlotterLayout(2, 2))
	if err != nil {
		log.Fatal(err)
	}

	pc := plot.PieChart{Title: "Some Pies"}
	pc.AddDataPair("Data1", []string{"2009", "2010", "2011"}, []float64{10, 20, 30})
	pc.Inner = 0.75
	pl1.Plot(&pc)

	piec := plot.PieChart{Title: "Some Pies"}
	piec.AddIntDataPair("Europe",
		[]string{"D", "AT", "CH", "F", "E", "I"},
		[]int{10, 20, 30, 35, 15, 25})
	piec.Data[0].Samples[3].Flag = true
	pl1.Plot(&piec)

	piec.Inner = 0.5
	piec.FmtVal = plot.AbsoluteValue
	pl1.Plot(&piec)

	piec.Inner = 0.65
	piec.Key.Cols = 2
	piec.FmtVal = plot.PercentValue
	plot.PieChartShrinkage = 0.45
	pl1.Plot(&piec)
	pl1.WriteToFile("piechart1")

	pl2, err := plot.NewPlotter(500, 400, plot.WithPlotterLayout(2, 1))
	if err != nil {
		log.Fatal(err)
	}
	pie := plot.PieChart{Title: "Some Pies"}
	data := []plot.CatValue{{"D", 10, false}, {"GB", 20, true}, {"CH", 30, false}, {"F", 60, false}}
	lw := 4
	red := plot.Style{LineColor: color.NRGBA{0xcc, 0x00, 0x00, 0xff}, FillColor: color.NRGBA{0xff, 0x80, 0x80, 0xff},
		LineStyle: plot.SolidLine, LineWidth: lw}
	green := plot.Style{LineColor: color.NRGBA{0x00, 0xcc, 0x00, 0xff}, FillColor: color.NRGBA{0x80, 0xff, 0x80, 0xff},
		LineStyle: plot.SolidLine, LineWidth: lw}
	blue := plot.Style{LineColor: color.NRGBA{0x00, 0x00, 0xcc, 0xff}, LineWidth: lw, LineStyle: plot.SolidLine, FillColor: color.NRGBA{0x80, 0x80, 0xff, 0xff}}
	pink := plot.Style{LineColor: color.NRGBA{0x99, 0x00, 0x99, 0xff}, LineWidth: lw, LineStyle: plot.SolidLine, FillColor: color.NRGBA{0xaa, 0x60, 0xaa, 0xff}}

	styles := []plot.Style{red, green, blue, pink}
	pie.FmtKey = plot.IntegerValue
	pie.AddData("Data1", data, styles)
	pie.Inner = 0
	pie.Key.Cols = 2
	pie.Key.Pos = "ibr"
	pl2.Plot(&pie)

	pie = plot.PieChart{Title: "Some Rings"}
	data2 := []plot.CatValue{{"D", 15, false}, {"GB", 25, false}, {"CH", 30, false}, {"F", 50, false}}
	data[1].Flag = false
	lw = 2
	lightred := plot.Style{LineColor: color.NRGBA{0xcc, 0x40, 0x40, 0xff}, FillColor: color.NRGBA{0xff, 0xc0, 0xc0, 0xff},
		LineStyle: plot.SolidLine, LineWidth: lw}
	lightgreen := plot.Style{LineColor: color.NRGBA{0x40, 0xcc, 0x40, 0xff}, FillColor: color.NRGBA{0xc0, 0xff, 0xc0, 0xff},
		LineStyle: plot.SolidLine, LineWidth: lw}
	lightblue := plot.Style{LineColor: color.NRGBA{0x40, 0x40, 0xcc, 0xff}, FillColor: color.NRGBA{0xc0, 0xc0, 0xff, 0xff},
		LineWidth: lw, LineStyle: plot.SolidLine}
	lightpink := plot.Style{LineColor: color.NRGBA{0xaa, 0x00, 0xaa, 0xff}, FillColor: color.NRGBA{0xff, 0x80, 0xff, 0xff},
		LineWidth: lw, LineStyle: plot.SolidLine}
	lightstyles := []plot.Style{lightred, lightgreen, lightblue, lightpink}

	pie.Inner = 0.3
	pie.Key.Cols = 2
	pie.Key.Pos = "ibr"
	pie.FmtVal = plot.PercentValue
	plot.PieChartShrinkage = 0.55
	pie.FmtKey = plot.IntegerValue

	pie.AddData("1980", data, styles)
	pie.AddData("2010", data2, lightstyles)
	pl2.Plot(&pie)

	pl2.WriteToFile("piechart2")
}
