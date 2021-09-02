package main

import (
	"fmt"
	"image/color"
	"log"
	"os"
	"strconv"

	"github.com/sugarme/lab/data"
	"github.com/sugarme/lab/plot"
)

func main() {
	lossChart()
}

func lossChart() {
	tfile, err := os.Open("./train-loss-1.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer tfile.Close()

	d := data.ReadCSV(tfile)
	losses := d.Col("loss").Records()
	epochs := d.Col("epoch").Records()

	var (
		tsteps  []float64 = make([]float64, len(losses))
		tlosses []float64 = make([]float64, len(losses))
	)
	for i := 0; i < len(losses); i++ {
		e, err := strconv.Atoi(epochs[i])
		if err != nil {
			log.Fatal(err)
		}
		l, err := strconv.ParseFloat(losses[i], 64)
		if err != nil {
			log.Fatal(err)
		}

		tsteps[i] = float64(e)
		tlosses[i] = l
	}

	vfile, err := os.Open("./valid-loss-1.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer vfile.Close()

	d = data.ReadCSV(vfile)
	losses = d.Col("loss").Records()
	epochs = d.Col("epoch").Records()

	var (
		vsteps  []float64 = make([]float64, len(losses))
		vlosses []float64 = make([]float64, len(losses))
	)
	for i := 0; i < len(losses); i++ {
		e, err := strconv.Atoi(epochs[i])
		if err != nil {
			log.Fatal(err)
		}
		l, err := strconv.ParseFloat(losses[i], 64)
		if err != nil {
			log.Fatal(err)
		}

		vsteps[i] = float64(e)
		vlosses[i] = l
	}

	p, err := plot.NewPlotter(800, 600, plot.WithPlotterType("png"))
	if err != nil {
		log.Fatal(err)
	}

	// create scatter chart
	c := plot.ScatterChart{Title: "Train & Valid Loss"}
	style1 := plot.Style{Symbol: '.', SymbolColor: color.NRGBA{0x00, 0x00, 0xff, 0xff}, LineStyle: plot.SolidLine}
	style2 := plot.Style{Symbol: '.', SymbolColor: color.NRGBA{0xff, 0x00, 0x00, 0xff}, LineStyle: plot.SolidLine}
	c.AddDataPair("Train", tsteps, tlosses, plot.PlotStyleLinesPoints, style1)
	c.AddDataPair("Valid", vsteps, vlosses, plot.PlotStyleLinesPoints, style2)
	c.XRange.TicSetting.Grid = 0
	c.YRange.TicSetting.Grid = 0
	c.XRange.Label = "epochs"
	c.YRange.Label = "losses"
	c.Key.Cols = 1
	c.Key.Pos = "itr"
	c.XRange.Min = 0
	c.YRange.Min = 0
	c.YRange.TicSetting.Format = yFmtFloat
	c.XRange.TicSetting.Format = xFmtFloat

	p.Plot(&c)
	p.WriteToFile("loss")
}

func yFmtFloat(f float64) string {
	switch {
	case f == 0:
		return "0"
	case f < 0:
		return ""
	default:
		return fmt.Sprintf("%3.1f", f)
	}
}

func xFmtFloat(f float64) string {
	if f < 0 {
		return ""
	}

	return fmt.Sprintf("%.0f", f)
}
