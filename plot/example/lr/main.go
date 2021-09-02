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
	lrChart()
}

func lrChart() {
	file, err := os.Open("./find-lr.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	d := data.ReadCSV(file)
	lossRecs := d.Col("loss").Records()
	lrRecs := d.Col("lr").Records()

	var (
		lrs    []float64 = make([]float64, len(lossRecs))
		losses []float64 = make([]float64, len(lossRecs))
	)
	for i := 0; i < len(lossRecs); i++ {
		lr, err := strconv.ParseFloat(lrRecs[i], 64)
		if err != nil {
			log.Fatal(err)
		}
		l, err := strconv.ParseFloat(lossRecs[i], 64)
		if err != nil {
			log.Fatal(err)
		}

		lrs[i] = lr
		losses[i] = l
	}

	p, err := plot.NewPlotter(800, 600, plot.WithPlotterType("png"))
	if err != nil {
		log.Fatal(err)
	}

	// create scatter chart
	c := plot.ScatterChart{Title: "Find Learning Rate"}
	style1 := plot.Style{Symbol: '.', SymbolColor: color.NRGBA{0x00, 0x00, 0xff, 0xff}, LineStyle: plot.SolidLine}
	c.AddDataPair("Loss vs. LR", lrs, losses, plot.PlotStyleLinesPoints, style1)
	c.XRange.Log = true
	c.XRange.TicSetting.Grid = 0
	c.YRange.TicSetting.Grid = 0
	c.XRange.Label = "learning rate (log-scale)"
	c.YRange.Label = "losses"
	c.Key.Cols = 1
	c.Key.Pos = "itr"
	c.XRange.Min = 0
	c.YRange.Min = 0
	c.YRange.TicSetting.Format = yFmtFloat
	c.XRange.TicSetting.Format = xFmtFloat

	p.Plot(&c)
	p.WriteToFile("find-lr")
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

	return fmt.Sprintf("%.0e", f)
}
