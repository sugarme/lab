package main

import (
	"image/color"
	"log"

	"github.com/sugarme/lab/plot"
)

func main() {
	barChart()
	categoricalBarChart()
}

//
// Bar Charts
//
func barChart() {
	pl, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(3, 2))
	if err != nil {
		log.Fatal(err)
	}

	red := plot.Style{Symbol: 'o', LineColor: color.NRGBA{0xcc, 0x00, 0x00, 0xff},
		FillColor: color.NRGBA{0xff, 0x80, 0x80, 0xff},
		LineStyle: plot.SolidLine, LineWidth: 2}
	green := plot.Style{Symbol: '#', LineColor: color.NRGBA{0x00, 0xcc, 0x00, 0xff},
		FillColor: color.NRGBA{0x80, 0xff, 0x80, 0xff},
		LineStyle: plot.SolidLine, LineWidth: 2}

	barc := plot.BarChart{Title: "Simple Bar Chart"}
	barc.Key.Hide = true
	barc.XRange.ShowZero = true
	barc.AddDataPair("Amount",
		[]float64{-10, 10, 20, 30, 35, 40, 50},
		[]float64{90, 120, 180, 205, 230, 150, 190}, red)
	pl.Plot(&barc)
	barc.XRange.TicSetting.Delta = 0

	barc = plot.BarChart{Title: "Simple Bar Chart"}
	barc.Key.Hide = true
	barc.XRange.ShowZero = true
	barc.AddDataPair("Test", []float64{-5, 15, 25, 35, 45, 55}, []float64{110, 80, 95, 80, 120, 140}, green)
	pl.Plot(&barc)
	barc.XRange.TicSetting.Delta = 0

	barc.YRange.TicSetting.Delta = 0
	barc.Title = "Combined (ugly as bar positions do not match)"
	barc.AddDataPair("Amount", []float64{-10, 10, 20, 30, 35, 40, 50}, []float64{90, 120, 180, 205, 230, 150, 190}, red)
	pl.Plot(&barc)

	barc.Title = "Stacked (still ugly)"
	barc.Stacked = true
	pl.Plot(&barc)

	barc = plot.BarChart{Title: "Nicely Stacked"}
	barc.Key.Hide = true
	barc.XRange.Fixed(0, 60, 10)
	barc.AddDataPair("A", []float64{10, 30, 40, 50}, []float64{110, 95, 60, 120}, red)
	barc.AddDataPair("B", []float64{10, 30, 40, 50}, []float64{40, 130, 15, 100}, green)
	pl.Plot(&barc)

	barc.Stacked = true
	pl.Plot(&barc)

	pl.WriteToFile("bar-chart")
}

//
// Categorical Bar Charts
//
func categoricalBarChart() {
	pl1, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(3, 2))
	if err != nil {
		log.Fatal(err)
	}

	x := []float64{0, 1, 2, 3}
	europe := []float64{10, 15, 25, 20}
	asia := []float64{15, 30, 10, 20}
	africa := []float64{20, 5, 5, 5}
	blue := plot.Style{Symbol: '#', LineColor: color.NRGBA{0x00, 0x00, 0xff, 0xff}, LineWidth: 4, FillColor: color.NRGBA{0x40, 0x40, 0xff, 0xff}}
	green := plot.Style{Symbol: 'x', LineColor: color.NRGBA{0x00, 0xaa, 0x00, 0xff}, LineWidth: 4, FillColor: color.NRGBA{0x40, 0xff, 0x40, 0xff}}
	pink := plot.Style{Symbol: '0', LineColor: color.NRGBA{0x99, 0x00, 0x99, 0xff}, LineWidth: 4, FillColor: color.NRGBA{0xaa, 0x60, 0xaa, 0xff}}
	red := plot.Style{Symbol: '%', LineColor: color.NRGBA{0xcc, 0x00, 0x00, 0xff}, LineWidth: 4, FillColor: color.NRGBA{0xff, 0x40, 0x40, 0xff}}

	// Categorized Bar Chart
	c := plot.BarChart{Title: "Income"}
	c.XRange.Category = []string{"none", "low", "average", "high"}

	// Unstacked, different labelings
	c.ShowVal = 1
	c.AddDataPair("Europe", x, europe, blue)
	pl1.Plot(&c)

	c.ShowVal = 2
	c.AddDataPair("Asia", x, asia, pink)
	pl1.Plot(&c)

	c.ShowVal = 3
	c.AddDataPair("Africa", x, africa, green)
	pl1.Plot(&c)

	// Stacked with different labelings
	c.Stacked = true
	c.ShowVal = 1
	pl1.Plot(&c)

	c.ShowVal = 2
	pl1.Plot(&c)

	c.ShowVal = 3
	pl1.Plot(&c)
	pl1.WriteToFile("category-bar1")

	// Including negative ones
	pl2, err := plot.NewPlotter(400, 300, plot.WithPlotterLayout(3, 2))
	if err != nil {
		log.Fatal(err)
	}

	c = plot.BarChart{Title: "Income"}
	c.XRange.Category = []string{"none", "low", "average", "high"}
	c.Key.Hide = true
	c.YRange.ShowZero = true
	c.ShowVal = 3

	c.AddDataPair("Europe", x, []float64{-10, -15, -20, -5}, blue)
	pl2.Plot(&c)

	c.AddDataPair("Asia", x, []float64{-15, -10, -5, -20}, pink)
	pl2.Plot(&c)

	c.Stacked = true
	pl2.Plot(&c)

	// Mixed
	c = plot.BarChart{Title: "Income"}
	c.XRange.Category = []string{"none", "low", "average", "high"}
	c.Key.Hide = true
	c.YRange.ShowZero = true
	c.ShowVal = 3

	c.AddDataPair("Europe", x, []float64{-10, 15, -20, 5}, blue)
	pl2.Plot(&c)

	c.AddDataPair("Asia", x, []float64{-15, 10, -5, 20}, pink)
	pl2.Plot(&c)

	c.Stacked = true
	pl2.Plot(&c)

	// Very Mixed
	c = plot.BarChart{Title: "Income"}
	c.XRange.Category = []string{"none", "low", "average", "high"}
	c.Key.Hide = true
	c.YRange.ShowZero = true
	c.ShowVal = 3

	c.AddDataPair("Europe", x, []float64{-10, 15, -20, 5}, blue)
	c.AddDataPair("Asia", x, []float64{-15, 10, 5, 20}, pink)
	c.AddDataPair("Africa", x, []float64{10, -10, 15, -5}, green)
	pl2.Plot(&c)

	c.Stacked = true
	pl2.Plot(&c)

	c.AddDataPair("America", x, []float64{15, -5, -10, -20}, red)
	c.YRange.TicSetting.Delta = 0
	pl2.Plot(&c)

	pl2.WriteToFile("category-bar2")
}
