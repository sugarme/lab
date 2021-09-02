package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"

	"github.com/sugarme/lab/plot"
)

var background color.RGBA = color.RGBA{0xff, 0xff, 0xff, 0xff}

func main() {
	textlen()
	testGraphics()
}

func textlen() {
	s2f, _ := os.Create("text.svg")
	mysvg := plot.NewSVG(s2f)
	mysvg.Start(1600, 800)
	mysvg.Title("My Plot")
	mysvg.Rect(0, 0, 2000, 800, "fill: #ffffff")
	background := color.RGBA{0xff, 0xff, 0xff, 0xff}
	sgr := plot.NewSvgGraphics(mysvg, 2000, 800, "Arial", 18, background)
	sgr.Begin()

	texts := []string{"ill", "WWW", "Some normal text.", "Illi, is. illigalli: ill!", "OO WORKSHOOPS OMWWW BMWWMB"}
	fonts := []string{"Arial", "Helvetica", "Times", "Courier" /* "Calibri", "Palatino" */}
	sizes := []plot.FontSize{plot.TinyFontSize, plot.SmallFontSize, plot.NormalFontSize, plot.LargeFontSize, plot.HugeFontSize}
	font := plot.Font{Color: color.NRGBA{0x00, 0x00, 0x00, 0xff}}

	df := plot.Font{Name: "Arial", Color: color.NRGBA{0x20, 0x20, 0xff, 0xff}, Size: -3}
	x, y := 20, 40
	for _, t := range texts {
		for _, f := range fonts {
			for _, s := range sizes {
				font.Name, font.Size = f, s
				tvl := sgr.TextLen(t, font)
				sgr.Text(x+tvl/2, y-2, t, "cc", 0, font)
				sgr.Line(x, y, x+tvl, y, plot.Style{LineColor: color.NRGBA{0xff, 0x00, 0x00, 0xff}, LineWidth: 2, LineStyle: plot.SolidLine})
				r := fmt.Sprintf("%s (%d)", f, s)
				sgr.Text(x+tvl+10, y-2, r, "cl", 0, df)
				y += 30
				if y > 760 {
					y = 40
					x += 300
				}
			}
		}
	}

	sgr.End()
	mysvg.End()
	s2f.Close()
}

//
// Test of graphic primitives
//
func testGraphics() {
	f, err := os.Create("test-graphics.svg")
	if err != nil {
		log.Fatal(err)
	}

	svg := plot.NewSVG(f)
	svg.Start(900, 416)

	img := image.NewRGBA(image.Rect(0, 0, 900, 416))

	igr := plot.AddTo(img, 0, 0, 900, 416, color.RGBA{0xff, 0xff, 0xff, 0xff}, nil, nil)
	sgr := plot.AddToSvg(svg, 0, 0, 900, 416, "", 14, color.RGBA{0xff, 0xff, 0xff, 0xff})

	style := plot.Style{LineWidth: 0, LineColor: color.NRGBA{0x00, 0x00, 0x00, 0xff}, LineStyle: plot.SolidLine}

	// Line Width
	x0, y0 := 10, 10
	for w := 1; w <= 10; w++ {
		style.LineWidth = w
		igr.Line(x0, y0, x0+160, y0, style)
		sgr.Line(x0, y0, x0+160, y0, style)
		y0 += w + 5
	}

	// Line Color
	style = plot.Style{LineWidth: 19, LineColor: color.NRGBA{0x80, 0x80, 0x80, 0xff}, LineStyle: plot.SolidLine}
	igr.Line(x0+40, y0-5, x0+40, y0+174, style)
	sgr.Line(x0+40, y0-5, x0+40, y0+174, style)
	igr.Line(x0+80, y0-5, x0+80, y0+174, style)
	sgr.Line(x0+80, y0-5, x0+80, y0+174, style)
	igr.Line(x0+120, y0-5, x0+120, y0+174, style)
	sgr.Line(x0+120, y0-5, x0+120, y0+174, style)

	style = plot.Style{LineWidth: 5, LineStyle: plot.SolidLine}
	for _, col := range []color.NRGBA{
		color.NRGBA{0x00, 0x00, 0x00, 0xff}, color.NRGBA{0xff, 0x00, 0x00, 0xff},
		color.NRGBA{0x00, 0xff, 0x00, 0xff}, color.NRGBA{0x00, 0x00, 0xff, 0xff},
		color.NRGBA{0xff, 0xff, 0x00, 0xff}, color.NRGBA{0xff, 0x00, 0xff, 0xff},
		color.NRGBA{0x00, 0xff, 0xff, 0xff},
		color.NRGBA{0x3f, 0x3f, 0x3f, 0xff}, color.NRGBA{0x7f, 0x7f, 0x7f, 0xff},
		color.NRGBA{0xbf, 0xbf, 0xbf, 0xff}, color.NRGBA{0xff, 0xff, 0xff, 0xff},
		color.NRGBA{0xcc, 0x00, 0x00, 0xff}, color.NRGBA{0x00, 0xbb, 0x00, 0xff},
		color.NRGBA{0x00, 0x00, 0xdd, 0xff}, color.NRGBA{0x99, 0x66, 0x00, 0xff},
		color.NRGBA{0xbb, 0x00, 0xbb, 0xff}, color.NRGBA{0x00, 0xaa, 0xaa, 0xff},
		color.NRGBA{0xaa, 0xaa, 0x00, 0xff},
	} {
		d := 0
		for _, a := range []uint8{0xff, 0xc0, 0x80, 0x40, 0x00} {
			c := col
			c.A = a
			style.LineColor = c
			igr.Line(x0+d, y0, x0+d+40, y0, style)
			sgr.Line(x0+d, y0, x0+d+40, y0, style)
			d += 40
		}
		y0 += 10
	}

	// Line Style
	style.LineColor = color.NRGBA{0x00, 0x00, 0x00, 0xff}
	style.LineWidth = 1
	for _, st := range []plot.LineStyle{
		plot.SolidLine, plot.DashedLine, plot.DottedLine, plot.DashDotDotLine,
		plot.LongDashLine, plot.LongDotLine,
	} {
		style.LineStyle = st
		igr.Line(x0, y0, x0+160, y0, style)
		sgr.Line(x0, y0, x0+160, y0, style)
		y0 += 5

	}
	style.LineWidth = 9
	y0 += 10
	for _, st := range []plot.LineStyle{
		plot.SolidLine, plot.DashedLine, plot.DottedLine, plot.DashDotDotLine,
		plot.LongDashLine, plot.LongDotLine,
	} {
		style.LineStyle = st
		igr.Line(x0, y0, x0+160, y0, style)
		sgr.Line(x0, y0, x0+160, y0, style)
		y0 += 12

	}

	// Text Alignment
	font := plot.Font{}
	rx, ry := 180, 10
	px, py := 450, 90
	text := "(JgbXÄj)"
	alignedText(igr, text, font, rx, ry, px, py, 0, 0)
	alignedText(sgr, text, font, rx, ry, px, py, 0, 0)

	font.Size = plot.HugeFontSize
	rx, ry = 180, 100
	px, py = 450, 180
	alignedText(igr, text, font, rx, ry, px, py, 0, 0)
	alignedText(sgr, text, font, rx, ry, px, py, 0, 0)

	font.Size = plot.TinyFontSize
	rx, ry = 180, 190
	px, py = 450, 270
	alignedText(igr, text, font, rx, ry, px, py, 0, 0)
	alignedText(sgr, text, font, rx, ry, px, py, 0, 0)

	// Rectangles
	x0, y0 = 180, 285
	dx, dy := 40, 30
	w, h := 30, 20
	n, m := 7, 4

	// background cross
	style = plot.Style{LineWidth: 19, LineColor: color.NRGBA{0x80, 0x80, 0x80, 0xff}, LineStyle: plot.SolidLine}
	igr.Line(x0, y0+2, x0+(n-1)*dx+w, y0+(m-1)*dy+h-2, style)
	sgr.Line(x0, y0+2, x0+(n-1)*dx+w, y0+(m-1)*dy+h-2, style)
	style = plot.Style{LineWidth: 19, LineColor: color.NRGBA{0x0, 0xd0, 0x0, 0xff}, LineStyle: plot.SolidLine}
	igr.Line(x0, y0+(m-1)*dy+h-2, x0+(n-1)*dx+w, y0+2, style)
	sgr.Line(x0, y0+(m-1)*dy+h-2, x0+(n-1)*dx+w, y0+2, style)

	for i := 1; i <= n*m; i++ {
		alpha := uint8(i * 256 / (n*m + 1))
		style := plot.Style{LineWidth: 3, LineColor: color.NRGBA{0xc0, 0x0, 0x0, alpha},
			LineStyle: plot.SolidLine, FillColor: color.NRGBA{0x0, 0x0, 0xc0, 0xff - alpha}}
		igr.Rect(x0, y0, w, h, style)
		sgr.Rect(x0, y0, w, h, style)
		if i%n == 0 {
			x0 = 180
			y0 += dy
		} else {
			x0 += dx
		}
	}

	// Symbols
	scols := []color.NRGBA{{0, 0, 0, 0xff}, {0xc0, 0x00, 0x00, 0xff}, {0x00, 0xc0, 0x00, 0xff},
		{0x00, 0x00, 0xc0, 0xff}, {0x80, 0x80, 0x80, 0xff}}

	dx, dy = 30, 27
	font = plot.Font{Color: color.NRGBA{0x00, 0x00, 0x00, 0xff}}
	for i, st := range []int{'o', '=', '%', '&', '+', 'X', '*', '0', '@', '#', 'A', 'W', 'V', 'Z', '.'} {
		igr.Text(470, 20+i*dy, fmt.Sprintf("%c", st), "cc", 0, font)
		sgr.Text(470, 20+i*dy, fmt.Sprintf("%c", st), "cc", 0, font)
	}
	for si, size := range []float64{0.5, 0.67, 1.0, 1.5, 2.0} {
		x0, y0 = 500+si*dx, 20
		for _, st := range []int{'o', '=', '%', '&', '+', 'X', '*', '0', '@', '#', 'A', 'W', 'V', 'Z', '.'} {
			style := plot.Style{Symbol: st, SymbolColor: scols[si], SymbolSize: size}
			igr.Symbol(x0, y0, style)
			sgr.Symbol(x0, y0, style)
			y0 += dy
		}
	}

	// Rotated text
	gray := color.NRGBA{0x80, 0x80, 0x80, 0xff}
	style = plot.Style{LineColor: gray, FillColor: gray}
	rx, ry = 675, 50
	px, py = 875, 200
	// igr.Rect(rx, ry, px-rx, py-ry, style)
	// sgr.Rect(rx, ry, px-rx, py-ry, style)
	text = "[##X##]"
	alignedText(igr, text, font, rx, ry, px, py, 20, 5)
	alignedText(sgr, text, font, rx, ry, px, py, 20, 5)

	x0, y0 = 775, 325
	black := color.NRGBA{0x00, 0x00, 0x00, 0xff}
	font.Color = black
	style.LineWidth, style.LineColor, style.LineStyle = 1, black, plot.SolidLine
	igr.Line(x0-100, y0, x0+100, y0, style)
	igr.Line(x0, y0-100, x0, y0+100, style)
	igr.Text(x0, y0, "abcABCxyz", "cc", 1, font)
	igr.Text(x0, y0, "abcABCxyz", "cc", 30, font)
	igr.Text(x0, y0, "abcABCxyz", "cc", 60, font)
	igr.Text(x0, y0, "abcABCxyz", "cc", 90, font)

	/*
		font.Color = color.NRGBA{0xee, 0x00, 0x00, 0xff}
		igr.Text(x0, y0, "abcABCxyz", "tl", 1, font)
		igr.Text(x0, y0, "abcABCxyz", "tl", 30, font)
		igr.Text(x0, y0, "abcABCxyz", "tl", 60, font)
		igr.Text(x0, y0, "abcABCxyz", "tl", 90, font)
		font.Color = color.NRGBA{0x11, 0xee, 0x11, 0xff}
		igr.Text(x0, y0, "abcABCxyz", "br", 1, font)
		igr.Text(x0, y0, "abcABCxyz", "br", 30, font)
		igr.Text(x0, y0, "abcABCxyz", "br", 60, font)
		igr.Text(x0, y0, "abcABCxyz", "br", 90, font)
	*/

	pngFile, err := os.Create("test-graphics.png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(pngFile, igr.Image)
	if err != nil {
		log.Fatal(err)
	}

	svg.End()
	sgr.End()
}

func alignedText(g plot.Graphics, text string, font plot.Font, rx, ry, px, py int, rot int, drot int) {
	mx, my := (rx+px)/2, (ry+py)/2
	var style plot.Style
	style.LineWidth, style.LineColor, style.LineStyle = 1, color.NRGBA{0xff, 0x00, 0x00, 0xff}, plot.SolidLine
	g.Line(rx, ry, px, ry, style)
	g.Line(px, ry, px, py, style)
	g.Line(px, py, rx, py, style)
	g.Line(rx, py, rx, ry, style)
	g.Line(mx, ry, mx, py, style)
	g.Line(rx, my, px, my, style)

	font.Color = color.NRGBA{0x00, 0x00, 0x00, 0xff}
	g.Text(rx, ry, text, "tl", rot, font)
	font.Color = color.NRGBA{0xff, 0x00, 0x00, 0xff}
	rot += drot
	g.Text(mx, ry, text, "tc", rot, font)
	font.Color = color.NRGBA{0x00, 0xff, 0x00, 0xff}
	rot += drot
	g.Text(px, ry, text, "tr", rot, font)
	font.Color = color.NRGBA{0x00, 0x00, 0xff, 0xff}
	rot += drot
	g.Text(rx, my, text, "cl", rot, font)
	font.Color = color.NRGBA{0xbb, 0xbb, 0x00, 0xff}
	rot += drot
	g.Text(mx, my, text, "cc", rot, font)
	font.Color = color.NRGBA{0xff, 0x00, 0xff, 0xff}
	rot += drot
	g.Text(px, my, text, "cr", rot, font)
	font.Color = color.NRGBA{0x00, 0xff, 0xff, 0xff}
	rot += drot
	g.Text(rx, py, text, "bl", rot, font)
	font.Color = color.NRGBA{0x60, 0x60, 0x60, 0xff}
	rot += drot
	g.Text(mx, py, text, "bc", rot, font)
	font.Color = color.NRGBA{0x00, 0x00, 0x00, 0xff}
	rot += drot
	g.Text(px, py, text, "br", rot, font)
}
