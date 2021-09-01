package plot

import (
	"bufio"
	"bytes"
	"fmt"
	"image/color"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// Plotter helps saving plots of size WxH in a NxM grid layout
// in several formats
type Plotter struct {
	N           int // number of charts on a row
	M           int // number of charts on a column
	W           int // chart width
	H           int // chart height
	Cnt         int // keeping number of charts been plotted
	GraphicType string
	Graphics    Graphics
	writer      *bufio.Writer
	buf         *bytes.Buffer
}

type PlotterOptions struct {
	N           int
	M           int
	GraphicType string
}

type PlotterOption func(*PlotterOptions)

func defaultPlotterOptions() *PlotterOptions {
	return &PlotterOptions{
		N:           1,
		M:           1,
		GraphicType: "png",
	}
}

// WithPlotterLayout adds layout option.
// - n: number of charts per row
// - m: number of charts per column
func WithPlotterLayout(n, m int) PlotterOption {
	return func(o *PlotterOptions) {
		o.N = n
		o.M = m
	}
}

// WithPlotterType adds graphic type options.
// Only 'png', 'text' or 'svg' types are supported.
func WithPlotterType(gtype string) PlotterOption {
	return func(o *PlotterOptions) {
		// TODO. add more 'jpg', 'gif'
		switch gtype {
		case "png":
			o.GraphicType = "png"
		case "text":
			o.GraphicType = "text"
		case "svg":
			o.GraphicType = "svg"

		default:
			err := fmt.Errorf("Unsupported graphic type (%s). Only support 'svg' or 'png' graphic types.", gtype)
			log.Fatal(err)
		}
	}
}

// NewPlotter creates a new plotter.
func NewPlotter(w, h int, opts ...PlotterOption) *Plotter {
	options := defaultPlotterOptions()
	for _, opt := range opts {
		opt(options)
	}

	var graphics Graphics
	var buf *bytes.Buffer = new(bytes.Buffer)
	writer := bufio.NewWriter(buf)
	bg := color.RGBA{0xff, 0xff, 0xff, 0xff}
	switch options.GraphicType {
	case "text":
		graphics = NewTextGraphics(100, 30)
	case "png":
		graphics = NewImageGraphics(options.N*w, options.M*h, bg, nil, nil)

	case "svg":
		sp := NewSVG(writer)
		graphics = NewSvgGraphics(sp, w, h, "", 12, bg)
		graphics.(*SvgGraphics).svg.Start(options.N*w, options.M*h)
	}

	return &Plotter{
		N:           options.N,
		M:           options.M,
		W:           w,
		H:           h,
		Cnt:         0,
		GraphicType: options.GraphicType,
		Graphics:    graphics,
		writer:      writer,
		buf:         buf,
	}
}

func (p *Plotter) Plot(c Chart) {
	c.Plot(p.Graphics)
	p.Cnt++
}

func (p *Plotter) WriteToFile(filename string) error {
	var fn string
	ext := strings.ToLower(filepath.Ext(filename))
	switch p.GraphicType {
	case "png":
		if ext == ".png" {
			fn = filename
		} else {
			fn = fmt.Sprintf("%s.png", filename)
		}
	case "text":
		if ext == ".txt" {
			fn = filename
		} else {
			fn = fmt.Sprintf("%s.txt", filename)
		}
	case "svg":
		if ext == ".svg" {
			fn = filename
		} else {
			fn = fmt.Sprintf("%s.svg", filename)
		}
	}

	f, err := os.Create(fn)
	if err != nil {
		err = fmt.Errorf("Plotter.WriteToFile - creating file %s failed.\n", fn)
		return err
	}

	switch p.GraphicType {
	case "png":
		return png.Encode(f, p.Graphics.(*ImageGraphics).Image)
	case "text":
		f.Write([]byte(p.Graphics.(*TextGraphics).String() + "\n\n\n"))
	case "svg":
		p.Graphics.(*SvgGraphics).svg.End()
		err := p.writer.Flush()
		if err != nil {
			return err
		}
		_, err = f.Write([]byte(p.buf.String()))
		if err != nil {
			return err
		}

	default:
		err := fmt.Errorf("Plotter.WriteToFile - Unsupported graphics type: %q\n", p.GraphicType)
		return err
	}

	return nil
}
