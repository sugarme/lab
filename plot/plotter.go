package plot

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strings"
)

type Plotter interface {
	Plot(c Chart)
	WriteToFile(filename string) error
}

type plotter struct {
	n   int // num of charts on a row
	m   int // num of charts on a column
	w   int // chart width
	h   int // chart height
	cnt int // counting number plotted charts
}

func newPlotter(n, m, w, h int) *plotter {
	return &plotter{n, m, w, h, 0}
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

// ImagePlotter:
// =============

type ImagePlotter struct {
	plotter *plotter
	img     *image.RGBA
}

// NewImagePlotter creates a new ImagePlotter.
func NewImagePlotter(w, h int, opts ...PlotterOption) *ImagePlotter {
	o := defaultPlotterOptions()
	for _, opt := range opts {
		opt(o)
	}

	plotter := newPlotter(o.N, o.M, w, h)

	img := image.NewRGBA(image.Rect(0, 0, o.N*w, o.M*h))
	bg := image.NewUniform(color.RGBA{0xff, 0xff, 0xff, 0xff})
	draw.Draw(img, img.Bounds(), bg, image.ZP, draw.Src)

	return &ImagePlotter{
		plotter: plotter,
		img:     img,
	}
}

func (p *ImagePlotter) Plot(c Chart) {
	w := p.plotter.w
	h := p.plotter.h
	n := p.plotter.n
	cnt := p.plotter.cnt
	row, col := cnt/n, cnt%n
	igr := AddTo(p.img, col*w, row*h, w, h, color.RGBA{0xff, 0xff, 0xff, 0xff}, nil, nil)
	c.Plot(igr)
	p.plotter.cnt++
}

func (p *ImagePlotter) WriteToFile(filename string) error {
	var fn string
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == ".png" {
		fn = filename
	} else {
		fn = fmt.Sprintf("%s.png", filename)
	}
	f, err := os.Create(fn)
	if err != nil {
		err = fmt.Errorf("ImagePlotter.WriteToFile - creating file %s failed.\n", fn)
		return err
	}
	defer f.Close()

	return png.Encode(f, p.img)
}

// TextPlotter:
// ============

type TextPlotter struct {
	plotter *plotter
	buf     *bytes.Buffer
	writer  *bufio.Writer
}

// NewTextPlotter creates new TextPlotter
func NewTextPlotter(w, h int, opts ...PlotterOption) *TextPlotter {
	o := defaultPlotterOptions()
	for _, opt := range opts {
		opt(o)
	}

	plotter := newPlotter(o.N, o.M, w, h)
	var buf *bytes.Buffer = new(bytes.Buffer)
	writer := bufio.NewWriter(buf)

	return &TextPlotter{
		plotter: plotter,
		buf:     buf,
		writer:  writer,
	}
}

func (p *TextPlotter) Plot(c Chart) {
	tgr := NewTextGraphics(100, 30)
	c.Plot(tgr)
	p.writer.Write([]byte(tgr.String() + "\n\n\n"))
}

func (p *TextPlotter) WriteToFile(filename string) error {
	var fn string
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == ".txt" {
		fn = filename
	} else {
		fn = fmt.Sprintf("%s.txt", filename)
	}
	f, err := os.Create(fn)
	if err != nil {
		err = fmt.Errorf("TextPlotter.WriteToFile - creating file %s failed.\n", fn)
		return err
	}
	defer f.Close()

	err = p.writer.Flush()
	if err != nil {
		return err
	}
	_, err = f.Write([]byte(p.buf.String()))
	return err
}

// SvgPlotter:
// ===========

type SvgPlotter struct {
	plotter  *plotter
	buf      *bytes.Buffer
	writer   *bufio.Writer
	graphics *SvgGraphics
}

// NewSvgPlotter creates new SvgPlotter
func NewSvgPlotter(w, h int, opts ...PlotterOption) *SvgPlotter {
	o := defaultPlotterOptions()
	for _, opt := range opts {
		opt(o)
	}

	plotter := newPlotter(o.N, o.M, w, h)
	var buf *bytes.Buffer = new(bytes.Buffer)
	writer := bufio.NewWriter(buf)
	sp := NewSVG(writer)
	bg := color.RGBA{0xff, 0xff, 0xff, 0xff}
	graphics := NewSvgGraphics(sp, w, h, "", 12, bg)
	graphics.svg.Start(o.N*w, o.M*h)

	return &SvgPlotter{
		plotter:  plotter,
		buf:      buf,
		writer:   writer,
		graphics: graphics,
	}
}

func (p *SvgPlotter) Plot(c Chart) {
	w := p.plotter.w
	h := p.plotter.h
	n := p.plotter.n
	cnt := p.plotter.cnt
	row, col := cnt/n, cnt%n
	bg := color.RGBA{0xff, 0xff, 0xff, 0xff}
	sgr := AddToSvg(p.graphics.svg, col*w, row*h, w, h, "", 12, bg)
	c.Plot(sgr)
	p.plotter.cnt++
}

func (p *SvgPlotter) WriteToFile(filename string) error {
	var fn string
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == ".svg" {
		fn = filename
	} else {
		fn = fmt.Sprintf("%s.svg", filename)
	}
	f, err := os.Create(fn)
	if err != nil {
		err = fmt.Errorf("SvgPlotter.WriteToFile - creating file %s failed.\n", fn)
		return err
	}
	defer f.Close()
	p.graphics.svg.End()
	err = p.writer.Flush()
	if err != nil {
		return err
	}
	_, err = f.Write([]byte(p.buf.String()))
	return err
}

// NewPlotter creates new Plotter.
func NewPlotter(w, h int, opts ...PlotterOption) (Plotter, error) {
	o := defaultPlotterOptions()
	for _, opt := range opts {
		opt(o)
	}

	gtype := o.GraphicType
	switch gtype {
	case "png":
		return NewImagePlotter(w, h, opts...), nil
	case "text":
		return NewTextPlotter(w, h, opts...), nil
	case "svg":
		return NewSvgPlotter(w, h, opts...), nil
	default:
		err := fmt.Errorf("NewPlotter: unsupported graphic type %q\n", gtype)
		return nil, err
	}
}
