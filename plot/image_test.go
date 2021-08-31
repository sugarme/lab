package plot

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"testing"
)

func marker(i *image.RGBA, x, y int) {
	red := color.RGBA{0xff, 0x00, 0x00, 0xff}
	n := 15
	for xx := x - n; xx <= x+n; xx++ {
		i.Set(xx, y, red)
	}
	for yy := y - n; yy <= y+n; yy++ {
		i.Set(x, yy, red)
	}
}

func TestText(t *testing.T) {
	g := NewImageGraphics(400, 200, color.RGBA{220, 220, 220, 255}, nil, nil)
	g.Text(2, 2, "Hallo (tl)", "tl", 0, Font{})
	g.Text(200, 2, "schöne (tc)", "tc", 0, Font{})
	g.Text(398, 2, "Welt (tr)", "tr", 0, Font{})

	g.Text(2, 100, "Hallo (cl)", "cl", 0, Font{})
	g.Text(200, 100, "schöne (cc)", "cc", 0, Font{})
	g.Text(398, 100, "Welt (cr)", "cr", 0, Font{})

	g.Text(2, 198, "Hallo (bl)", "bl", 0, Font{})
	g.Text(200, 198, "schöne (bc)", "bc", 0, Font{})
	g.Text(398, 198, "Welt (br)", "br", 0, Font{})

	marker(g.Image, 100, 50)
	marker(g.Image, 300, 50)
	marker(g.Image, 100, 150)
	marker(g.Image, 300, 150)

	g.Text(100, 50, "XcXcX", "cc", 45, Font{})
	g.Text(300, 50, "XbXlX", "bl", 90, Font{})
	g.Text(100, 150, "XbXcX", "bc", 90, Font{})
	g.Text(300, 150, "XbXrX", "br", 90, Font{})

	marker(g.Image, 200, 50)
	marker(g.Image, 200, 150)
	g.Text(200, 50, "HcHlH", "cl", 90, Font{})
	g.Text(200, 150, "HcHrH", "cr", 90, Font{})

	file, err := os.Create("text.png")
	if err != nil {
		log.Fatal(err)
	}
	png.Encode(file, g.Image)
	file.Close()
}
