package draw

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	N     = 8
	Scale = 1
)

func TestFindContour(t *testing.T) {
	err := makeImageMask()
	if err != nil {
		t.Fatal(err)
	}

}

func makeImageMask() error {
	// Initialize the graphic context on an RGBA image
	var (
		w int = 472
		h int = 720
	)

	shapes := [][]Point{
		{
			{43.262773722627685, 345.98540145985396},
			{43.99270072992698, 375.91240875912405},
			{49.832116788321116, 397.81021897810217},
			{44.722627737226276, 419.70802919708024},
			{47.642335766423344, 442.33576642335765},
			{60.781021897810206, 456.20437956204375},
			{76.10948905109484, 459.8540145985401},
			{87.05839416058393, 451.82481751824815},
			{92.89781021897807, 424.81751824817513},
			{95.81751824817513, 403.6496350364963},
			{103.11678832116786, 399.2700729927007},
			{108.956204379562, 413.1386861313868},
			{115.52554744525543, 432.8467153284671},
			{111.87591240875906, 451.82481751824815},
			{120.63503649635032, 464.2335766423357},
			{135.96350364963502, 462.7737226277372},
			{155.6715328467153, 452.5547445255474},
			{173.18978102189777, 445.98540145985396},
			{180.4890510948905, 440.87591240875906},
			{184.8686131386861, 438.6861313868613},
			{187.05839416058393, 425.54744525547443},
			{189.24817518248176, 417.51824817518246},
			{196.54744525547437, 413.8686131386861},
			{203.11678832116786, 414.5985401459854},
			{200.92700729927003, 424.0875912408759},
			{198.00729927007296, 429.92700729927003},
			{208.956204379562, 434.30656934306563},
			{227.934306569343, 427.7372262773722},
			{234.50364963503648, 416.05839416058393},
			{230.12408759124082, 399.99999999999994},
			{221.36496350364962, 395.62043795620434},
			{214.0656934306569, 387.5912408759124},
			{214.0656934306569, 379.5620437956204},
			{208.22627737226276, 362.7737226277372},
			{190.70802919708024, 351.82481751824815},
			{179.7591240875912, 345.98540145985396},
			{168.81021897810217, 340.8759124087591},
			{150.5620437956204, 335.0364963503649},
			{138.1532846715328, 330.6569343065693},
			{130.12408759124082, 310.94890510948903},
			{133.04379562043795, 294.8905109489051},
			{140.34306569343062, 281.75182481751824},
			{141.0729927007299, 269.3430656934306},
			{125.01459854014593, 262.04379562043795},
			{116.98540145985396, 267.15328467153284},
			{115.52554744525543, 278.1021897810219},
			{103.8467153284671, 289.0510948905109},
			{95.81751824817513, 302.91970802919707},
			{92.89781021897807, 311.67883211678827},
			{89.2481751824817, 323.3576642335766},
			{87.78832116788317, 330.6569343065693},
			{79.02919708029196, 335.0364963503649},
			{71.72992700729924, 339.41605839416053},
			{69.54014598540141, 340.1459854014598},
			{62.24087591240874, 341.60583941605836},
			{62.24087591240874, 341.60583941605836},
			{56.40145985401455, 342.33576642335765},
		},
		{
			{66.62043795620434, 520.4379562043795},
			{57.86131386861308, 548.1751824817518},
			{74.6496350364963, 568.6131386861314},
			{87.78832116788317, 575.1824817518248},
			{97.27737226277367, 562.7737226277371},
			{93.62773722627736, 539.4160583941606},
			{79.02919708029196, 525.5474452554744},
		},
	}

	// Draw a closed shape
	maskCanvas := image.NewRGBA(image.Rect(0, 0, w, h))
	gc := NewGraphicContext(maskCanvas)
	gc.SetFillColor(color.White)
	gc.SetStrokeColor(color.White)
	gc.SetLineWidth(1)
	gc.BeginPath() // Initialize a new path
	for _, points := range shapes {
		gc.MoveTo(points[0].X, points[0].Y) // Move to a position to start the new path
		for i := 1; i < len(points); i++ {
			x := points[i].X
			y := points[i].Y
			gc.LineTo(x, y)
		}
		gc.Close()
	}
	gc.FillStroke()

	// Draw contour
	z := 0.5
	gc.SetFillColor(color.RGBA{0, 0, 0, 0})
	gc.SetLineWidth(1)
	contours := FromImage(maskCanvas).Contours(z)
	gc.BeginPath() // Initialize a new path
	for _, points := range contours {
		// contour
		gc.SetStrokeColor(color.RGBA{0, 255, 0, 255})
		gc.MoveTo(points[0].X, points[0].Y) // Move to a position to start the new path
		for _, p := range points {
			gc.LineTo(p.X, p.Y)
		}
		gc.Close()
		gc.FillStroke()

		// bounding box
		gc.SetStrokeColor(color.RGBA{255, 0, 0, 255})
		rec := bbox(points)
		gc.MoveTo(rec[0].X, rec[0].Y)
		for _, p := range rec {
			gc.LineTo(p.X, p.Y)
		}
		gc.Close()
		gc.FillStroke()
	}

	// Save mask
	maskFile := "./mask.png"
	err := saveImage(maskCanvas, maskFile)
	if err != nil {
		err = fmt.Errorf("save image mask failed: %w\n", err)
		return err
	}

	return nil
}

func saveImage(img image.Image, filename string, extOpt ...string) error {
	ext := strings.ToLower(filepath.Ext(filename))
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	if len(extOpt) > 0 {
		ext = extOpt[0]
	}

	defer f.Close()
	switch ext {
	case ".jpg", ".jpeg":
		return jpeg.Encode(f, img, nil)
	case ".png":
		return png.Encode(f, img)
	case ".gif":
		return gif.Encode(f, img, nil)

	default:
		err := fmt.Errorf("saveImage - unsupported image format %q\n", ext)
		return err
	}
}

func bbox(contour []Point) []Point {
	var (
		xs []float64
		ys []float64
	)

	for _, p := range contour {
		xs = append(xs, p.X)
		ys = append(ys, p.Y)
	}

	left, right := minmax(xs)
	top, bottom := minmax(ys)

	return []Point{
		{left, top}, {right, top}, {right, bottom}, {left, bottom},
	}
}

func minmax(vals []float64) (float64, float64) {
	var (
		max float64 = 0
		min float64 = math.Inf(1)
	)
	for _, v := range vals {
		if v < min {
			min = v
		}

		if v > max {
			max = v
		}
	}
	return min, max
}
