package draw

import (
	"bufio"
	"image"
	"image/draw"
	"image/png"
	"os"
)

func imageToGray16(im image.Image) *image.Gray16 {
	dst := image.NewGray16(im.Bounds())
	draw.Draw(dst, im.Bounds(), im, image.ZP, draw.Src)
	return dst
}

// SaveToPngFile create and save an image to a file using PNG format
func SaveToPngFile(filePath string, m image.Image) error {
	// Create the file
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	// Create Writer from file
	b := bufio.NewWriter(f)
	// Write the image into the buffer
	err = png.Encode(b, m)
	if err != nil {
		return err
	}
	err = b.Flush()
	if err != nil {
		return err
	}
	return nil
}

// LoadFromPngFile Open a png file
func LoadFromPngFile(filePath string) (image.Image, error) {
	// Open file
	f, err := os.OpenFile(filePath, 0, 0)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b := bufio.NewReader(f)
	img, err := png.Decode(b)
	if err != nil {
		return nil, err
	}
	return img, nil
}
