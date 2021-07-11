package main

import (
	"github.com/sugarme/gotch/vision/aug"
)

func CustomAugment() (aug.Transformer, error) {
	normOpt := aug.WithNormalize(aug.WithNormalizeMean([]float64{0.485, 0.456, 0.406}), aug.WithNormalizeStd([]float64{0.229, 0.224, 0.225}))

	return aug.Compose(normOpt)
}
