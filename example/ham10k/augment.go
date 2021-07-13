package main

import (
	"github.com/sugarme/gotch/vision/aug"
)

func CustomAugment() (aug.Transformer, error) {
	normOpt := aug.WithNormalize(aug.WithNormalizeMean([]float64{0.485, 0.456, 0.406}), aug.WithNormalizeStd([]float64{0.229, 0.224, 0.225}))

	return aug.Compose(normOpt)
}

func HamAugment()(aug.Transformer, error){
	hFlipOpt := aug.WithRandomHFlip(0.5)
	vFlipOpt := aug.WithRandomVFlip(0.5)
	rotateOpt := aug.WithRandRotate(0, 90)
	// normOpt := aug.WithNormalize(aug.WithNormalizeMean([]float64{0.485, 0.456, 0.406}), aug.WithNormalizeStd([]float64{0.229, 0.224, 0.225}))

	// return aug.Compose(hFlipOpt, rotateOpt,normOpt)
	return aug.Compose(hFlipOpt, vFlipOpt, rotateOpt)
}

func ValidAugment() (aug.Transformer, error) {
	normOpt := aug.WithNormalize(aug.WithNormalizeMean([]float64{0.485, 0.456, 0.406}), aug.WithNormalizeStd([]float64{0.229, 0.224, 0.225}))

	return aug.Compose(normOpt)
}
