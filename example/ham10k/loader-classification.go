package main

import (
	"fmt"
	"reflect"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/dutil"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
	"github.com/sugarme/gotch/vision/aug"
	"github.com/sugarme/lab"
)

type SkinSample struct {
	Filename string
	Label    int64
}

type SkinDataset struct {
	Samples []SkinSample
	Config lab.Transform
	Transformer aug.Transformer
	IsTrain bool
}

func (m *SkinDataset) Item(idx int) (interface{}, error) {
	sample := m.Samples[idx]
	var(
		imgTs *ts.Tensor
		err error
	) 
	
	resize := m.Config.ResizeTo
	switch len(resize){
	case 2:
		imgTs, err = vision.LoadAndResize(sample.Filename, resize[0], resize[1])
		if err != nil {
			err := fmt.Errorf("SkinDataset.Item load image failed: %w\n", err)
			return nil, err
		}

	case 1:
		imgTs, err = vision.LoadAndResize(sample.Filename, resize[0], resize[0])
		if err != nil {
			err := fmt.Errorf("SkinDataset.Item load image failed: %w\n", err)
			return nil, err
		}

	default:
		imgTs, err = vision.Load(sample.Filename)
		if err != nil {
			err := fmt.Errorf("SkinDataset.Item load image failed: %w\n", err)
			return nil, err
		}
	}

	imgTs0 := imgTs.MustTo(gotch.CudaIfAvailable(), true)

	var img *ts.Tensor
	if m.Transformer != nil{
		img = m.Transformer.Transform(imgTs0)
	} else {
		img = imgTs0.MustShallowClone()
	}
	imgTs0.MustDrop()

	imgTs1 := img.MustDiv1(ts.FloatScalar(255.0), true).MustTotype(gotch.Float, true)
	labelTs := ts.MustOfSlice([]int64{sample.Label}).MustTotype(gotch.Int64, true)

	return []ts.Tensor{*imgTs1, *labelTs}, nil
}

func (m *SkinDataset) Len() int {
	return len(m.Samples)
}

func (m *SkinDataset) DType() reflect.Type {
	return reflect.TypeOf(m.Samples)
}

func NewSkinDataset(data []SkinDz, cfg *lab.Config, isTrain bool) (dutil.Dataset, error){
	var samples []SkinSample
	for _, rec := range data {
		filename := rec.File
		label := rec.ClassID
		samples = append(samples, SkinSample{filename, int64(label)})
	}

	var transformer aug.Transformer
	var err error
	if isTrain{
		switch cfg.Transform.Augment{
		case "RandAugment":
			transformer, err = lab.NewRandomAugment(lab.WithRandomAugmentNval(cfg.Transform.Params.N), lab.WithRandomAugmentMval(cfg.Transform.Params.M))
			if err != nil {
				return nil, err
			}

		case "CustomAugment":
			transformer, err = CustomAugment()
			if err != nil {
				return nil, err
			}
		case "ResNetAugment":
			transformer, err = ResNetAugment()
			if err != nil {
				return nil, err
			}

		case "NoAugment":
			return nil, nil

		default:
			fmt.Printf("No transform method found...\n")
		}
	} else { 
		switch cfg.Transform.Augment{
			case "NormAugment":
				transformer, err = NormAugment()
				if err != nil {
					return nil, err
				}
			case "NoAugment":
				transformer, err = NoAugment()
				if err != nil {
					return nil, err
				}
			default:
				fmt.Printf("No transform method found...\n")
		}
	}

	return &SkinDataset{
		Samples: samples,
		Config: cfg.Transform,
		Transformer: transformer,
		IsTrain: isTrain,
	}, nil
}
