package lab

import (
	"fmt"
	"math/rand"
	"sort"
	"time"

	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision/aug"
	"gonum.org/v1/gonum/stat/distuv"
)

// TransformOP is an interface to create a image transformer.
type TransformOp interface{
	Transform(imgTs *ts.Tensor, v float64) *ts.Tensor
}

type OpConfig struct{
	OpName string
	MinVal float64
	MaxVal float64
}

var optConfigs []OpConfig = []OpConfig{
	{"Equalize", 0, 1},
	{"Posterize", 0, 4},
	{"Color", 0.1, 1.9},
	{"Brightness", 0.1, 1.9},
	{"Contrast", 0.1, 1.9},
	{"Sharpness", 0.1, 1.9},
	{"Rotate", 0, 30},
	{"Cutout", 0, 0.6},
	{"Solarize", 0, 256},
	// {"SolarizeAdd", 0, 110},
	{"Invert", 0, 1},
	{"TranslateX", 0, 0.3},
	{"TranslateY", 0, 0.3},
	{"ShearX", 0, 0.3},
	{"ShearY", 0, 0.3},

	// {"Downsample", 0, 1},
	// {"ZoomIn", 0, 0.5},
	// {"ZoomOut", 0, 0.5},
	// {"AutoContrast", 0, 1},
}

// Make creates transformer option from specified input value.
func(c *OpConfig) Make(v float64) (aug.Option, error){
	if err := c.validate(v); err != nil{
		return nil, err
	}

	switch c.OpName{
	case "AutoContrast":
		return aug.WithRandomAutocontrast(v), nil
	case "Equalize":
		return aug.WithRandomEqualize(v), nil
	case "Invert":
		return aug.WithRandomInvert(v), nil
	case "Downsample": // resize image to half
		// TODO: resize downsample
		// return aug.WithResize(h/2, w/2), nil
	case "ZoomIn":
		// TODO: resize crop image to v*w and v*h
	case "ZoomOut":
		// TODO: resize pad image to v*w and v*h
	case "Rotate":
		// return aug.WithRotate(v), nil
		return aug.WithRandomAffine(aug.WithAffineDegree([]int64{0, int64(v)})), nil
	case "Posterize":
		return aug.WithRandomPosterize(aug.WithPosterizeBits(uint8(v))), nil
	case "Solarize":
		return aug.WithRandomSolarize(), nil
	case "SolarizeAdd":
		return aug.WithRandomSolarize(aug.WithSolarizeThreshold(v)), nil
	case "Color":
		// TODO: is saturation?
		return aug.WithColorJitter(aug.WithColorSaturation([]float64{v, v})), nil
	case "Contrast":
		return aug.WithColorJitter(aug.WithColorContrast([]float64{v, v})), nil
	case "Brightness":
		return aug.WithColorJitter(aug.WithColorBrightness([]float64{v, v})), nil
	case "Sharpness":
		return aug.WithRandomAdjustSharpness(aug.WithSharpnessFactor(v)), nil
	case "ShearX":
		return aug.WithRandomAffine(aug.WithAffineShear([]float64{v, 1, 0, 0, 0, 1})), nil
	case "ShearY":
		return aug.WithRandomAffine(aug.WithAffineShear([]float64{0, 1, 0, v, 0, 1})), nil
	case "Cutout":
		return aug.WithRandomCutout(aug.WithCutoutValue([]int64{127, 127, 127}), aug.WithCutoutRatio([]float64{0.0001, v/2.0})), nil
	case "CutoutAbs":
		// TODO. For now, just the same as Cutout 
		return aug.WithRandomCutout(aug.WithCutoutValue([]int64{127, 127, 127}), aug.WithCutoutRatio([]float64{0.0001, v/2.0})), nil
	case "TranslateX":
		return aug.WithRandomAffine(aug.WithAffineTranslate([]float64{v, 0})), nil
	case "TranslateY":
		return aug.WithRandomAffine(aug.WithAffineTranslate([]float64{0, v})), nil

	default:
		err := fmt.Errorf("Unsupported transformer option: %s\n", c.OpName)
		return nil, err
	} 

	return nil, nil
}

func (c *OpConfig) validate(v float64) error{
	if v >= c.MinVal && v <= c.MaxVal{
		return nil
	}

	err := fmt.Errorf("Invalid input value. Expected in range [%.2f, %.2f], got %.2f\n", c.MinVal, c.MaxVal, v)
	return err
}

type RandomAugmentOptions struct{
	N int
	M int
	Normalize bool
}

type RandomAugmentOption func(*RandomAugmentOptions)

func defaultRandomAugmentOptions() *RandomAugmentOptions{
	return &RandomAugmentOptions{
		N: 3,
		M: 12,
		Normalize: true,
	}
}

func WithRandomAugmentNval(v int) RandomAugmentOption{
	return func(o *RandomAugmentOptions){
		o.N = v
	}
}

func WithRandomAugmentMval(v int) RandomAugmentOption{
	return func(o *RandomAugmentOptions){
		o.M = v
	}
}

func WithNormalize(v bool) RandomAugmentOption{
	return func(o *RandomAugmentOptions){
		o.Normalize = v
	}
}


// NewRandomAugment randomly select n augmentation options from a list of augmentation options
// to compose Transformer.
//
// Ref.
// https://raw.githubusercontent.com/ildoonet/pytorch-randaugment/master/RandAugment/augmentations.py
// https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
func NewRandomAugment(opts ...RandomAugmentOption) (aug.Transformer, error){
	options := defaultRandomAugmentOptions()
	for _, opt := range opts{
		opt(options)
	}

	var augOpts []aug.Option

	choices, err := makeAugmentOptions(options.N, options.M)
	if err != nil{
		return nil, err
	}

	augOpts = append(augOpts, choices...)

	if options.Normalize{
		normOpt := aug.WithNormalize(aug.WithNormalizeMean([]float64{0.485, 0.456, 0.406}), aug.WithNormalizeStd([]float64{0.229, 0.224, 0.225}))
		augOpts = append(augOpts, normOpt)
	}


	t, err := aug.Compose(augOpts...)
	return t, nil
}

func makeAugmentOptions(n, m int) ([]aug.Option, error){
	var options []aug.Option
	min := 0
	var opIndexes []int
	if len(optConfigs) > 0{
		// max := len(optConfigs) - 1
		max := len(optConfigs)
		opIndexes = randomInts(n, min, max)	
	}

	for _, idx := range opIndexes{
		cfg := optConfigs[idx]
		m := randomPoisson(float64(m))
		if m > 30{
			m = 30
		}
		val := float64(m)/float64(30) * (cfg.MaxVal - cfg.MinVal) + cfg.MinVal
		// fmt.Printf("aug: %s - v value: %v\n", cfg.OpName,val)
		
		opt, err := cfg.Make(val)
		if err != nil{
			return nil, err
		}
		options = append(options, opt)
	}

	return options, nil
}

// makes random n integers in range [min, max]
func randomInts(n, min, max int) []int{
	rand.Seed(time.Now().UnixNano())
	var choices []int
	for i := 0; i < n; i++{
		c := rand.Intn(max - min) + min
		if !contains(c, choices){
			choices = append(choices, c)
		}
	}

	// sort
	sort.Ints(choices)
	
	return choices
}

func contains(item int, items []int) bool{
	for _, i := range items{
		if item == i{
			return true
		}
	}
	return false
}

func randomPoisson(lambda float64) float64{
	p := distuv.Poisson{Lambda: lambda,}
	return p.Rand()
}

/*
 * class RandAugment:
 *     def __init__(self, n=3, m=12):
 *         self.n = n
 *         self.m = m      # [0, 30]
 *         self.augment_list = augment_list()
 *
 *     def __call__(self, image):
 *         img = Image.fromarray(image)
 *         ops = random.choices(self.augment_list, k=self.n)
 *         for op, minval, maxval in ops:
 *             m = np.random.poisson(self.m)
 *             m = 30 if m > 30 else m
 *             val = (float(m) / 30) * float(maxval - minval) + minval
 *             img = op(img, val)
 *         return {'image': np.asarray(img)}
 *  */
