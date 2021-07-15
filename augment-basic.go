package lab

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch/vision/aug"
)


func MakeTransformer(cfg TransformConfig) (aug.Transformer, error){
	var augments []aug.Option 
	if cfg.IsTransformer{
		switch cfg.TransformerName{
		case "RandomAugment":
			return NewRandomAugment()

		default:
			err := fmt.Errorf("makeTransformer failed: unsupported TransformerName %q\n", cfg.TransformerName)
			return nil, err
		}
	}

	// Compose a transformer from augment options
	for _, augOpt := range cfg.AugmentOpts{
		switch augOpt.Name{
		case "RandomAutocontrast":
			var pvalue float64 = 0.5
			for k, v := range augOpt.Params{
				if k == "pvalue"{
					pvalue = v.(float64)
					break
				}
			}
			a := aug.WithRandomAutocontrast(pvalue)
			augments = append(augments, a)

		case "RandomSolarize":
			var opts []aug.SolarizeOption
			for k, v := range augOpt.Params{
				switch k{
				case "threshold":
					threshold := v.(float64)
					o := aug.WithSolarizeThreshold(threshold)
					opts = append(opts, o)

				case "pvalue":
					pvalue := v.(float64)
					o := aug.WithSolarizePvalue(pvalue)
					opts = append(opts, o)
				}
			}
			a := aug.WithRandomSolarize(opts...)
			augments = append(augments, a)

		case "RandomAdjustSharpness":
			var opts []aug.SharpnessOption
			for k, v := range augOpt.Params{
				switch k{
				case "factor":
					factor := v.(float64)
					o := aug.WithSharpnessFactor(factor)
					opts = append(opts, o)

				case "pvalue":
					pvalue := v.(float64)
					o := aug.WithSharpnessPvalue(pvalue)
					opts = append(opts, o)
				}
			}
			a := aug.WithRandomAdjustSharpness(opts...)
			augments = append(augments, a)

		case "RandomRotate":
			var min, max float64
			for k, v := range augOpt.Params{
				switch k{
				case "min":
					min = v.(float64)
				case "max":
					max = v.(float64)
				}
			}
			a := aug.WithRandRotate(min, max)
			augments = append(augments, a)

		case "Rotate":
			var angle float64
			for k, v := range augOpt.Params{
				if k == "angle"{
					angle = v.(float64)
					break
				}
			}
			a := aug.WithRotate(angle)
			augments = append(augments, a)

		case "RandomAffine":
			var opts []aug.AffineOption
			for n, v := range augOpt.Params{
				switch n{
				case "fill_value":
					o := aug.WithAffineFillValue(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "mode":
					o := aug.WithAffineMode(v.(string))
					opts = append(opts, o)
				case "scale":
					o := aug.WithAffineScale(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "shear":
					o := aug.WithAffineShear(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "degree":
					o := aug.WithAffineDegree(sliceInterface2Int64(v.([]interface{})))
					opts = append(opts, o)
				case "translate":
					o := aug.WithAffineTranslate(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				}
			}

			a := aug.WithRandomAffine(opts...)
			augments = append(augments, a)

		// NOTE. skip this because resize is handled at DataLoader
		case "Resize":
			/*
			var h, w int64
			for k, v := range augOpt.Params{
				switch k{
				case "height":
					h = v.(int64)
				case "width":
					w = v.(int64)
				}
			}

			a := aug.WithResize(h, w)
			augments = append(augments, a)
			*/

		case "ZoomOut":
			var val float64 = 0.1 // default value
			for k, v := range augOpt.Params{
				if k == "value"{ // range [0, 0.5]
					val = v.(float64)
					break
				}
			}

			a := aug.WithZoomOut(val)
			augments = append(augments, a)

		case "RandomPosterize":
			var opts []aug.PosterizeOption
			for k, v := range augOpt.Params{
				switch k{
				case "bits":
					bits := v.(int)
					o := aug.WithPosterizeBits(uint8(bits))
					opts = append(opts, o)
				case "pvalue":
					o := aug.WithPosterizePvalue(v.(float64))
					opts = append(opts, o)
				}
			}
			a := aug.WithRandomPosterize(opts...)
			augments = append(augments, a)

		case "RandomPerspective":
			var opts []aug.PerspectiveOption
			for k, v := range augOpt.Params{
				switch k{
				case "mode":
					o := aug.WithPerspectiveMode(v.(string))
					opts = append(opts, o)
				case "pvalue":
					o := aug.WithPerspectivePvalue(v.(float64))
					opts = append(opts, o)
				case "value":
					o := aug.WithPerspectiveValue(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "scale":
					o := aug.WithPerspectiveScale(v.(float64))
					opts = append(opts, o)
				}
			}
			a := aug.WithRandomPerspective(opts...)
			augments = append(augments, a)

		case "Normalize":
			var opts []aug.NormalizeOption
			for k, v := range augOpt.Params{
				switch k{
				case "mean":
					o := aug.WithNormalizeMean(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "stdev":
					o := aug.WithNormalizeStd(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				}
			}
			a := aug.WithNormalize(opts...)
			augments = append(augments, a)

		case "RandomInvert":
			var pvalue float64 = 0.5
			for k, v := range augOpt.Params{
				if k == "pvalue"{
					pvalue = v.(float64)
					break
				}
			}
			a := aug.WithRandomInvert(pvalue)
			augments = append(augments, a)

		case "RandomGrayscale":
			var pvalue float64 = 0.5
			for k, v := range augOpt.Params{
				if k == "pvalue"{
					pvalue = v.(float64)
					break
				}
			}
			a := aug.WithRandomGrayscale(pvalue)
			augments = append(augments, a)

		case "RandomVFlip":
			var pvalue float64 = 0.5
			for k, v := range augOpt.Params{
				if k == "pvalue"{
					pvalue = v.(float64)
					break
				}
			}
			a := aug.WithRandomVFlip(pvalue)
			augments = append(augments, a)

		case "RandomHFlip":
			var pvalue float64 = 0.5
			for k, v := range augOpt.Params{
				if k == "pvalue"{
					pvalue = v.(float64)
					break
				}
			}
			a := aug.WithRandomHFlip(pvalue)
			augments = append(augments, a)

		case "RandomEqualize":
			var pvalue float64 = 0.5
			for k, v := range augOpt.Params{
				if k == "pvalue"{
					pvalue = v.(float64)
					break
				}
			}
			a := aug.WithRandomEqualize(pvalue)
			augments = append(augments, a)

		case "RandomCutout":
			var opts []aug.CutoutOption
			for k, v := range augOpt.Params{
				switch k{
				case "ratio":
					o := aug.WithCutoutRatio(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "scale":
					o := aug.WithCutoutScale(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "value":
					o := aug.WithCutoutValue(sliceInterface2Int64(v.([]interface{})))
					opts = append(opts, o)
				case "pvalue":
					o := aug.WithCutoutPvalue(v.(float64))
					opts = append(opts, o)
				}
			}
			
			a := aug.WithRandomCutout(opts...)
			augments = append(augments, a)

		case "CenterCrop":

			var size []int64
			for k, v := range augOpt.Params{
				if k == "size"{
					size = sliceInterface2Int64(v.([]interface{}))
					break
				}
			}
			a := aug.WithCenterCrop(size)
			augments = append(augments, a)

		case "ColorJitter":
			var opts []aug.ColorOption
			for n, v := range augOpt.Params{
				switch n{
				case "brightness":
					o := aug.WithColorBrightness(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "saturation":
					o := aug.WithColorSaturation(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "contrast":
					o := aug.WithColorContrast(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				case "hue":
					o := aug.WithColorHue(sliceInterface2Float64(v.([]interface{})))
					opts = append(opts, o)
				}
			}
			
			a := aug.WithColorJitter(opts...)
			augments = append(augments, a)

		default:
			err := fmt.Errorf("MakeTransformer failed: Unsupport augment option: %q\n", augOpt.Name)
			log.Fatal(err)
		}
	}

	return aug.Compose(augments...)
}

func sliceInterface2Float64(vals []interface{}) []float64{
	var retVal []float64
	for _, v := range vals{
		retVal = append(retVal, v.(float64))
	}

	return retVal
}

func sliceInterface2Int64(vals []interface{}) []int64{
	var retVal []int64
	for _, v := range vals{
		val := v.(int)
		retVal = append(retVal, int64(val))
	}

	return retVal
}
