package lab

import (
	"fmt"
	"log"
)

type options struct {
	Smooth      float64
	Eps         float64
	Dims        []int64
	Mode        string // one of ["BinaryMode", "MultiClassMode", "MultiLabelMode"]
	FromLogits  bool
	Classes     []int // List of classes that contribute in loss computation. By default, all channels are included.
	LogLoss     bool  // default = false
	Threshold   float64
	AverageMode string // average mode for classification meter. One of "macro", "weighted", "micro". Default = "macro"
}

type MetricOption func(*options)

func defaultMetricOptions() *options {
	return &options{
		Smooth:      0.0,
		Eps:         1.0e-7,
		Dims:        nil,
		Mode:        "BinaryMode",
		FromLogits:  true,
		Classes:     nil,
		LogLoss:     false,
		Threshold:   0.5,
		AverageMode: "macro",
	}
}

func WithMetricSmooth(val float64) MetricOption {
	return func(o *options) {
		o.Smooth = val
	}
}

func WithMetricEpsilon(val float64) MetricOption {
	return func(o *options) {
		o.Eps = val
	}
}

func WithMetricDims(val []int64) MetricOption {
	return func(o *options) {
		o.Dims = val
	}
}

func WithMetricFromLogits(val bool) MetricOption {
	return func(o *options) {
		o.FromLogits = val
	}
}

func WithMetricLogLoss(val bool) MetricOption {
	return func(o *options) {
		o.LogLoss = val
	}
}

func WithMetricClasses(val []int) MetricOption {
	return func(o *options) {
		o.Classes = val
	}
}

func WithMetricMode(val string) MetricOption {
	switch validMode(val) {
	case true:
		return func(o *options) {
			o.Mode = val
		}

	default:
		err := fmt.Errorf("Invalid mode option: %q\n", val)
		log.Fatal(err)
		return nil
	}
}

func WithMetricThreshold(val float64) MetricOption {
	return func(o *options) {
		o.Threshold = val
	}
}

func validMode(mode string) bool {
	modes := []string{"BinaryMode", "MultiClassMode", "MultiLabelMode"}
	for _, m := range modes {
		if mode == m {
			return true
		}
	}
	return false
}

func WithAverageMode(val string) MetricOption {
	switch val {
	case "macro", "weighted", "micro":
		return func(o *options) {
			o.Mode = val
		}

	default:
		err := fmt.Errorf("Invalid average mode option: %q\n", val)
		log.Fatal(err)
		return nil
	}
}
