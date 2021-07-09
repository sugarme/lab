package loss

import (
	"fmt"
	"log"
)

type options struct{
	Smooth float64
	Eps float64
	Dims []int64
	Mode string // one of ["BinaryMode", "MultiClassMode", "MultiLabelMode"]
	FromLogits bool
	Classes []int // List of classes that contribute in loss computation. By default, all channels are included.
	LogLoss bool // default = false
}

type Option func(*options)

func defaultOptions() *options{
	return &options{
		Smooth: 0.0,
		Eps: 1.0e-7,
		Dims: nil,
		Mode: "BinaryMode",
		FromLogits: true,
		Classes: nil,
		LogLoss: false,
	}
}

func WithSmooth(val float64) Option{
	return func(o *options){
		o.Smooth = val
	}
}

func WithEpsilon(val float64) Option{
	return func(o *options){
		o.Eps = val
	}
}

func WithDims(val []int64) Option{
	return func(o *options){
		o.Dims = val
	}
}

func WithFromLogits(val bool) Option{
	return func(o *options){
		o.FromLogits = val
	}
}

func WithLogLoss(val bool) Option{
	return func(o *options){
		o.LogLoss = val
	}
}

func WithClasses(val []int) Option{
	return func(o *options){
		o.Classes = val
	}
}

func WithMode(val string) Option{
	switch validMode(val){
	case true:
		return func(o *options){
			o.Mode = val
		}

	default:
		err := fmt.Errorf("Invalid mode option: %q\n", val)
		log.Fatal(err)
		return nil
	}
}

func validMode(mode string) bool{
	modes := []string{"BinaryMode", "MultiClassMode", "MultiLabelMode"}
	for _, m := range modes{
		if mode == m{
			return true
		}
	}
	return false
}
