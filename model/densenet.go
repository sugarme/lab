package model

import (
	"log"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

// DenseNet creates DenseNet ModuleT.
func DenseNet(p *nn.Path, nclasses int64, backbone string) ts.ModuleT {
	var m ts.ModuleT
	switch backbone {
	case "densenet121":
		m = vision.DenseNet121(p, nclasses) 
	case "densenet161":
		m = vision.DenseNet161(p, nclasses) 
	case "densenet169":
		m = vision.DenseNet169(p, nclasses) 
	case "densenet201":
		m = vision.DenseNet201(p, nclasses) 
	default:
		log.Fatalf("Invalid backbone type: %s\n", backbone)
	}

	return m
}
