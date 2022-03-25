package model

import (
	"log"

	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/iseg/unet"
)

func resnetUnet(vs *nn.VarStore) ts.ModuleT {
	net := unet.DefaultUNet(vs.Root())

	return net
}

func UNet(p *nn.Path, backbone string) ts.ModuleT {
	var m ts.ModuleT
	switch backbone {
	case "resnet34_unet":
		m = unet.DefaultUNet(p)
	default:
		log.Fatalf("Invalid backbone type: %s\n", backbone)
	}

	return m
}
