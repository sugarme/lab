package lab

import (
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

var ModelZoo map[string]string = map[string]string{
	"efficientnet_b0": "EffNet",
	"efficientnet_b1": "EffNet",
	"efficientnet_b2": "EffNet",
	"efficientnet_b3": "EffNet",
	"efficientnet_b4": "EffNet",
	"efficientnet_b5": "EffNet",
	"efficientnet_b6": "EffNet",
	"efficientnet_b7": "EffNet",

	"tf_efficientnet_b0_ns": "EffNet",
	"tf_efficientnet_b1_ns": "EffNet",
	"tf_efficientnet_b2_ns": "EffNet",
	"tf_efficientnet_b3_ns": "EffNet",
	"tf_efficientnet_b4_ns": "EffNet",
	"tf_efficientnet_b5_ns": "EffNet",
	"tf_efficientnet_b6_ns": "EffNet",
	"tf_efficientnet_b7_ns": "EffNet",

	"resnet34_unet": "UNet",
}

// Model represents a deep learning model.
type Model struct {
	Name    string
	Module  ts.ModuleT
	Weights *nn.VarStore
}

// Eval set model to evaluation mode
func (m *Model) Eval() {
	ts.MustGradSetEnabled(false)
}

// Train set model to training mode
func (m *Model) Train() {
	ts.MustGradSetEnabled(true)
}
