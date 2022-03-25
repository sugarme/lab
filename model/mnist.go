package model

import (
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type Net struct {
	conv1 *nn.Conv2D
	conv2 *nn.Conv2D
	fc1   *nn.Linear
	fc2   *nn.Linear
}

func MNIST(vs *nn.Path) *Net {
	conv1 := nn.NewConv2D(vs, 1, 32, 5, nn.DefaultConv2DConfig())
	conv2 := nn.NewConv2D(vs, 32, 64, 5, nn.DefaultConv2DConfig())
	fc1 := nn.NewLinear(vs, 1024, 1024, nn.DefaultLinearConfig())
	fc2 := nn.NewLinear(vs, 1024, 10, nn.DefaultLinearConfig())

	return &Net{
		conv1,
		conv2,
		fc1,
		fc2}
}

func (n *Net) ForwardT(xs *ts.Tensor, train bool) *ts.Tensor {
	outView1 := xs.MustView([]int64{-1, 1, 28, 28}, false)
	defer outView1.MustDrop()

	outC1 := outView1.Apply(n.conv1)

	outMP1 := outC1.MaxPool2DDefault(2, true)
	defer outMP1.MustDrop()

	outC2 := outMP1.Apply(n.conv2)

	outMP2 := outC2.MaxPool2DDefault(2, true)

	outView2 := outMP2.MustView([]int64{-1, 1024}, true)
	defer outView2.MustDrop()

	outFC1 := outView2.Apply(n.fc1)

	outRelu := outFC1.MustRelu(true)
	defer outRelu.MustDrop()
	outDropout := ts.MustDropout(outRelu, 0.5, train)
	defer outDropout.MustDrop()

	return outDropout.Apply(n.fc2)
}
