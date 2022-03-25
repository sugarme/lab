package lab

import (
	"fmt"
	"log"
	"sort"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"

	lib "github.com/sugarme/lab/model"
)

// CheckBuildModel tries to build model from input configuration.
// It returns error if build failed.
func CheckBuildModel(cfg *Config) error {
	b := NewBuilder(cfg)

	_, err := b.BuildModel()
	if err != nil {
		err = fmt.Errorf("CheckModel - Building model failed: %w", err)
		return err
	}

	return nil
}

// ModelSummary prints out model architecture to stdout.
func ModelSummary(modelName string) {
	vs := nn.NewVarStore(gotch.CPU)
	switch modelName {
	case "densenet121":
		_ = lib.DenseNet121(vs.Root(), 1000)
	default:
		log.Fatalf("Unsupported model %q\n", modelName)
	}

	var namedTensors []ts.NamedTensor
	for name, x := range vs.Variables() {
		namedTensors = append(namedTensors, ts.NamedTensor{
			Name:   name,
			Tensor: &x,
		})
	}

	printNamedTensors(namedTensors)
}

// PretrainedSummary loads and prints out layers of pretrained model from input file.
func PretrainedSummary(file string) {
	vs := nn.NewVarStore(gotch.CPU)
	err := vs.Load(file)

	namedTensors, err := ts.LoadMultiWithDevice(file, vs.Device())
	if err != nil {
		log.Fatal(err)
	}

	vs = nil

	printNamedTensors(namedTensors)
}

// Print named tensors
func printNamedTensors(namedTensors []ts.NamedTensor) {
	layers := make([]string, 0, len(namedTensors))
	for _, namedTensor := range namedTensors {
		layers = append(layers, namedTensor.Name)
	}
	sort.Strings(layers)
	for _, l := range layers {
		var x *ts.Tensor
		for _, nts := range namedTensors {
			if nts.Name == l {
				x = nts.Tensor
				break
			}
		}
		fmt.Printf("%s - %+v\n", l, x.MustSize())
	}

	fmt.Printf("Num of layers: %v\n", len(namedTensors))
}
