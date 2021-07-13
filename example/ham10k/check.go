package main

import (
	"fmt"
	"log"
	"sort"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/lab"

	lib "github.com/sugarme/lab/model"
)

func checkModel(cfg *lab.Config) error {
	b := lab.NewBuilder(cfg)

	model, err := b.BuildModel()
	if err != nil {
		err = fmt.Errorf("Building model failed: %w", err)
		return err
	}

	fmt.Println(model)
	return nil
}

func checkModelArchitecture(file string) {
	modelSummary("densenet121")

	pretrainedFile := "./pretrained/densenet121.bin"
	pretrainedSummary(pretrainedFile)
}

func modelSummary(modelName string){
	vs := nn.NewVarStore(gotch.CPU)
	var net ts.ModuleT
	switch modelName{
	case "densenet121":
		net = lib.DenseNet121(vs.Root(), 1000)
	default:
		log.Fatalf("Unsupported model %q\n", modelName)
	}

	var namedTensors []ts.NamedTensor
	for name, x := range vs.Variables(){
		namedTensors = append(namedTensors, ts.NamedTensor{
			Name: name,
			Tensor: x,
		})
	}

	printNamedTensors(namedTensors)

	fmt.Println(net)
}

func pretrainedSummary(file string){
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
func printNamedTensors(namedTensors []ts.NamedTensor){
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

