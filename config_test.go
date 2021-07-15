package lab

import (
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestTransformConfig(t *testing.T){
	file := "./config-sample.yaml"
	cfg, err := NewConfig(file)

	if err != nil{
		t.Errorf("Parse config file failed: %v\n", err)
	}

	t.Logf("%+v\n", cfg)

	tf, err := makeTransformer(cfg.Transform.Train)

	if err != nil{
		t.Errorf("Make tranformer failed: %v\n", err)
	}

	t.Log(tf)
}

func TestDatasetConfig(t *testing.T){
	yamlFile := []byte(`dataset:
  name: SampleDataset
  params:
    flip: true
    verbose: true
  data_dir: ["data/images"]
  csv_filename: data/GroundTruth.csv
`)

	var config Config
	err := yaml.Unmarshal(yamlFile, &config)
	if err != nil{
		t.Errorf("Unmarshal data failed: %v\n", err)
	}

	got := config.Dataset
	want := DatasetConfig{
		Name: "SampleDataset",
		Params: map[string]interface{}{
			"flip": true,
			"verbose": true,
		},
		DataDir: []string{"data/images"},
		CSVFilename: "data/GroundTruth.csv",
	}

	if !reflect.DeepEqual(want, got){
		t.Errorf("Want: %v\n", want)
		t.Errorf("Got: %v\n", got)
	}
}

func TestTrainConfig(t *testing.T){

}
