package lab

import (
	"fmt"
	"io/ioutil"

	"gopkg.in/yaml.v3"
)

type ModelConfig struct {
	Name   string `yaml:"name"`
	Params struct {
		Backbone           string  `yaml:"backbone"`
		Pretrained         bool    `yaml:"pretrained"`
		PretrainedPath     string  `yaml:"pretrained_path"`
		NumClasses         int64   `yaml:"num_classes"`
		Dropout            float64 `yaml:"dropout"`
		MultisampleDropout bool    `yaml:"multisample_dropout"`
	} `yaml:"params"`
}

type TrainConfig struct {
	BatchSize    int64  `yaml:"batch_size"`
	Trainer      string `yaml:"trainer"`       // Trainer name
	LoadPrevious string `yaml:"load_previous"` // filepath to load pretrained weights
	Params       struct {
		GradientAcc      float64 `yaml:"gradient_accumulation"`
		Epochs           int     `yaml:"num_epochs"`
		StepsPerEpoch    int     `yaml:"steps_per_epoch"`
		ValidateInterval int     `yaml:"validate_interval"`
		Verbosity        int     `yaml:"verbosity"`
		Amp              bool    `yaml:"amp"`
	} `yaml:"params"`
}

type Transform struct{
	Augment string `yaml:"augment"`
	Params struct{
		N int `yaml:"n"`
		M int `yaml:"m"`
	} `yaml:"params"`
	PadRatio float64 `yaml:"pad_ratio"`
	ResizeTo []int64 `yaml:"resize_to"`
	Preprocess struct{
		ImageRange []int `yaml:"image_range"`
		InputRange []int `yaml:"input_range"`
		Mean []float64 `yaml:"mean"`
		Sdev []float64 `yaml:"sdev"`
	} `yaml:"preprocess"`
}

type Config struct {
	Seed int64 `yaml:"seed"`
	SlackURL string `yaml:"slack_url"`

	Dataset struct {
		Name   string `yaml:"name"`
		Params struct {
			Flip    bool `yaml:"flip"`
			Verbose bool `yaml:"verbose"`
		} `yaml:"params"`
		InnerFold   int      `yaml:"inner_fold"`
		OuterFold   int      `yaml:"outer_fold"`
		OuterOnly   bool     `yaml:"outer_only"`
		DataDir     []string `yaml:"data_dir"`
		CSVFilename string   `yaml:"csv_filename"`
	} `yaml:"dataset"`

	Transform Transform `yaml:"transform"`

	Model ModelConfig `yaml:"model"`

	FindLR struct {
		StartLR float64 `yaml:"start_lr"`
		EndLR   float64 `yaml:"end_lr"`
		NumIter int     `yaml:"num_inter"`
		SaveFig bool    `yaml:"save_fig"`
	} `yaml:"find_lr"`

	Train TrainConfig `yaml:"train"`

	Evaluation struct {
		BatchSize int64  `yaml:"batch_size"`
		Evaluator string `yaml:"evaluator"`
		Params    struct {
			SaveCheckpointDir string   `yaml:"save_checkpoint_dir"`
			SaveBest          bool     `yaml:"save_best"`
			Prefix            string   `yaml:"prefix"`
			Metrics           []string `yaml:"metrics"`
			ValidMetric       string   `yaml:"valid_metric"`
			Mode              string   `yaml:"mode"`
			ImproveThresh     float64  `yaml:"improve_thresh"`
		} `yaml:"params"`
	} `yaml:"evaluation"`

	Loss struct {
		Name   string `yaml:"name"`
		Params struct {
		} `yaml:"params"`
	} `yaml:"loss"`

	Optimizer struct {
		Name   string `yaml:"name"`
		Params struct {
			Eps         float64 `yaml:"eps"`
			LR          float64 `yaml:"lr"`
			WeightDecay float64 `yaml:"weight_decay"`
		} `yaml:"params"`
	} `yaml:"optimizer"`

	Scheduler struct {
		Name   string `yaml:"name"`
		Params struct {
			MaxLR    float64 `yaml:"max_lr"`
			FinalLR  float64 `yaml:"final_lr"`
			PctStart float64 `yaml:"pct_start"`
		} `yaml:"params"`
	} `yaml:"scheduler"`

	Test struct {
		Checkpoint      string `yaml:"checkpoint"`
		BatchSize       int64  `yaml:"batch_size"`
		DataDir         string `yaml:"data_dir"`
		SavePredsDir    string `yaml:"save_preds_dir"`
		LabelsAvailable string `yaml:"labels_available"`
		OuterOnly       bool   `yaml:"outer_only"`
		SaveFile        string `yaml:"save_file"`
	} `yaml:"test"`
}

// NewConfig returns a new Config struct
func NewConfig(filename string) (*Config, error) {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	c := &Config{}
	err = yaml.Unmarshal(buf, c)
	if err != nil {
		return nil, fmt.Errorf("in file %q: %v", filename, err)
	}

	return c, nil
}

func (cfg *Config) SetInferenceBatchSize() {
	// TODO.
}

func (cfg *Config) SetReproducibility() {
	// TODO. set config.Seed here
}
