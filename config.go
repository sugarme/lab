package lab

import (
	"fmt"
	"io/ioutil"

	"github.com/sugarme/gotch/vision/aug"
	"gopkg.in/yaml.v3"
)

// Dataset Config:
// ===============
type DatasetConfig struct {
		Name   string `yaml:"name"`
		Params map[string]interface{} `yaml:"params"`
		DataDir     []string `yaml:"data_dir"`
		CSVFilename string   `yaml:"csv_filename"`
}

// Transform Config:
// =================
type AugmentOpt struct{
	Name string `yaml:"name"`
	Params map[string]interface{} `yaml:"params"`
}

type TransformConfig struct{
	IsTransformer bool `yaml:"is_transformer"`
	TransformerName string `yaml:"transformer_name"`
	Transformer aug.Transformer `yaml:"transformer"`
	AugmentOpts []AugmentOpt `yaml:"augment_opts"` // Augment options to compose a transformer
}


// Model Config:
// =============
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

// Train Config:
// ============
type TrainConfig struct {
	BatchSize    int64  `yaml:"batch_size"`
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

// FindLR Config:
// ==============
type FindLRConfig struct{
		StartLR float64 `yaml:"start_lr"`
		EndLR   float64 `yaml:"end_lr"`
		NumIter int     `yaml:"num_iter"`
		SaveFig bool    `yaml:"save_fig"`
}

// Evaluation Config:
// ==================
type EvaluationConfig struct{
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
}

// Loss Config:
// ============
type LossConfig struct{
	Name string `yaml:"name"`
	Params map[string]interface{} `yaml:"params"`
}

// Optimizer Config:
// =================
type OptimizerConfig struct{
	Name string `yaml:"name"`
	Params map[string]interface{} `yaml:"params"`
}

// LRScheduler Config:
// ===================
type LRSchedulerConfig struct{
	Name string `yaml:"name"`
	Params map[string]interface{} `yaml:"params"`
}

// Test Config:
// ============
type TestConfig struct{
		Checkpoint      string `yaml:"checkpoint"`
		BatchSize       int64  `yaml:"batch_size"`
		DataDir         string `yaml:"data_dir"`
		SavePredsDir    string `yaml:"save_preds_dir"`
		LabelsAvailable string `yaml:"labels_available"`
		OuterOnly       bool   `yaml:"outer_only"`
		SaveFile        string `yaml:"save_file"`
}

type Config struct {
	Seed int64 `yaml:"seed"`
	SlackURL string `yaml:"slack_url"`
	Dataset DatasetConfig `yaml:"dataset"`
	Transform struct{
		Train TransformConfig `yaml:"train"`
		Valid TransformConfig `yaml:"valid"`
	} `yaml:"transform"`
	Model ModelConfig `yaml:"model"`
	FindLR FindLRConfig `yaml:"find_lr"`
	Train TrainConfig `yaml:"train"`
	Evaluation EvaluationConfig `yaml:"evaluation"`
	Loss LossConfig `yaml:"loss"`
	Optimizer OptimizerConfig `yaml:"optimizer"`
	Scheduler LRSchedulerConfig `yaml:"scheduler"`
	Test TestConfig `yaml:"test"`
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
