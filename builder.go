package lab

import (
	// "log"
	"fmt"

	"github.com/sugarme/gotch/dutil"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"github.com/sugarme/lab/loss"
	lib "github.com/sugarme/lab/model"
)

type Builder struct {
	Config *Config
}

// func NewBuilder(cfg *Config, trainData, validData dutil.Dataset) *Builder {
func NewBuilder(cfg *Config) *Builder {
	return &Builder{
		Config: cfg,
	}
}

func (b *Builder) BuildDataLoader(data dutil.Dataset, mode string) (*dutil.DataLoader, error) {
	var shuffle bool
	var batchSize int64
	switch mode {
	case "train":
		shuffle = true
		batchSize = b.Config.Train.BatchSize
	default:
		shuffle = false
		batchSize = b.Config.Evaluation.BatchSize
	}

	n := data.Len()
	sampler, err := dutil.NewBatchSampler(n, int(batchSize), true, shuffle)
	if err != nil {
		err := fmt.Errorf("BuildTrainLoader failed: %w\n", err)
		return nil, err
	}

	return dutil.NewDataLoader(data, sampler)
}

func (b *Builder) BuildModel(configOpt ...ModelConfig) (*Model, error) {
	// device := gotch.CPU
	device := gotch.CudaIfAvailable()
	vs := nn.NewVarStore(device)
	cfg := b.Config.Model
	backbone := cfg.Params.Backbone
	mclass, ok := ModelZoo[backbone]
	if !ok {
		err := fmt.Errorf("Could not find model name %q in model zoo.", backbone)
		return nil, err
	}

	// Build model
	var module ts.ModuleT
	switch mclass {
	case "EffNet":
		module = lib.EffNet(vs.Root(), cfg.Params.NumClasses, cfg.Params.Backbone, cfg.Params.Dropout)

	case "UNet":
		module = lib.UNet(vs.Root(), cfg.Params.Backbone)

	// TODO: continue

	default:
		err := fmt.Errorf("Invalid Model Class %q", mclass)
		return nil, err
	}

	// Load pretrained if specified
	if cfg.Params.Pretrained {
		pretrainedFile, err := lib.PretrainedFile(cfg.Params.Backbone, lib.WithPath(cfg.Params.PretrainedPath))
		if err != nil {
			err := fmt.Errorf("Get pretrained file failed: %w\n", err)
			return nil, err
		}
		// Load partial because we modify number of classes in classify layer.
		_, err = vs.LoadPartial(pretrainedFile)
		if err != nil {
			err = fmt.Errorf("Load pretrained backbone weights failed: %w\n", err)
			return nil, err
		}
	}

	m := &Model{
		Name:    backbone,
		Weights: vs,
		Module:  module,
	}
	return m, nil
}

type LossFunc func(logits, labels *ts.Tensor) *ts.Tensor

// BuildLoss builds loss function.
func (b *Builder) BuildLoss() (LossFunc, error) {
	name := b.Config.Loss.Name
	var lossFunc LossFunc
	switch name {
	case "CrossEntropyLoss":
		lossFunc = func(logits, labels *ts.Tensor) *ts.Tensor {
			return logits.CrossEntropyForLogits(labels)
		}

	case "CriterionBCE":
		lossFunc = func(logits, labels *ts.Tensor) *ts.Tensor{
			return loss.CriterionBinaryCrossEntropy(logits, labels)				
		}

	case "BCELoss":
		lossFunc = func(logits, labels *ts.Tensor) *ts.Tensor{
			return loss.BCELoss(logits, labels)				
		}
	default:
		err := fmt.Errorf("Unsupported loss function: %s\n", name)
		return nil, err
	}

	return lossFunc, nil
}

// BuildOptimizer builds optimizer.
func (b *Builder) BuildOptimizer(vs *nn.VarStore) (*nn.Optimizer, error) {
	modelParams := b.Config.Model.Params
	params := b.Config.Optimizer.Params
	name := b.Config.Optimizer.Name

	var (
		opt *nn.Optimizer
		err error
	)

	fmt.Printf("modelParams: %+v\n", modelParams)
	fmt.Printf("optimizer params: %+v\n", params)
	fmt.Printf("optimizer name: %+v\n", name)

	lr := params.LR
	switch name {
	case "AdamW":
		opt, err = nn.DefaultAdamWConfig().Build(vs, lr)
		if err != nil {
			err = fmt.Errorf("Build AdamW optimizer failed: %w", err)
			return nil, err
		}
	case "Adam":
		opt, err = nn.DefaultAdamConfig().Build(vs, lr)
		if err != nil {
			err = fmt.Errorf("Build Adam optimizer failed: %w", err)
			return nil, err
		}
	case "SGD":
		opt, err = nn.DefaultSGDConfig().Build(vs, lr)
		if err != nil {
			err = fmt.Errorf("Build SGD optimizer failed: %w", err)
			return nil, err
		}
	default:
		err = fmt.Errorf("Unsupported optimizer config %q\n", name)
		return nil, err
	}

	return opt, nil
}

// BuildScheduler builds optimizer scheduler.
func (b *Builder) BuildScheduler(opt *nn.Optimizer) (*Scheduler, error) {
	// TODO.
	name := b.Config.Scheduler.Name
	var s *nn.LRScheduler
	switch name {
	case "OneCycleLR":
		maxLR := b.Config.Scheduler.Params.MaxLR
		finalLR := b.Config.Scheduler.Params.FinalLR
		pctStart := b.Config.Scheduler.Params.PctStart
		epochs := b.Config.Train.Params.Epochs
		stepPerEpoch := int(b.Config.Train.BatchSize)
		s = nn.NewOneCycleLR(opt, maxLR, nn.WithOneCycleFinalDivFactor(finalLR), nn.WithOneCyclePctStart(pctStart), nn.WithOneCycleEpochs(epochs), nn.WithOneCycleStepsPerEpoch(stepPerEpoch)).Build()
	case "CosineAnnealingWarmRestarts":
		t0 := 10
		tMult := 1
		etaMin := 0.001
		s = nn.NewCosineAnnealingWarmRestarts(opt, t0, nn.WithTMult(tMult), nn.WithEtaMin(etaMin)).Build()
	case "StepLR":
		stepSize := 1000
		gamma := 0.0001
		s = nn.NewStepLR(opt, stepSize, gamma).Build()
	}

	funcName := name
	update := "on_batch"

	return &Scheduler{s, update, funcName}, nil
}
