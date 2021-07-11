package lab

import (
	// "log"
	"fmt"
	"math"

	"github.com/sugarme/gotch/dutil"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

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
	case "valid":
		shuffle = false
		batchSize = b.Config.Evaluation.BatchSize
	default:
		err := fmt.Errorf("Unsuported mode: %q\n", mode)
		return nil, err
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

type LossFunc func(logits, target *ts.Tensor) *ts.Tensor

// BuildLoss builds loss function.
func (b *Builder) BuildLoss() (LossFunc, error) {
	name := b.Config.Loss.Name
	var lossFunc LossFunc
	switch name {
	case "CrossEntropyLoss":
		lossFunc = func(logits, target *ts.Tensor) *ts.Tensor {
			return CrossEntropyLoss(logits, target)
		}

	case "BCELoss":
		lossFunc = func(logits, target *ts.Tensor) *ts.Tensor{
			return BCELoss(logits, target)				
		}

	case "DiceLoss":
		lossFunc = func(logits, target *ts.Tensor) *ts.Tensor{
			return DiceLoss(logits, target)				
		}

	case "JaccardLoss":
		lossFunc = func(logits, target *ts.Tensor) *ts.Tensor{
			return JaccardLoss(logits, target)				
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
	var update string
	switch name {
	case "OneCycleLR":
		maxLR := b.Config.Scheduler.Params.MaxLR
		finalLR := b.Config.Scheduler.Params.FinalLR
		pctStart := b.Config.Scheduler.Params.PctStart
		epochs := b.Config.Train.Params.Epochs
		stepPerEpoch := int(b.Config.Train.BatchSize)
		s = nn.NewOneCycleLR(opt, maxLR, nn.WithOneCycleFinalDivFactor(finalLR), nn.WithOneCyclePctStart(pctStart), nn.WithOneCycleEpochs(epochs), nn.WithOneCycleStepsPerEpoch(stepPerEpoch)).Build()
		update = "on_batch"
	case "CosineAnnealingWarmRestarts":
		t0 := 10
		tMult := 1
		etaMin := 0.001
		s = nn.NewCosineAnnealingWarmRestarts(opt, t0, nn.WithTMult(tMult), nn.WithEtaMin(etaMin)).Build()
		update = "on_batch"
	case "StepLR": // reduce LR by 0.1 every 10 epochs
		stepSize := 10
		gamma := 0.1
		s = nn.NewStepLR(opt, stepSize, gamma).Build()
		update = "on_epoch"
	case "LambdaLR":
		ld1 := func(epoch interface{}) float64 {
			return float64(epoch.(int) / 30)
		}
		s = nn.NewLambdaLR(opt, []nn.LambdaFn{ld1}).Build()
		update = "on_epoch"
	case "MultiplicativeLR":
		ld1 := func(epoch interface{}) float64 {
			e := float64(epoch.(int))
			return math.Pow(2, e) // 2 ** epoch
		}
		s = nn.NewMultiplicativeLR(opt, []nn.LambdaFn{ld1}).Build()
		update = "on_epoch"
	case "ExponentialLR":
		gamma := 0.1
		s = nn.NewExponentialLR(opt, gamma).Build()
		update = "on_epoch"
	case "CosineAnnealingLR":
		steps := 10
		s = nn.NewCosineAnnealingLR(opt, steps, 0.0).Build()
		update = "on_batch"
	case "CyclicLR":
		baseLRs := []float64{0.001}
		maxLRs := []float64{0.1}
		s = nn.NewCyclicLR(opt, baseLRs, maxLRs, nn.WithCyclicStepSizeUp(5), nn.WithCyclicMode("triangular")).Build()
		update = "on_epoch"
	case "ReduceLROnPlateau":
		s = nn.NewReduceLROnPlateau(opt).Build()
		update = "on_valid"
	default:
		err := fmt.Errorf("BuildScheduler failed: Unsupported LR scheduler: %q\n", name)
		return nil, err
	}

	scheduler := NewScheduler(s, name, update)

	return scheduler, nil
}
