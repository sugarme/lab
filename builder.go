package lab

import (
	"fmt"
	"math"

	"github.com/sugarme/gotch/dutil"
	"github.com/sugarme/gotch/vision/aug"

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

	case "ResNet":
		module = lib.ResNet(vs.Root(), cfg.Params.NumClasses, cfg.Params.Backbone)

	case "DenseNet":
		module = lib.DenseNet(vs.Root(), cfg.Params.NumClasses, cfg.Params.Backbone)

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

	var lr float64
	for k, v := range params{
		switch k{
		case "lr":
			lr = v.(float64)
		}
	}

	switch name {
	case "AdamW":
		cfg := nn.DefaultAdamWConfig()
		for k, v := range params{
			switch k{
			case "beta1":
				cfg.Beta1 = v.(float64)
			case "beta2":
				cfg.Beta2 = v.(float64)
			case "wd":
				cfg.Wd = v.(float64)
			}
		}
		opt, err := cfg.Build(vs, lr)
		if err != nil {
			err = fmt.Errorf("Build AdamW optimizer failed: %w", err)
			return nil, err
		}
		return opt, nil

	case "Adam":
		cfg := nn.DefaultAdamConfig()
		for k, v := range params{
			switch k{
			case "beta1":
				cfg.Beta1 = v.(float64)
			case "beta2":
				cfg.Beta2 = v.(float64)
			case "wd":
				cfg.Wd = v.(float64)
			}
		}
		opt, err := cfg.Build(vs, lr)
		if err != nil {
			err = fmt.Errorf("Build Adam optimizer failed: %w", err)
			return nil, err
		}
		return opt, nil

	case "SGD":
		cfg := nn.DefaultSGDConfig()
		for k, v := range params{
			switch k{
			case "dampening":
				cfg.Dampening = v.(float64)
			case "momentum":
				cfg.Momentum = v.(float64)
			case "wd":
				cfg.Wd = v.(float64)
			case "nesterov":
				cfg.Nesterov = v.(bool)
			}
		}
		opt, err := cfg.Build(vs, lr)
		if err != nil {
			err = fmt.Errorf("Build SGD optimizer failed: %w", err)
			return nil, err
		}
		return opt, nil

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
	params := b.Config.Scheduler.Params
	var s *nn.LRScheduler
	var update string
	switch name {
	case "OneCycleLR":
		var opts []nn.OneCycleOption
		var maxLR float64
		for k, v := range params{
			switch k{
			case "max_lr":
				maxLR = v.(float64)
			case "final_lr":
				finalLR := v.(float64)
				o := nn.WithOneCycleFinalDivFactor(finalLR)
				opts = append(opts, o)

			case "pct_start":
				pctStart := v.(float64)
				o := nn.WithOneCyclePctStart(pctStart)
				opts = append(opts, o)
			case "epochs":
				epochs := v.(int)
				o := nn.WithOneCycleLastEpoch(epochs)
				opts = append(opts, o)
			}
		}
		stepPerEpoch := int(b.Config.Train.BatchSize)
		o := nn.WithOneCycleStepsPerEpoch(stepPerEpoch)
		opts = append(opts, o)
		s = nn.NewOneCycleLR(opt, maxLR, opts...).Build()
		update = "on_batch"
	case "CosineAnnealingWarmRestarts":
		t0 := 10
		tMult := 1
		etaMin := 0.001
		for k, v := range params{
			switch k{
			case "t0":
				t0 = v.(int)
			case "t_mult":
				tMult = v.(int)
			case "eta_min":
				etaMin = v.(float64)
			}
		}
		s = nn.NewCosineAnnealingWarmRestarts(opt, t0, nn.WithTMult(tMult), nn.WithEtaMin(etaMin)).Build()
		update = "on_batch"
	case "StepLR": // reduce LR by 0.1 every 10 epochs
		stepSize := 10
		gamma := 0.1
		for k, v := range params{
			switch k{
			case "step_size":
				stepSize = v.(int)
			}
		}
		s = nn.NewStepLR(opt, stepSize, gamma).Build()
		update = "on_epoch"
	case "LambdaLR":
		denominator := 30
		for k, v := range params{
			switch k{
			case "denominator":
				denominator = v.(int)
			}
		}
		ld1 := func(epoch interface{}) float64 {
			return float64(epoch.(int) / denominator)
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
		for k, v := range params{
			switch k{
			case "gamma":
				gamma = v.(float64)
			}
		}
		s = nn.NewExponentialLR(opt, gamma).Build()
		update = "on_epoch"
	case "CosineAnnealingLR":
		tmax := 10
		etaMin := 0.0
		for k, v := range params{
			switch k{
			case "tmax":
				tmax = v.(int)
			case "eta_min":
				etaMin = v.(float64)
			}
		}
		s = nn.NewCosineAnnealingLR(opt, tmax, etaMin).Build()
		update = "on_batch"
	case "CyclicLR":
		baseLRs := []float64{0.001}
		maxLRs := []float64{0.1}

		for k, v := range params{
			switch k{
			case "base_lr":
				baseLRs = sliceInterface2Float64(v.([]interface{}))
			case "max_lr":
				maxLRs = sliceInterface2Float64(v.([]interface{}))
			}
		}
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


func (b *Builder) BuildTransformer(mode string) (aug.Transformer, error) {
	switch mode{
	case "train":
		config := b.Config.Transform.Train
		return makeTransformer(config)

	case "valid":
		config := b.Config.Transform.Train
		return makeTransformer(config)
	default:
		err := fmt.Errorf("BuildTrainformer failed. Invalid mode. Mode should be either 'train' or 'valid'. Got %q\n", mode)
		return nil, err
	}
}

