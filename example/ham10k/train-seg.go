package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/lab"
)

func trainSegmentation(cfg *lab.Config, isFull bool) {
	fmt.Printf("Start training...\n")
	dataDir := cfg.Dataset.DataDir[0]
	var(
		trainSet, validSet []Sample
		err error
	)
	if isFull{
		trainSet, validSet, err = makeFullDatasets(dataDir)
	} else {
		trainSet, validSet, err = makeDatasets(dataDir)
	}

	trainData, err := NewHamDataset(trainSet, cfg, true)
	if err != nil{
		log.Fatalf("Create train dataset failed: %v\n", err)
	}
	validData, err := NewHamDataset(validSet, cfg, false)
	if err != nil{
		log.Fatalf("Create valid dataset failed: %v\n", err)
	}

	b := lab.NewBuilder(cfg)
	trainLoader, err := b.BuildDataLoader(trainData, "train")
	if err != nil {
		log.Fatal(err)
	}

	validLoader, err := b.BuildDataLoader(validData, "valid")
	if err != nil {
		log.Fatal(err)
	}

	model, err := b.BuildModel()
	if err != nil {
		err = fmt.Errorf("Building model failed: %w", err)
		log.Fatal(err)
	}

	logFile := fmt.Sprintf("%s/log_%v.txt", cfg.Evaluation.Params.SaveCheckpointDir, time.Now())
	logger, err := lab.NewLogger(lab.WithLoggerFile(logFile), lab.WithLoggerSlackURL(cfg.SlackURL))
	if err != nil {
		err = fmt.Errorf("Creating logger failed: %w", err)
		log.Fatal(err)
	}


	// Load pretrained weights. NOTE. this is after loading backbone weights
	// that happens in side builder.BuildModel
	if cfg.Train.LoadPrevious != "" {
		err := model.Weights.Load(cfg.Train.LoadPrevious)
		if err != nil {
			err = fmt.Errorf("Load pretrained failed: %w", err)
			log.Fatal(err)
		}
		logger.Printf("Previously trained weights loaded from %q\n", cfg.Train.LoadPrevious)
	}

	// Adjust steps per epoch if necessary (i.e., equal to 0)
	// We assume if gradient accumulation is specified, then the user
	// has already adjusted the steps_per_epoch accordingly in the
	// config file
	stepsPerEpoch := cfg.Train.Params.StepsPerEpoch
	if stepsPerEpoch == 0 {
		cfg.Train.Params.StepsPerEpoch = trainData.Len() / int(cfg.Train.BatchSize)
	}

	criterion, err := b.BuildLoss()
	if err != nil {
		err = fmt.Errorf("Building loss function failed: %w", err)
		log.Fatal(err)
	}

	optimizer, err := b.BuildOptimizer(model.Weights)
	if err != nil {
		err = fmt.Errorf("Building optimizer failed: %w", err)
		log.Fatal(err)
	}

	// scheduler, err := b.BuildScheduler(optimizer)
	// if err != nil {
		// err = fmt.Errorf("Building scheduler failed: %w", err)
		// log.Fatal(err)
	// }
	scheduler := CustomScheduler(optimizer)

	var metrics []lab.Metric
	for _, name := range cfg.Evaluation.Params.Metrics{
		switch name{
			case "dice_coefficient":
				m := lab.NewDiceCoefficientMetric()
				metrics = append(metrics, m)
			case "jaccard_index":
				m := lab.NewJaccardIndexMetric()
				metrics = append(metrics, m)
			default:
				err := fmt.Errorf("Unsupported metric: %q\n", name)
				log.Fatal(err)
		}
	}
	var validMetric lab.Metric
	switch cfg.Evaluation.Params.ValidMetric{
	case "dice_coefficient":
		validMetric = lab.NewDiceCoefficientMetric()
	default:
		err := fmt.Errorf("Unsupported valid metric: %q\n", cfg.Evaluation.Params.ValidMetric)
		log.Fatal(err)
	}

	// var metrics []lab.Metric = []lab.Metric{NewDiceCoeffBatch()}
	// var validMetric lab.Metric = NewDiceCoeffBatch()
	evaluator, err := lab.NewEvaluator(cfg, validLoader, metrics, validMetric)
	if err != nil{
		log.Fatal(err)
	}
	evaluator.SetLogger(logger)

	trainer := &lab.Trainer{
		Loader:    trainLoader,
		Model:     model,
		Criterion: criterion,
		Optimizer: optimizer,
		Scheduler: scheduler,
		Evaluator: evaluator,
		Logger:    logger,
		Verbosity: cfg.Train.Params.Verbosity,
		Epochs:    cfg.Train.Params.Epochs,

		LossTracker: lab.NewLossTracker(),
		TimeTracker: lab.NewTimeTracker(),
	}

	trainer.Train(cfg)
}


