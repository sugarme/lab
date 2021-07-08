package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/lab"
)

func train(cfg *lab.Config) {
	fmt.Printf("Start training...\n")

	dataDir := cfg.Dataset.DataDir[0]
	trainSet, validSet, err := makeDatasets(dataDir)

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

	scheduler, err := b.BuildScheduler(optimizer)
	if err != nil {
		err = fmt.Errorf("Building scheduler failed: %w", err)
		log.Fatal(err)
	}

	var metrics []lab.Metric = []lab.Metric{NewDiceCoeffBatch()}
	var validMetric lab.Metric = NewDiceCoeffBatch()
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


