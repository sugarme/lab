package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/gotch/dutil"
	"github.com/sugarme/lab"
)

func trainClassification(cfg *lab.Config, data []SkinDz, folds []dutil.Fold) {
	fmt.Printf("Start training...\n")

	// Fold 1
	trainSet := getSet(folds[0].Train, data)
	validSet := getSet(folds[0].Test, data)

	trainData, err := NewSkinDataset(trainSet, cfg, true)
	if err != nil{
		log.Fatalf("Create train dataset failed: %v\n", err)
	}
	validData, err := NewSkinDataset(validSet, cfg, false)
	if err != nil{
		log.Fatalf("Create valid dataset failed: %v\n", err)
	}

	// b := lab.NewBuilder(cfg, trainData, validData)
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

	// criterion, err := b.BuildLoss()
	// if err != nil {
		// err = fmt.Errorf("Building loss function failed: %w", err)
		// log.Fatal(err)
	// }


	classes := []string{
		"MEL", 	// 0.4618
		"NV", 		// 0.0767
		"BCC", 	// 1.0000
		"AKIEC", // 1.5719
		"BKL", 	// 0.4677
		"DF", 		// 4.4700
		"VASC", 	// 3.6197
	}
	classWeights := classWeights(trainSet, classes)
	logger.Printf("class weights: %0.4f\n", classWeights)
	criterion := CustomCrossEntropyLoss(WithLossFnWeights(classWeights))

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

	var metrics []lab.Metric = []lab.Metric{NewSkinAccuracy()}
	var validMetric lab.Metric = NewSkinAccuracy()
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
