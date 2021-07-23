package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/lab"
)

// func trainClassification(cfg *lab.Config, data []SkinDz, folds []dutil.Fold) {
func trainClassification(cfg *lab.Config) {
	classes := []string{
		"mel",
		"nv",
		"bcc",
		"akiec",
		"bkl",
		"df",
		"vasc",
	}

	logFile := fmt.Sprintf("%s/log_%v.txt", cfg.Evaluation.Params.SaveCheckpointDir, time.Now())
	logger, err := lab.NewLogger(lab.WithLoggerFile(logFile), lab.WithLoggerSlackURL(cfg.SlackURL))
	if err != nil {
		err = fmt.Errorf("Creating logger failed: %w", err)
		log.Fatal(err)
	}

	var trainSet, validSet []SkinDz
	switch cfg.Train.StartEpoch{
	case 0: // train from scratch
		csvFile := cfg.Dataset.CSVFilename
		dataDir := cfg.Dataset.DataDir[0]
		trainSet, validSet, err = makeTrainValid(csvFile, dataDir, dataBalancing) // whether to balance data
		if err != nil{
			log.Fatal(err)
		}

		// Save to csv files for continuous training
		trainCSV := fmt.Sprintf("%s/train.csv", cfg.Evaluation.Params.SaveCheckpointDir)
		if lab.IsFileExist(trainCSV) {
			log.Fatalf("Trying to save CSV with name %q. File existed. \n", trainCSV)
		}
		err = saveData(trainSet, trainCSV)
		if err != nil{
			log.Fatalf("Save trainCSV failed.\n")
		}
		validCSV := fmt.Sprintf("%s/valid.csv", cfg.Evaluation.Params.SaveCheckpointDir)
		if lab.IsFileExist(validCSV) {
			log.Fatalf("Trying to save CSV with name %q. File existed. \n", validCSV)
		}
		err = saveData(validSet, validCSV)
		if err != nil{
			log.Fatalf("Save validCSV failed.\n")
		}

	default: // continue training
		trainCSV := fmt.Sprintf("%s/train.csv", cfg.Evaluation.Params.SaveCheckpointDir)
		trainSet, err = loadData(trainCSV)
		if err != nil{
			log.Fatalf("Load trainCSV failed.\n")
		}
		validCSV := fmt.Sprintf("%s/valid.csv", cfg.Evaluation.Params.SaveCheckpointDir)
		validSet, err = loadData(validCSV)
		if err != nil{
			log.Fatalf("Load validCSV failed.\n")
		}
	}


	// Log dataset summary
	logger.Printf("Train Dataset:\n")
	logger.Printf("--------------\n")
	printStat(logger, trainSet, classes)
	logger.Printf("Valid Dataset:\n")
	logger.Printf("--------------\n")
	printStat(logger, validSet, classes)

	trainData, err := NewSkinDataset(trainSet, cfg, true)
	if err != nil{
		log.Fatalf("Create train dataset failed: %v\n", err)
	}
	validData, err := NewSkinDataset(validSet, cfg, false)
	if err != nil{
		log.Fatalf("Create valid dataset failed: %v\n", err)
	}

	b := lab.NewBuilder(cfg)

	// Build data loaders
	trainLoader, err := b.BuildDataLoader(trainData, "train")
	if err != nil {
		log.Fatal(err)
	}

	validLoader, err := b.BuildDataLoader(validData, "valid")
	if err != nil {
		log.Fatal(err)
	}

	// Build model and load pretrained weights
	model, err := b.BuildModel()
	if err != nil {
		err = fmt.Errorf("Building model failed: %w", err)
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

	// Data Balancing and Loss function:
	// =================================
	/*
	var criterion lab.LossFunc
	switch dataBalancing{
	case true:
		criterion, err = b.BuildLoss()
		if err != nil {
			err = fmt.Errorf("Building loss function failed: %w", err)
			log.Fatal(err)
		}
		logger.Printf("Data balancing using upsampling...")
		logger.Printf("Using LossFunc %q...\n", cfg.Loss.Name)
	case false:
		classWeights := classWeights(trainSet, classes)
		criterion = CustomCrossEntropyLoss(WithLossFnWeights(classWeights))
		logger.Printf("Data balancing using custom CrossEntropyLoss with class weights.\n")
		logger.Printf("class weights: %0.4f\n", classWeights)
	}
	*/

	// Ref. https://github.com/skrantidatta/Attention-based-Skin-Cancer-Classification/blob/main/Ham10000%20models/Model%20without%20Soft%20Attention/ResNet34.ipynb
	classWeights := []float64{
		5.0, // mel
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
	}
	criterion := CustomCrossEntropyLoss(WithLossFnWeights(classWeights))

	// Build optimizer
	optimizer, err := b.BuildOptimizer(model.Weights)
	if err != nil {
		err = fmt.Errorf("Building optimizer failed: %w", err)
		log.Fatal(err)
	}

	//// Build LR scheduler
	// scheduler, err := b.BuildScheduler(optimizer)
	// if err != nil {
		// err = fmt.Errorf("Building scheduler failed: %w", err)
		// log.Fatal(err)
	// }
	// scheduler := CustomScheduler(optimizer)
	var scheduler *lab.Scheduler  = lab.NewScheduler(nil, "", "")

	// Build metrics
	var metrics []lab.Metric
	for _, m := range cfg.Evaluation.Params.Metrics{
		switch m{
		case "skin_accuracy":
			metric := NewSkinAccuracy()
			metrics = append(metrics, metric)

		default:
			log.Fatalf("Unsupported metric: %q\n", m)
		}
	}

	var validMetric lab.Metric
	switch cfg.Evaluation.Params.ValidMetric{
	case "skin_accuracy":
		validMetric = NewSkinAccuracy()
	case "valid_loss":
		validMetric = NewValidLoss(criterion)
	default:
		log.Fatalf("Unsupported metric: %q\n", cfg.Evaluation.Params.ValidMetric)
	}
	

	// Build Evaluator
	evaluator, err := lab.NewEvaluator(cfg, validLoader, metrics, validMetric)
	if err != nil{
		log.Fatal(err)
	}
	evaluator.SetLogger(logger)

	// Build Trainer
	trainer := &lab.Trainer{
		Loader:    trainLoader,
		Model:     model,
		Criterion: criterion,
		Optimizer: optimizer,
		Scheduler: scheduler,
		Evaluator: evaluator,
		Logger:    logger,
		Verbosity: cfg.Train.Params.Verbosity,
		CurrentEpoch: cfg.Train.StartEpoch,
		Epochs:    cfg.Train.Params.Epochs + cfg.Train.StartEpoch,

		LossTracker: lab.NewLossTracker(),
		TimeTracker: lab.NewTimeTracker(),
	}

	// Now, time to train
	fmt.Printf("Start training...From epoch: %d\n", cfg.Train.StartEpoch)
	trainer.Train(cfg)
}
