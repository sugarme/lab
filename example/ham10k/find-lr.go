package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sugarme/lab"
)

func findLR(cfg *lab.Config) error{
	csvFile := cfg.Dataset.CSVFilename
	dataDir := cfg.Dataset.DataDir[0]
	trainSet, _, err := makeTrainValid(csvFile, dataDir)
	if err != nil{
		log.Fatal(err)
	}

	trainData, err := NewSkinDataset(trainSet, cfg, true)
	if err != nil{
		log.Fatalf("Create train dataset failed: %v\n", err)
	}

	b := lab.NewBuilder(cfg)
	trainLoader, err := b.BuildDataLoader(trainData, "train")
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

	if cfg.Train.LoadPrevious != "" {
		err := model.Weights.Load(cfg.Train.LoadPrevious)
		if err != nil {
			err = fmt.Errorf("Load pretrained failed: %w", err)
			log.Fatal(err)
		}
		logger.Printf("Previously trained weights loaded from %q\n", cfg.Train.LoadPrevious)
	}

	criterion, err := b.BuildLoss()
	if err != nil {
		err = fmt.Errorf("Building loss function failed: %w", err)
		log.Fatal(err)
	}

	opt, err := b.BuildOptimizer(model.Weights)
	if err != nil {
		err = fmt.Errorf("Building optimizer failed: %w", err)
		log.Fatal(err)
	}

	startLR := cfg.FindLR.StartLR
	endLR := cfg.FindLR.EndLR
	steps := cfg.FindLR.NumIter
	saveFig := cfg.FindLR.SaveFig

	finder := lab.NewLRFinder(model, trainLoader, opt, criterion, checkpointDir)
	return finder.FindLR(startLR, endLR, steps, saveFig)
}
