package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/sugarme/lab"
)

var (
	config        string
	task          string
	fold          int
	checkpoint    string
	backbone      string
	loadPrevious  string // weight file to load previous train
	batchSize     int64
	saveFile      string
	modelConfig   string  // config file for model build
	eps           float64 // epsilon for optimizer
	checkpointDir string
	fullDataset bool
	graphDir string
)

func init() {
	flag.StringVar(&config, "config", "config-baseline.yaml", "Specify config file.")
	flag.StringVar(&task, "task", "", "Specify a task to run.")
	flag.IntVar(&fold, "fold", 0, "Specify a fold to run on.")
	flag.StringVar(&checkpoint, "checkpoint", "", "Specify a checkpoint file.")
	flag.StringVar(&backbone, "backbone", "", "Specify a model backbone.")
	flag.StringVar(&loadPrevious, "load_previous", "", "Specify a file to load previous trained weights")
	flag.Int64Var(&batchSize, "batch_size", 0, "Specify batch size.")
	flag.StringVar(&saveFile, "save_file", "", "Specify full filename and path to save.")
	flag.StringVar(&modelConfig, "model_config", "", "Specify a model config file.")
	flag.Float64Var(&eps, "eps", 0.0, "Specify epsilon value.")
	flag.StringVar(&checkpointDir, "checkpoint_dir", "", "Specify checkpoint directory.")
	flag.BoolVar(&fullDataset, "full", false, "Specify whether to use full dataset or not.")
	flag.StringVar(&graphDir, "graph-dir", "", "Specify directory where graph data is.")
}

func main() {
	flag.Parse()

	cfg, err := lab.NewConfig(config)
	if err != nil {
		err = fmt.Errorf("Load config file failed: %w\n", err)
		log.Fatal(err)
	}

	if task != "predict_kfold" && backbone != "" {
		cfg.Model.Params.Backbone = backbone
	}

	if batchSize > 0 {
		cfg.Train.BatchSize = batchSize
		cfg.Evaluation.Params.SaveCheckpointDir = fmt.Sprintf("%s/bs%d", cfg.Evaluation.Params.SaveCheckpointDir, batchSize)
	}

	if fold > 0 {
		switch task {
		case "train":
			cfg.Evaluation.Params.SaveCheckpointDir = fmt.Sprintf("%s/fold%d", cfg.Evaluation.Params.SaveCheckpointDir, fold)
			seedStr := fmt.Sprintf("%v%v", cfg.Seed, fold)
			seed, err := strconv.Atoi(seedStr)
			if err != nil {
				err = fmt.Errorf("Forming seeding number from fold failed: %w", err)
				log.Fatal(err)
			}
			cfg.Seed = int64(seed)

		case "test":
			// TODO.
		}
	}

	if checkpoint != "" {
		cfg.Test.Checkpoint = checkpoint
	}

	if loadPrevious != "" {
		cfg.Train.LoadPrevious = loadPrevious
	}

	cfg.SetInferenceBatchSize()
	cfg.SetReproducibility()

	// Make directory to save checkpoints
	checkpointDir := cfg.Evaluation.Params.SaveCheckpointDir
	if _, err := os.Stat(checkpointDir); os.IsNotExist(err) {
		os.Mkdir(checkpointDir, 0775)
	}

	logger, err := lab.NewLogger()
	if err != nil {
		err = fmt.Errorf("Creating logger failed: %w\n", err)
		log.Fatal(err)
	}
	logger.Printf("Saving to %q...\n", cfg.Evaluation.Params.SaveCheckpointDir)

	switch task {
	case "train":
		train(cfg, fullDataset)

	case "check-loader":
		err := checkLoader(cfg)
		if err != nil{
			log.Fatal(err)
		}
	case "check-model":
		// checkModel("./pretrained/densenet121.bin")
		err := checkModel(cfg)
		if err != nil{
			log.Fatal(err)
		}
	case "check-data":
		_, _, err = makeFullDatasets("./data/10k")
	case "preprocess":
		// _, _, err := preprocess(cfg)
		_, _, err := makeClassificationDatasets(cfg)
		if err != nil{
			log.Fatal(err)
		}

	case "train-classification":
		// ds, folds, err := preprocess(cfg)
		// if err != nil{
			// log.Fatal(err)
		// }
		trainClassification(cfg)

	case "split-data":
		_, _, err := makeTrainValid(cfg.Dataset.CSVFilename, cfg.Dataset.DataDir[0])
		if err != nil{
			log.Fatal(err)
		}

	case "find-lr":
		err := findLR(cfg)
		if err != nil{
			log.Fatal(err)
		}

	case "make-graph":
		// err := makeLossGraphFromCSV(cfg.Evaluation.Params.SaveCheckpointDir)
		err := makeLossGraphFromCSV(graphDir)
		if err != nil{
			log.Fatal(err)
		}

	case "make-lrgraph":
		err := makeFindLRGraphFromCSV(graphDir)
		if err != nil{
			log.Fatal(err)
		}

	default:
		log.Fatalf("Unsupported task: %s\n", task)
	}
}

