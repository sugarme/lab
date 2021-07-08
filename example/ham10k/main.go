package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
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
)

func init() {
	flag.StringVar(&config, "config", "baseline.yaml", "Specify config file.")
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
			cfg.Dataset.OuterOnly = true
			cfg.Dataset.OuterFold = fold
			cfg.Evaluation.Params.SaveCheckpointDir = fmt.Sprintf("%s/fold%d", cfg.Evaluation.Params.SaveCheckpointDir, fold)
			seedStr := fmt.Sprintf("%v%v", cfg.Seed, fold)
			seed, err := strconv.Atoi(seedStr)
			if err != nil {
				err = fmt.Errorf("Forming seeding number from fold failed: %w", err)
				log.Fatal(err)
			}
			cfg.Seed = int64(seed)

		case "test":
			cfg.Dataset.OuterOnly = true
			cfg.Dataset.OuterFold = fold
			fp := filepath.Dir(cfg.Test.SaveFile)
			fn := filepath.Base(cfg.Test.SaveFile)
			cfg.Test.SaveFile = fmt.Sprintf("%s/fold%d/%s", fp, fold, fn)
		}
	}

	if checkpoint != "" {
		cfg.Test.Checkpoint = checkpoint
	}

	if loadPrevious != "" {
		cfg.Train.LoadPrevious = loadPrevious
	}

	if eps > 0 {
		cfg.Optimizer.Params.Eps = eps
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
		train(cfg)

	case "check-loader":
		err := checkLoader(cfg)
		if err != nil{
			log.Fatal(err)
		}
	case "check-model":
		err := checkModel(cfg)
		if err != nil{
			log.Fatal(err)
		}
	}
}

