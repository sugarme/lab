package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/sugarme/lab"
)

var (
	config        string
	task          string
)

func init() {
	flag.StringVar(&config, "config", "config-baseline.yaml", "Specify config file.")
	flag.StringVar(&task, "task", "", "Specify a task to run.")
}

func main() {
	flag.Parse()

	cfg, err := lab.NewConfig(config)
	if err != nil {
		err = fmt.Errorf("Load config file failed: %w\n", err)
		log.Fatal(err)
	}

	logger, err := lab.NewLogger()
	if err != nil {
		err = fmt.Errorf("Creating logger failed: %w\n", err)
		log.Fatal(err)
	}
	logger.Printf("Saving to %q...\n", cfg.Evaluation.Params.SaveCheckpointDir)

	switch task {
	case "check-data":
		checkData(cfg)

	default:
		log.Fatalf("Unsupported task: %s\n", task)
	}
}


