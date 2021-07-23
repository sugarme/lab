package main

import (
	"fmt"

	"github.com/sugarme/lab"
)

func checkModel(cfg *lab.Config) error {
	b := lab.NewBuilder(cfg)

	model, err := b.BuildModel()
	if err != nil {
		err = fmt.Errorf("Building model failed: %w", err)
		return err
	}

	fmt.Println(model)
	return nil
}

