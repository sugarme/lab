package lab

import (
	"testing"
)

func TestConfiguration(t *testing.T){
	file := "./config-sample.yaml"
	cfg, err := NewConfig(file)

	if err != nil{
		t.Errorf("Parse config file failed: %v\n", err)
	}

	t.Logf("%+v\n", cfg)

	tf, err := makeTransformer(cfg.Transform.Train)

	if err != nil{
		t.Errorf("Make tranformer failed: %v\n", err)
	}

	t.Log(tf)

}
