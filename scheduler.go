package lab

import (
	"github.com/sugarme/gotch/nn"
)

type Scheduler struct {
	*nn.LRScheduler
	Update   string
	FuncName string
}
