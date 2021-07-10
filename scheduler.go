package lab

import (
	"github.com/sugarme/gotch/nn"
)

type Scheduler struct {
	*nn.LRScheduler
	Name string
	Update   string // specify when to run Scheduler.Step() to update learning rate
}

func NewScheduler(scheduler *nn.LRScheduler, name string, update string) *Scheduler{

	return &Scheduler{scheduler, name, update}
}
