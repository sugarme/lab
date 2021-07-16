package lab

import (
	"math"

	"github.com/sugarme/gotch/nn"
)

// LinearLR linearly increases the learning rate between two boundaries over a number ofiterations.
type LinearLR struct {
	opt        *nn.Optimizer
	initialLRs []float64
	stepCount  int
	lastEpoch  int
	totalSteps int // number of iterations
	endLR float64 // default = 10
}

// NewLinearLR creates a new LinearLR.
func NewLinearLR(opt *nn.Optimizer, totalSteps int, endLROpt ...float64) *LinearLR {
	endLR := 10.0
	if len(endLROpt) > 0{
		endLR = endLROpt[0]
	}
	initialLRs := opt.GetLRs()
	return &LinearLR{
		opt:        opt,
		initialLRs: initialLRs,
		stepCount:  0,
		lastEpoch:  -1,
		totalSteps: totalSteps,
		endLR: endLR,
	}
}

// Build implements scheduler interface.
func (l *LinearLR) Build() *nn.LRScheduler {
	s := nn.NewLRScheduler(l)
	s.Step()
	return s
}

// SetLRs implements scheduler interface by setting new LR for optimizer.
func (l *LinearLR) SetLRs(opts ...nn.SchedulerOption) {
	options := nn.DefaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		l.lastEpoch += 1
	default:
		l.lastEpoch = options.LastEpoch
	}

	var newLRs []float64 = make([]float64, len(l.initialLRs))
	switch l.lastEpoch {
	case 0:
		newLRs = l.initialLRs
	default:
		currentStep := l.lastEpoch + 1
		r := float64(currentStep) / float64(l.totalSteps)
		for i, baseLR := range l.initialLRs {
			newLR := baseLR + r*(l.endLR - baseLR)
			newLRs[i] = newLR
		}
	}

	l.opt.SetLRs(newLRs)
}

// ExponentialLR exponentially increases the learning rate between two boundaries over a number of iterations.
type ExponentialLR struct {
	opt        *nn.Optimizer
	initialLRs []float64
	stepCount  int
	lastEpoch  int
	totalSteps int // number of iterations
	endLR float64 // default = 10
}

// NewLinearLR creates a new LinearLR.
func NewExponentialLR(opt *nn.Optimizer, totalSteps int, endLROpt ...float64) *ExponentialLR {
	endLR := 10.0
	if len(endLROpt) > 0{
		endLR = endLROpt[0]
	}
	initialLRs := opt.GetLRs()
	return &ExponentialLR{
		opt:        opt,
		initialLRs: initialLRs,
		stepCount:  0,
		lastEpoch:  -1,
		totalSteps: totalSteps,
		endLR: endLR,
	}
}

// Build implements scheduler interface.
func (e *ExponentialLR) Build() *nn.LRScheduler {
	s := nn.NewLRScheduler(e)
	s.Step()
	return s
}

// SetLRs implements scheduler interface by setting new LR for optimizer.
func (e *ExponentialLR) SetLRs(opts ...nn.SchedulerOption) {
	options := nn.DefaultSchedulerOptions()
	for _, o := range opts {
		o(options)
	}
	switch options.LastEpoch {
	case -1:
		e.lastEpoch += 1
	default:
		e.lastEpoch = options.LastEpoch
	}

	var newLRs []float64 = make([]float64, len(e.initialLRs))
	switch e.lastEpoch {
	case 0:
		newLRs = e.initialLRs
	default:
		currentStep := e.lastEpoch + 1
		r := float64(currentStep) / float64(e.totalSteps)
		for i, baseLR := range e.initialLRs {
			newLR := baseLR * math.Pow(e.endLR/baseLR, r)
			newLRs[i] = newLR
		}
	}

	e.opt.SetLRs(newLRs)
}

