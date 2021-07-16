package lab

import (
	"fmt"
	"math"
	"os"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/dutil"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type history struct{
	Loss float64
	LR float64
}

// ts "github.com/sugarme/gotch/tensor"

// Learning rate range test.
// The learning rate range test increases the learning rate in a pre-training run
// between two boundaries in a linear or exponential manner. It provides valuable
// information on how well the network can be trained over a range of learning rates
// and what is the optimal learning rate.
//
// Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
//
// fastai/lr_find: https://github.com/fastai/fastai

// LRFinder is a struct to determine an initial LR.
// It is essentially similar to Trainer.
type LRFinder struct {
	Loader    *dutil.DataLoader
	Model     *Model
	Optimizer *nn.Optimizer
	Scheduler *Scheduler
	Criterion func(logits, labels *ts.Tensor) *ts.Tensor // loss function
	BestLoss float64
	CheckpointDir string
	CUDA    bool
	History []history
}

// NewLRFinder creates a new LRFinder.
func NewLRFinder(model *Model, loader *dutil.DataLoader, opt *nn.Optimizer, scheduler *Scheduler, criterion LossFunc, checkpointDir string, cudaOpt ...bool) *LRFinder{
	cuda := true
	if len(cudaOpt) > 0{
		cuda = cudaOpt[0]
	}
	return &LRFinder{
		Model: model,
		Optimizer: opt,
		Scheduler: scheduler,
		Criterion: criterion,
		BestLoss: math.Inf(1),
		CheckpointDir: checkpointDir,
		CUDA: cuda,
		History: nil,
	}
}

type lrfinderOptions struct{
	StepMode string // Learning rate policies. Either "exponential" or "linear"
	SmoothFactor float64 // Loss smoothing factor [0, 1]
	DivergeThreshold float64 // The test is stopped when loss surpasses the threshold. Default = 5
}

type LRFinderOption func(*lrfinderOptions)

func defaultLRFinderOptions() *lrfinderOptions{
	return &lrfinderOptions{
		StepMode: "exponential",
		SmoothFactor: 0.05,
		DivergeThreshold: 5.0,
	}
}

func WithLRFinderOptionStepMode(v string) LRFinderOption{
	return func(o *lrfinderOptions){
		o.StepMode = v
	}
}

func WithLRFinderOptionSmoothFactor(v float64) LRFinderOption{
	return func(o *lrfinderOptions){
		o.SmoothFactor = v
	}
}

func WithLRFinderOptionDivergeThreshold(v float64) LRFinderOption{
	return func(o *lrfinderOptions){
		o.DivergeThreshold = v
	}
}

// FindLR train data over specified start and end LR for steps then save data to CSV and (optional) plot graph of LR vs Loss.
func(fd *LRFinder) FindLR(startLR, endLR float64, totalSteps int, plotting bool, opts ...LRFinderOption) error{
	options := defaultLRFinderOptions()
	// set start learning rate
	fd.Optimizer.SetLRs([]float64{startLR})

	// Initialize learning rate policy
	switch options.StepMode{
		case "exponential":
			schedulerBuilder := NewExponentialLR(fd.Optimizer, totalSteps, endLR)
			fd.Scheduler.LRScheduler = schedulerBuilder.Build()
		case "linear":
			schedulerBuilder := NewLinearLR(fd.Optimizer, totalSteps, endLR)
			schedulerBuilder.SetLRs(nn.WithLastEpoch(totalSteps))
			fd.Scheduler.LRScheduler = schedulerBuilder.Build()
		default:
			err := fmt.Errorf("Unsupported learning rate policy: %q\n", options.StepMode)
			return err
	}

	// Make history capacity
	fd.History = make([]history, totalSteps)

	// Validate smooth factor
	if options.SmoothFactor < 0 || options.SmoothFactor >= 1{
		err := fmt.Errorf("Expect smooth factor in range [0,1). Got %v\n", options.SmoothFactor)
		return err
	}

	// Training loop
	for i := 0; i < totalSteps; i++{
		if !fd.Loader.HasNext(){
			fd.Loader.Reset()
		}

		dataItem, err := fd.Loader.Next()
		if err != nil{
			err := fmt.Errorf("fd.Loader.Next() failed: %w\n", err)
			return err
		}

		var (
			batch  []ts.Tensor
			labels []ts.Tensor
		)
		batchSize := len(dataItem.([][]ts.Tensor))	
		for i := 0; i < batchSize; i++ {
			batch = append(batch, dataItem.([][]ts.Tensor)[i][0])
			labels = append(labels, dataItem.([][]ts.Tensor)[i][1])
		}
		batchTs := ts.MustStack(batch, 0)
		labelTs := ts.MustStack(labels, 0).MustSqueeze(true)
		for i := 0; i < len(batch); i++ {
			batch[i].MustDrop()
			labels[i].MustDrop()
		}

		device := gotch.CudaIfAvailable()
		if !fd.CUDA{
			device = gotch.CPU
		}
		input := batchTs.MustDetach(true).MustTo(device, true)
		target := labelTs.MustDetach(true).MustTo(device, true)

		logits := fd.Model.Module.ForwardT(input, true)
		lossTs := fd.Criterion(logits, target)
		if !lossTs.MustRequiresGrad(){
			fmt.Printf("Reset loss required grad... done.\n")
			lossTs.MustRequiresGrad_(true)
		}
		fd.Optimizer.BackwardStep(lossTs)
		lossVals := lossTs.Float64Values()

		// Delete intermediate tensors
		input.MustDrop()
		target.MustDrop()
		logits.MustDrop()
		lossTs.MustDrop()

		// Update learning rate
		fd.Scheduler.Step()

		// Track best loss and smooth it
		loss := lossVals[0]
		lr := fd.Optimizer.GetLRs()[0]
		if i == 0{
			fd.BestLoss = loss
			fd.History[0] = history{
				Loss: loss,
				LR: lr,
			}
		} else {
			if options.SmoothFactor > 0{
				loss = options.SmoothFactor * loss + (1 - options.SmoothFactor) * fd.History[i - 1].Loss
			}
			if loss < fd.BestLoss{
				fd.BestLoss = loss
			}

			fd.History[i] = history{
				Loss: loss,
				LR: lr,
			}
		}

		// Check if loss has diverged, stop
		if loss  > options.DivergeThreshold *fd.BestLoss{
			fmt.Printf("Stopping early, the loss has diverged...\n")
			break
		}
	}

	// Save data to CSV
	csvFile, err := os.Create(fmt.Sprintf("%s/find-lr.csv", fd.CheckpointDir))
	if err != nil{
		err = fmt.Errorf("Create csv file failed: %w\n", err)
		return err
	}
	// Header
	headers := fmt.Sprintf("%s,%s,%s\n", "step", "loss", "lr")
	_, err = csvFile.WriteString(headers)
	if err != nil{
		err = fmt.Errorf("Write csv header failed: %w\n", err)
		return err
	}

	for i, hx := range fd.History{
		line := fmt.Sprintf("%v,%v,%v", i, hx.Loss, hx.LR)
		_, err := csvFile.WriteString(line)
		if err != nil{
			err = fmt.Errorf("Write csv file at step %d failed: %w\n", i, err)
			return err
		}
	}

	// Generate graph if set so.
	if plotting{
		err = plotLossLR(fd.History, fd.CheckpointDir)
		if err != nil{
			return err
		}
	}

	return nil
}

func plotLossLR(data []history, dir string) error{
	p := plot.New()

	p.Title.Text = "Loss vs. Learning Rate"
	p.X.Label.Text = "Steps"
	p.Y.Label.Text = "Loss/LR"

	points :=  make(plotter.XYs, len(data))
	for i, hx := range data{
		points[i].Y = hx.Loss
		points[i].X = hx.LR
	}

	err := plotutil.AddLinePoints(p,
		"Finding LR", points,
	)
	if err != nil {
		err := fmt.Errorf("Making loss-lr plot failed: %w\n", err)
		return err
	}

	// Save the plot to a PNG file.
	pngFile := fmt.Sprintf("%s/find-lr.png", dir)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, pngFile); err != nil {
		err := fmt.Errorf("Saving loss-lr plot failed: %w\n", err)
		return err
	}	

	return nil
}

