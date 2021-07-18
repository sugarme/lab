package lab

import (
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/dutil"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
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
	SaveDir string
	CUDA    bool
	History []history
}

// NewLRFinder creates a new LRFinder.
func NewLRFinder(model *Model, loader *dutil.DataLoader, opt *nn.Optimizer, criterion LossFunc, saveDir string, cudaOpt ...bool) (*LRFinder, error){
	// Make SaveDir if not existing
	err := MakeDir(saveDir)
	if err != nil{
		err = fmt.Errorf("Make Directory to save data failed: %w\n", err)
		return nil, err
	}
	cuda := true
	if len(cudaOpt) > 0{
		cuda = cudaOpt[0]
	}
	return &LRFinder{
		Loader: loader,
		Model: model,
		Optimizer: opt,
		Scheduler: nil, // Will build it when calling FindLR()
		Criterion: criterion,
		BestLoss: math.Inf(1),
		SaveDir: saveDir,
		CUDA: cuda,
		History: nil,
	}, nil
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
func(fd *LRFinder) FindLR(startLR, endLR float64, totalSteps int, saveFig bool, customTicks bool,opts ...LRFinderOption) error{
	options := defaultLRFinderOptions()
	// set start learning rate
	fd.Optimizer.SetLRs([]float64{startLR})

	// Build LRScheduler
	switch options.StepMode{
		case "exponential":
			schedulerBuilder := NewExponentialLR(fd.Optimizer, totalSteps, endLR)
			scheduler := schedulerBuilder.Build()
			fd.Scheduler = NewScheduler(scheduler, "ExponentialLR", "on_epoch")
		case "linear":
			schedulerBuilder := NewLinearLR(fd.Optimizer, totalSteps, endLR)
			scheduler := schedulerBuilder.Build()
			fd.Scheduler = NewScheduler(scheduler, "LinearLR", "on_epoch")
		default:
			err := fmt.Errorf("Unsupported learning rate policy: %q\n", options.StepMode)
			return err
	}

	// Make history capacity
	// fd.History = make([]history, totalSteps)
	fd.History = make([]history, 0)

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

		// Track best loss and smooth it
		loss := lossVals[0]
		// Check if loss has diverged, stop
		if loss  > options.DivergeThreshold *fd.BestLoss{
			fmt.Printf("Stopping early at step %d/%d, the loss has diverged...\n", i, totalSteps)
			break
		}

		lr := fd.Optimizer.GetLRs()[0]
		if i == 0{
			fd.BestLoss = loss
			h := history{
				Loss: loss,
				LR: lr,
			}
			fd.History = append(fd.History, h)
		} else {
			if options.SmoothFactor > 0{
				loss = options.SmoothFactor * loss + (1 - options.SmoothFactor) * fd.History[i - 1].Loss
			}
			if loss < fd.BestLoss{
				fd.BestLoss = loss
			}

			h := history{
				Loss: loss,
				LR: lr,
			}
			fd.History = append(fd.History, h)
		}

		if i > 0 && i%10 ==0{
			fmt.Printf("Completed %03d/%d steps\n", i, totalSteps)
		}

		// Update learning rate
		fd.Scheduler.Step()

	} // for-loop train

	// Save data to CSV
	csvFile, err := os.Create(fmt.Sprintf("%s/find-lr.csv", fd.SaveDir))
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
		line := fmt.Sprintf("%v,%v,%v\n", i, hx.Loss, hx.LR)
		_, err := csvFile.WriteString(line)
		if err != nil{
			err = fmt.Errorf("Write csv file at step %d failed: %w\n", i, err)
			return err
		}
	}

	// Generate graph if set so.
	if saveFig{
		err = plotLossLR(fd.History, fd.SaveDir, customTicks)
		if err != nil{
			return err
		}
	}

	return nil
}

func plotLossLR(data []history, dir string, customTicks bool) error{
	p := plot.New()

	p.Title.Text = "Learning Rate Finding"
	p.X.Label.Text = "learning rate"
	p.Y.Label.Text = "loss"
	p.Legend.Top = true
	p.Legend.Padding = 5
	if customTicks{
		p.X.Tick.Marker = LRTicks{}
		p.X.Tick.Label.Rotation = 45
		p.X.Tick.Label.YAlign = draw.YCenter
		p.X.Tick.Label.XAlign = draw.XRight
	}
	
	points :=  make(plotter.XYs, len(data))
	for i, hx := range data{
		points[i].Y = hx.Loss
		points[i].X = hx.LR
	}

	err := plotutil.AddLinePoints(p,
		"loss vs lr", points,
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

type LRTicks struct{}

// Ticks returns Ticks in the specified range.
func (LRTicks) Ticks(min, max float64) []plot.Tick {
	if max <= min {
		panic("illegal range")
	}
	var ticks []plot.Tick

	// label every 10 unit
	count := 0
	for i := min; i <= max; i += min  {
		switch {
			case count %100 == 0:
				ticks = append(ticks, plot.Tick{Value: i, Label: strconv.FormatFloat(i, 'e', 0, 64)})
			case count %10 == 0:
				ticks = append(ticks, plot.Tick{Value: i, Label: ""})
		}
		count++
	}
	return ticks
}

