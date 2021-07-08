package lab

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/dutil"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type TimeTracker struct{
	StartTime time.Time // For calculating total training time.
	LastCheck time.Time // For calculating a period of time ie. time of one epoch training.
	LoadTime []time.Duration
	StepTime []time.Duration
}

func NewTimeTracker() *TimeTracker{
	return &TimeTracker{
		StartTime: time.Now(),
		LastCheck: time.Now(),
		LoadTime: make([]time.Duration, 0),
		StepTime: make([]time.Duration, 0),
	}
}

func (tt *TimeTracker) SetTime(dataTime, stepTime time.Duration) {
	tt.LoadTime = append(tt.LoadTime, dataTime)
	tt.StepTime = append(tt.StepTime, stepTime)
}

func(tt *TimeTracker) GetTime(unitOpt ...string) (loadTime, stepTime float64){
	unit := "minutes"
	if len(unitOpt) > 0{
		unit = unitOpt[0]
	}

	n := len(tt.LoadTime)
	var cumLoad, cumStep time.Duration
	for i := 0; i < n; i++{
		cumLoad += tt.LoadTime[i]
		cumStep += tt.StepTime[i]
	}

	switch unit{
	case "seconds":
		loadTime = cumLoad.Seconds()/float64(n)
		stepTime = cumStep.Seconds()/float64(n)
	case "minutes":
		loadTime = cumLoad.Minutes()/float64(n)
		stepTime = cumStep.Minutes()/float64(n)
	case "hours":
		loadTime = cumLoad.Hours()/float64(n)
		stepTime = cumStep.Hours()/float64(n)

	default:
		loadTime = cumLoad.Minutes()/float64(n)
		stepTime = cumStep.Minutes()/float64(n)
	}

	return loadTime, stepTime
}

type LossItem struct{
	Epoch int `json:"epoch"`
	Step int `json:"step"`
	Loss float64 `json:"loss"`
}

type LossTracker struct{
	Losses []LossItem `json:"losses"`
	ValidLosses []LossItem `json:"valid_losses"`
}

func NewLossTracker() *LossTracker{
	return &LossTracker{
		Losses: make([]LossItem, 0),
		ValidLosses: make([]LossItem, 0),
	}
}

// SetLoss set loss of a step to the tracker.
func (lt *LossTracker) SetLoss(loss float64, atStep int, epoch int) {
	lt.Losses = append(lt.Losses, LossItem{Epoch: epoch, Step: atStep, Loss: loss})
}

func (lt *LossTracker) SetValidLoss(loss float64, atStep int, epoch int) {
	lt.ValidLosses = append(lt.ValidLosses, LossItem{Epoch: epoch, Step: atStep, Loss: loss})
}

// GetAllLosses returns map of step to its average loss.
func (lt *LossTracker) GetAllLosses() map[int]float64{
	losses := make(map[int]float64, len(lt.Losses))
	for k, v := range lt.Losses{
		losses[k] = v.Loss
	}
	return losses
}

// LossAtStep return average loss at given step.
func (lt *LossTracker) LossAtStep(step int) (float64, error){
	for _, l := range lt.Losses{
		if l.Step == step{
			loss := l.Loss
			return loss, nil
		}
	}

	// Not found.
	err := fmt.Errorf("Not found. Cannot find loss for step %d\n", step)
	return 9999.0, err
}

func(lt *LossTracker) Reset(){
	lt.Losses = make([]LossItem, 0)
}

func(lt *LossTracker) SaveLossesToCSV(saveDir string) error{
	filePath := fmt.Sprintf("%s/losses.csv", saveDir)
	file, err := os.Create(filePath)
	defer file.Close()
	
	if err != nil{
		err := fmt.Errorf("Create losses csv file failed: %w\n", err)
		return err
	}

	headers := []string{"epoch","step","loss\n"}
	_, err = file.WriteString(strings.Join(headers, ","))
	if err != nil{
		return err
	}

	for n := 0; n < len(lt.Losses); n++{
		item := lt.Losses[n]
		line := fmt.Sprintf("%v,%v,%v\n", item.Epoch, item.Step, item.Loss)
		_, err := file.WriteString(line)
		if err != nil{
			return err
		}
	}

	return nil
}

func(lt *LossTracker) SaveValidLossesToCSV(saveDir string) error{
	filePath := fmt.Sprintf("%s/valid-losses.csv", saveDir)
	file, err := os.Create(filePath)
	defer file.Close()
	
	if err != nil{
		err := fmt.Errorf("Create valid losses csv file failed: %w\n", err)
		return err
	}

	headers := []string{"epoch","step","loss\n"}
	_, err = file.WriteString(strings.Join(headers, ","))
	if err != nil{
		return err
	}

	for n := 0; n < len(lt.ValidLosses); n++{
		item := lt.ValidLosses[n]
		line := fmt.Sprintf("%v,%v,%v\n", item.Epoch, item.Step, item.Loss)
		_, err := file.WriteString(line)
		if err != nil{
			return err
		}
	}

	return nil
}

func mean(vals []float64) float64{
	var cum float64
	for _, v := range vals{
		cum += v
	}
	return cum/float64(len(vals))
}

// Trainer holds data and methods to train model.
type Trainer struct {
	Loader    *dutil.DataLoader
	Model     *Model
	Optimizer *nn.Optimizer
	Scheduler *Scheduler
	Criterion func(logits, labels *ts.Tensor) *ts.Tensor // loss function
	Evaluator *Evaluator
	Logger    *Logger

	// Step
	GradientAccumulation float64
	Epochs               int
	StepsPerEpoch        int // = samples/batch-size
	ValidateInterval     int
	TotalSteps           int
	Steps                int // number of steps have been trained upto this point of time. 
	Verbosity            int
	CUDA                 bool
	AMP                  bool

	CurrentEpoch int
	TimeTracker  *TimeTracker
	LossTracker  *LossTracker
}

type TrainOptions struct {
	Verbosity    int
	CUDA         bool
	AMP          bool
}

type TrainOption func(*TrainOptions)

func defaultTrainOptions() *TrainOptions {
	return &TrainOptions{
		Verbosity:    100,
		CUDA:         true,
		AMP: false,
	}
}

func WithAMP() TrainOption {
	return func(o *TrainOptions) {
		o.AMP = true
	}
}

func (t *Trainer) Train(cfg *Config, trainOpts ...TrainOption) {
	opts := defaultTrainOptions()
	for _, o := range trainOpts {
		o(opts)
	}

	// Init
	t.GradientAccumulation = cfg.Train.Params.GradientAcc
	t.Epochs = cfg.Train.Params.Epochs
	if cfg.Train.Params.StepsPerEpoch == 0 {
		// TODO. pull request to add Loader.Len() in `gotch`
		// stepsPerEpoch = len(dataLoader)
	} else {
		t.StepsPerEpoch = cfg.Train.Params.StepsPerEpoch
	}

	t.ValidateInterval = cfg.Train.Params.ValidateInterval
	t.TotalSteps = t.StepsPerEpoch * t.Epochs
	t.Steps = 0
	t.CurrentEpoch = 0

	// Log configuration
	t.Logger.Printf("DATE: %v\n", time.Now())
	t.Logger.Printf("----------\n\n")
	t.Logger.Printf("TRAINING CONFIGURATION:\n")
	cfgMsg := fmt.Sprintf("%+v\n", cfg)
	t.Logger.Printf(cfgMsg)
	t.Logger.Printf("----------\n\n")
	epochMsg := fmt.Sprintf("Steps per epoch: %v - Epochs: %v - Total steps: %v\n", t.StepsPerEpoch, t.Epochs, t.TotalSteps)
	t.Logger.Printf(epochMsg)

	t.Logger.SendSlack("CONFIGURATION:")
	t.Logger.SendSlack(cfgMsg)
	t.Logger.SendSlack(epochMsg)

	// Start training
 	for epoch := 0; epoch < t.Epochs; epoch++{
	 for t.Loader.HasNext(){
			// Train one step
			dataStart := time.Now()
			dataItem, err := t.Loader.Next()
			if err != nil {
				err = fmt.Errorf("fetch data failed: %w\n", err)
				log.Fatal(err)
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
			// device := gotch.CPU
			input := batchTs.MustDetach(true).MustTo(device, true)
			target := labelTs.MustDetach(true).MustTo(device, true)

			dataTime := time.Since(dataStart)

			stepStart := time.Now()
			logits := t.Model.Module.ForwardT(input, true)
			loss := t.Criterion(logits, target)
			if !loss.MustRequiresGrad(){
				fmt.Printf("Reset loss required grad... done.\n")
				loss.MustRequiresGrad_(true)
			}
			t.Optimizer.BackwardStep(loss)
			lossVals := loss.Float64Values()
			// NOTE. take first element. Loss tensor has always 1 value, hasn't it?
			t.LossTracker.SetLoss(lossVals[0], t.Steps, t.CurrentEpoch)

			// Delete intermediate tensors
			input.MustDrop()
			target.MustDrop()
			logits.MustDrop()
			loss.MustDrop()


			stepTime := time.Since(stepStart)
			t.TimeTracker.SetTime(dataTime, stepTime)

			// Print progression
			if t.Steps%t.Verbosity == 0 && t.Steps > 0 {
				t.PrintProgress()
			}

			/*

			// TODO. delete this. Just for test validating
			if t.Steps%100 == 0 && t.Steps > 0{
				// Validation
				if (t.CurrentEpoch+1)%t.ValidateInterval == 0 {
					t.Logger.Println("VALIDATING...")
					validStartTime := time.Now()
					// t.Model.Eval()
					validMetric, err := t.Evaluator.Validate(t.Model, t.Criterion, t.CurrentEpoch)
					if err != nil{
						err = fmt.Errorf("Evaluator - Validate failed: %w\n", err)
						log.Fatal(err)
					}
					if t.Scheduler.Update == "on_valid"  && t.Scheduler.LRScheduler != nil {
						t.Scheduler.Step(nn.WithLoss(validMetric))
					}
					// t.Model.Train()
					t.Logger.Printf("Validation took %0.2f mins\n", time.Since(validStartTime).Minutes())
				}
			}

			*/

			t.Steps += 1
			if t.Scheduler.Update == "on_batch" && t.Scheduler.LRScheduler != nil{
				t.Scheduler.Step()
			}

		} // for loop step

		// Validation
		if (t.CurrentEpoch+1)%t.ValidateInterval == 0 {
			t.Logger.Println("VALIDATING...")
			validStartTime := time.Now()
			// t.Model.Eval()
			validMetric, validLoss, err := t.Evaluator.Validate(t.Model, t.Criterion, t.CurrentEpoch)
			if err != nil{
				err = fmt.Errorf("Evaluator - Validate failed: %w\n", err)
				log.Fatal(err)
			}
			t.LossTracker.SetValidLoss(validLoss, t.Steps, t.CurrentEpoch)
			if t.Scheduler.Update == "on_valid"  && t.Scheduler.LRScheduler != nil {
				t.Scheduler.Step(nn.WithLoss(validMetric))
			}
			// t.Model.Train()
			t.Logger.Printf("Validation took %0.2f mins\n", time.Since(validStartTime).Minutes())
		}

		// Update learning rate
		if t.Scheduler.Update == "on_epoch"  && t.Scheduler.LRScheduler != nil {
			t.Scheduler.Step()
		}
		t.CurrentEpoch += 1
		// t.Steps = 0
		t.Loader.Reset()
		t.Logger.Printf("Completed epoch %d. Taken time: %0.2f mins. Reset data loader...\n", t.CurrentEpoch, time.Since(t.TimeTracker.LastCheck).Minutes())
		t.TimeTracker.LastCheck = time.Now()

		// Reset best model if using cosine-annealing-warm-restarts
		if t.Scheduler.FuncName == "CosineAnnealingWarmRestarts"   && t.Scheduler.LRScheduler != nil{
			// TODO. How to do this.
			// if t.CurrentEpoch%t.Scheduler.T0 == 0 {
			// t.Evaluator.ResetBest()
			// }
		}
	} // for loop epoch

	t.Logger.Println("TRAINING: END")
	endMsg := fmt.Sprintf("Training took: %0.2fmins\n", time.Since(t.TimeTracker.StartTime).Minutes()) 
	t.Logger.Printf(endMsg)
	t.Logger.SendSlack(endMsg)
	t.Logger.Close()

	// Save losses to csv
	err := t.LossTracker.SaveLossesToCSV(cfg.Evaluation.Params.SaveCheckpointDir)
	if err != nil{
		fmt.Print(err)
	}
	t.LossTracker.SaveValidLossesToCSV(cfg.Evaluation.Params.SaveCheckpointDir)
	if err != nil{
		fmt.Print(err)
	}

	// Plot train and valid losses and save to a png file.
	t.makeLossGraph()
}

func (t *Trainer) SchedulerStep() {
	switch {
	case t.Scheduler.FuncName == "CosineAnnealingWarmRestarts":
		epoch := t.CurrentEpoch + t.Steps/t.StepsPerEpoch
		t.Scheduler.Step(nn.WithLastEpoch(epoch))
	default:
		t.Scheduler.Step()
	}
}

func (t *Trainer) PrintProgress() {
	currentStep := t.Steps
	n := 0
	if currentStep > t.Verbosity{
		n = currentStep - t.Verbosity
	}
	var loss float64
	for i := currentStep - 1; i >= n; i--{
		stepLoss, err := t.LossTracker.LossAtStep(i)
		if err != nil{
			t.Logger.Println(err)
		}
		loss += stepLoss
	}

	// average loss for report period
	avgLoss := loss/float64(t.Verbosity)
	loadTime, stepTime := t.TimeTracker.GetTime("seconds")

	// var lr float64 = 0.00003
	// msg := fmt.Sprintf("Epoch %2d/%d\t\tStep %5d/%d(avg. data time: %0.4fs/step, step time: %0.4fs/step)\t\t Loss %0.4f (lr %.1e)\n", t.CurrentEpoch + 1, t.Epochs,t.Steps, t.TotalSteps, loadTime,stepTime, avgLoss, lr)
	
	var lr float64 = t.Optimizer.GetLRs()[0] // For now, assuming there ONE param group!
	msg := fmt.Sprintf("Epoch %2d/%d\t\tStep %5d/%d(avg. data time: %0.4fs/step, step time: %0.4fs/step)\t\t Loss %0.4f (lr %.1e)\n", t.CurrentEpoch + 1, t.Epochs,t.Steps, t.TotalSteps, loadTime,stepTime, avgLoss, lr)
	t.Logger.Print(msg)
	t.Logger.SendSlack(msg)
}

func (t *Trainer) makeLossGraph(){
	p := plot.New()

	p.Title.Text = "Train/Valid Losses"
	p.X.Label.Text = "Steps"
	p.Y.Label.Text = "Loss"

	// train points
	trainPoints :=  make(plotter.XYs, len(t.LossTracker.Losses))
	validPoints := make(plotter.XYs, len(t.LossTracker.ValidLosses))
	for i, step := range t.LossTracker.Losses{
		trainPoints[i].X = float64(step.Step)
		trainPoints[i].Y = step.Loss 
	}

	for i, step := range t.LossTracker.ValidLosses{
		validPoints[i].X = float64(step.Step)
		validPoints[i].Y = step.Loss 
	}

	err := plotutil.AddLinePoints(p,
		"Train", trainPoints,
		"Valid", validPoints,
	)
	if err != nil {
		err := fmt.Errorf("Making train loss plot failed: %w\n", err)
		log.Fatal(err)
	}

	// Save the plot to a PNG file.
	lossFile := fmt.Sprintf("%s/loss.png", t.Evaluator.SaveCheckpointDir)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, lossFile); err != nil {
		err := fmt.Errorf("Saving train valid loss plot failed: %w\n", err)
		log.Fatal(err)
	}	
}

