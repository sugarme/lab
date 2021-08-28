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
)

type TimeTracker struct {
	StartTime time.Time // For calculating total training time.
	LastCheck time.Time // For calculating a period of time ie. time of one epoch training.
	LoadTime  []time.Duration
	StepTime  []time.Duration
}

func NewTimeTracker() *TimeTracker {
	return &TimeTracker{
		StartTime: time.Now(),
		LastCheck: time.Now(),
		LoadTime:  make([]time.Duration, 0),
		StepTime:  make([]time.Duration, 0),
	}
}

func (tt *TimeTracker) SetTime(dataTime, stepTime time.Duration) {
	tt.LoadTime = append(tt.LoadTime, dataTime)
	tt.StepTime = append(tt.StepTime, stepTime)
}

func (tt *TimeTracker) GetTime(unitOpt ...string) (loadTime, stepTime float64) {
	unit := "minutes"
	if len(unitOpt) > 0 {
		unit = unitOpt[0]
	}

	n := len(tt.LoadTime)
	var cumLoad, cumStep time.Duration
	for i := 0; i < n; i++ {
		cumLoad += tt.LoadTime[i]
		cumStep += tt.StepTime[i]
	}

	switch unit {
	case "seconds":
		loadTime = cumLoad.Seconds() / float64(n)
		stepTime = cumStep.Seconds() / float64(n)
	case "minutes":
		loadTime = cumLoad.Minutes() / float64(n)
		stepTime = cumStep.Minutes() / float64(n)
	case "hours":
		loadTime = cumLoad.Hours() / float64(n)
		stepTime = cumStep.Hours() / float64(n)

	default:
		loadTime = cumLoad.Minutes() / float64(n)
		stepTime = cumStep.Minutes() / float64(n)
	}

	return loadTime, stepTime
}

type LossItem struct {
	Epoch int     `json:"epoch"`
	Step  int     `json:"step"`
	Loss  float64 `json:"loss"`
}

type LossTracker struct {
	Losses      []LossItem `json:"losses"`
	ValidLosses []LossItem `json:"valid_losses"`
}

func NewLossTracker() *LossTracker {
	return &LossTracker{
		Losses:      make([]LossItem, 0),
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
func (lt *LossTracker) GetAllLosses() map[int]float64 {
	losses := make(map[int]float64, len(lt.Losses))
	for k, v := range lt.Losses {
		losses[k] = v.Loss
	}
	return losses
}

// LossAtStep return average loss at given step.
func (lt *LossTracker) LossAtStep(step int) (float64, error) {
	for _, l := range lt.Losses {
		if l.Step == step {
			loss := l.Loss
			return loss, nil
		}
	}

	// Not found.
	err := fmt.Errorf("Not found. Cannot find loss for step %d\n", step)
	return 9999.0, err
}

func (lt *LossTracker) Reset() {
	lt.Losses = make([]LossItem, 0)
}

func (lt *LossTracker) SaveLossesToCSV(filePath string) error {
	file, err := os.Create(filePath)
	defer file.Close()

	if err != nil {
		err := fmt.Errorf("Create losses csv file failed: %w\n", err)
		return err
	}

	headers := []string{"epoch", "step", "loss\n"}
	_, err = file.WriteString(strings.Join(headers, ","))
	if err != nil {
		return err
	}

	for n := 0; n < len(lt.Losses); n++ {
		item := lt.Losses[n]
		line := fmt.Sprintf("%v,%v,%v\n", item.Epoch, item.Step, item.Loss)
		_, err := file.WriteString(line)
		if err != nil {
			return err
		}
	}

	return nil
}

func (lt *LossTracker) SaveValidLossesToCSV(filePath string) error {
	file, err := os.Create(filePath)
	defer file.Close()

	if err != nil {
		err := fmt.Errorf("Create valid losses csv file failed: %w\n", err)
		return err
	}

	headers := []string{"epoch", "step", "loss\n"}
	_, err = file.WriteString(strings.Join(headers, ","))
	if err != nil {
		return err
	}

	for n := 0; n < len(lt.ValidLosses); n++ {
		item := lt.ValidLosses[n]
		line := fmt.Sprintf("%v,%v,%v\n", item.Epoch, item.Step, item.Loss)
		_, err := file.WriteString(line)
		if err != nil {
			return err
		}
	}

	return nil
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
	Config    *Config

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
	OffsetEpochs int
	TimeTracker  *TimeTracker
	LossTracker  *LossTracker
}

func NewTrainer(cfg *Config, loader *dutil.DataLoader, model *Model, optimizer *nn.Optimizer, scheduler *Scheduler, criterion LossFunc, evaluator *Evaluator, logger *Logger) *Trainer {
	// Init
	gradientAccum := cfg.Train.Params.GradientAcc
	var stepsPerEpoch int
	if cfg.Train.Params.StepsPerEpoch == 0 {
		stepsPerEpoch = loader.Len()
	} else {
		stepsPerEpoch = cfg.Train.Params.StepsPerEpoch
	}
	cuda := cfg.Train.Params.CUDA
	currEpoch := cfg.Train.StartEpoch
	epochs := cfg.Train.Params.Epochs
	offsetEpochs := cfg.Train.StartEpoch
	validateInterval := cfg.Train.Params.ValidateInterval
	totalSteps := stepsPerEpoch * epochs
	steps := 0
	amp := cfg.Train.Params.Amp
	verbosity := cfg.Train.Params.Verbosity
	lossTracker := NewLossTracker()
	timeTracker := NewTimeTracker()

	return &Trainer{
		Loader:    loader,
		Model:     model,
		Optimizer: optimizer,
		Scheduler: scheduler,
		Criterion: criterion,
		Evaluator: evaluator,
		Logger:    logger,
		Config:    cfg,

		// Step
		GradientAccumulation: gradientAccum,
		Epochs:               epochs,
		StepsPerEpoch:        stepsPerEpoch,
		ValidateInterval:     validateInterval,
		TotalSteps:           totalSteps,
		Steps:                steps,
		Verbosity:            verbosity,
		CUDA:                 cuda,
		AMP:                  amp,

		CurrentEpoch: currEpoch,
		OffsetEpochs: offsetEpochs,
		TimeTracker:  timeTracker,
		LossTracker:  lossTracker,
	}
}

func (t *Trainer) Train() {

	// Log configuration
	t.Logger.Printf("DATE: %v\n", time.Now())
	t.Logger.Printf("----------\n\n")
	t.Logger.Printf("TRAINING CONFIGURATION:\n")
	cfgMsg := fmt.Sprintf("%+v\n", t.Config)
	t.Logger.Printf(cfgMsg)
	t.Logger.Printf("----------\n\n")
	epochMsg := fmt.Sprintf("Sample size: %d - Steps per epoch: %v - Epochs: %v - Total steps: %v\n", t.Loader.Len(), t.StepsPerEpoch, t.Epochs, t.TotalSteps)
	t.Logger.Printf(epochMsg)

	t.Logger.SendSlack("CONFIGURATION:")
	t.Logger.SendSlack(cfgMsg)
	t.Logger.SendSlack(epochMsg)

	// Start training
	for epoch := 0; epoch < t.Epochs; epoch++ {

		var epochLosses []float64
		for t.Loader.HasNext() {
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
			if !loss.MustRequiresGrad() {
				fmt.Printf("Reset loss required grad... done.\n")
				loss.MustRequiresGrad_(true)
			}
			t.Optimizer.BackwardStep(loss)
			lossVals := loss.Float64Values()
			// NOTE. take first element. Loss tensor has always 1 value, hasn't it?
			t.LossTracker.SetLoss(lossVals[0], t.Steps, t.CurrentEpoch)
			epochLosses = append(epochLosses, lossVals[0])

			// Delete intermediate tensors
			input.MustDrop()
			target.MustDrop()
			logits.MustDrop()
			loss.MustDrop()

			// TODO. delete this. Just for test validating
			/*
				if t.Steps%10 == 0 && t.Steps > 0{
					// Validation
					if (t.CurrentEpoch+1)%t.ValidateInterval == 0 {
						t.Logger.Println("VALIDATING...")
						validStartTime := time.Now()
						t.Model.Eval()
						validMetric, validLoss, err := t.Evaluator.Validate(t.Model, t.Criterion, t.CurrentEpoch)
						if err != nil{
							err = fmt.Errorf("Evaluator - Validate failed: %w\n", err)
							log.Fatal(err)
						}
						t.Model.Train()
						t.LossTracker.SetValidLoss(validLoss, t.Steps, t.CurrentEpoch)
						if t.Scheduler.Update == "on_valid"  && t.Scheduler.LRScheduler != nil {
							t.Scheduler.Step(nn.WithLoss(validMetric))
						}
						t.Logger.Printf("Validation took %0.2f mins\n", time.Since(validStartTime).Minutes())
					}
				}
			*/

			t.Steps += 1

			stepTime := time.Since(stepStart)
			t.TimeTracker.SetTime(dataTime, stepTime)

			// Print progression
			if t.Steps%t.Verbosity == 0 && t.Steps > 0 {
				t.PrintProgress()
			}
			if t.Scheduler.Update == "on_batch" && t.Scheduler.LRScheduler != nil {
				t.Scheduler.Step()
			}

		} // for loop step

		// Log average epoch loss
		epochLoss := Mean(epochLosses)
		t.Logger.Printf("Epoch %2d/%d\t\tAvg. Loss %3.4f\n", t.CurrentEpoch+1, t.Epochs+t.OffsetEpochs, epochLoss)

		// Validation
		if (t.CurrentEpoch+1)%t.ValidateInterval == 0 {
			t.Logger.Println("VALIDATING...")
			validStartTime := time.Now()
			t.Model.Eval()
			validMetric, validLoss, err := t.Evaluator.Validate(t.Model, t.Criterion, t.CurrentEpoch)
			if err != nil {
				err = fmt.Errorf("Evaluator - Validate failed: %w\n", err)
				log.Fatal(err)
			}
			t.Model.Train()
			t.LossTracker.SetValidLoss(validLoss, t.Steps, t.CurrentEpoch)
			if t.Scheduler.Update == "on_valid" && t.Scheduler.LRScheduler != nil {
				t.Scheduler.Step(nn.WithLoss(validMetric))
			}
			t.Logger.Printf("Validation took %0.2f mins\n", time.Since(validStartTime).Minutes())

			// Early stopping
			if t.Evaluator.CheckStopping() {
				t.Logger.Printf("Training has not improved for %d consecutive epochs. Early stopping now....\n", t.Evaluator.EarlyStopping)
				break
			}
		}

		t.Loader.Reset()
		t.Logger.Printf("Completed epoch %d. Taken time: %0.2f mins. Reset data loader...\n", t.CurrentEpoch+1, time.Since(t.TimeTracker.LastCheck).Minutes())
		t.TimeTracker.LastCheck = time.Now()

		// Update learning rate
		if t.Scheduler.Update == "on_epoch" && t.Scheduler.LRScheduler != nil {
			t.Scheduler.Step()
		}

		t.CurrentEpoch += 1

		// Reset best model if using cosine-annealing-warm-restarts
		if t.Scheduler.Name == "CosineAnnealingWarmRestarts" && t.Scheduler.LRScheduler != nil {
			// TODO. How to do this.
			// if t.CurrentEpoch%t.Scheduler.T0 == 0 {
			// t.Evaluator.ResetBest()
			// }
		}
	} // for loop epoch

	// save last-epoch weights for continuing training purpose
	lastFile := fmt.Sprintf("%s/last-epoch.bin", t.Config.Evaluation.Params.SaveCheckpointDir)
	err := t.Model.Weights.Save(lastFile)
	if err != nil {
		err = fmt.Errorf("Trainer.Train - Save last model failed: %w\n", err)
		fmt.Print(err)
	}

	t.Logger.Println("TRAINING: END")
	endMsg := fmt.Sprintf("Training took: %0.2fmins\n", time.Since(t.TimeTracker.StartTime).Minutes())
	t.Logger.Printf(endMsg)
	t.Logger.SendSlack(endMsg)
	t.Logger.Close()

	// Save losses to csv
	tlossFile := fmt.Sprintf("%s/train-loss-%d.csv", t.Config.Evaluation.Params.SaveCheckpointDir, t.Config.Train.TrainCount)
	err = t.LossTracker.SaveLossesToCSV(tlossFile)
	if err != nil {
		fmt.Print(err)
	}
	vlossFile := fmt.Sprintf("%s/valid-loss-%d.csv", t.Config.Evaluation.Params.SaveCheckpointDir, t.Config.Train.TrainCount)
	t.LossTracker.SaveValidLossesToCSV(vlossFile)
	if err != nil {
		fmt.Print(err)
	}

	// // Plot train and valid losses and save to a png file.
	// gFile := fmt.Sprintf("%s/loss-%d.png", t.Config.Evaluation.Params.SaveCheckpointDir, t.Config.Train.TrainCount)
	// err = t.makeLossGraph(gFile)
	// if err != nil {
	// fmt.Print(err)
	// }
}

func (t *Trainer) SchedulerStep() {
	switch {
	case t.Scheduler.Name == "CosineAnnealingWarmRestarts":
		epoch := t.CurrentEpoch + t.Steps/t.StepsPerEpoch
		t.Scheduler.Step(nn.WithLastEpoch(epoch))
	default:
		t.Scheduler.Step()
	}
}

func (t *Trainer) PrintProgress() {
	currentStep := t.Steps
	n := 0
	if currentStep > t.Verbosity {
		n = currentStep - t.Verbosity
	}
	var loss float64
	for i := currentStep - 1; i >= n; i-- {
		stepLoss, err := t.LossTracker.LossAtStep(i)
		if err != nil {
			t.Logger.Println(err)
		}
		loss += stepLoss
	}

	// average loss for report period
	avgLoss := loss / float64(t.Verbosity)
	loadTime, stepTime := t.TimeTracker.GetTime("seconds")

	var lr float64 = t.Optimizer.GetLRs()[0] // For now, assuming there ONE param group!
	msg := fmt.Sprintf("Epoch %2d/%d\t\tStep %5d/%d(avg. data time: %0.4fs/step, step time: %0.4fs/step)\t\t Loss %0.4f (lr %.1e)\n", t.CurrentEpoch+1, t.Epochs+t.OffsetEpochs, t.Steps, t.TotalSteps, loadTime, stepTime, avgLoss, lr)
	t.Logger.Print(msg)
	t.Logger.SendSlack(msg)
}

/*

func (t *Trainer) makeLossGraph(filePath string) error {
	// train points
	tloss := lossByEpoch(t.LossTracker.Losses)
	vloss := lossByEpoch(t.LossTracker.ValidLosses)

	p := plot.New()

	p.Title.Text = "Train/Valid Losses"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"
	p.Legend.Top = true
	p.Legend.Padding = 5
	p.X.Tick.Marker = EpochTicks{}

	// train points
	trainPoints := make(plotter.XYs, len(tloss))
	validPoints := make(plotter.XYs, len(vloss))
	for i := 0; i < len(tloss); i++ {
		trainPoints[i].X = float64(i)
		trainPoints[i].Y = tloss[i]
	}

	for i := 0; i < len(vloss); i++ {
		validPoints[i].X = float64(i)
		validPoints[i].Y = vloss[i]
	}

	err := plotutil.AddLinePoints(p,
		"train", trainPoints,
		"valid", validPoints,
	)
	if err != nil {
		err := fmt.Errorf("Making train - valid loss plot failed: %w\n", err)
		return err
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, filePath); err != nil {
		err := fmt.Errorf("Saving train valid loss plot failed: %w\n", err)
		return err
	}

	return nil
}

*/

func lossByEpoch(data []LossItem) []float64 {
	currEpoch := 0
	var epochLoss []float64
	var losses []float64
	for _, item := range data {
		epoch := item.Epoch
		loss := item.Loss
		// New epoch
		if epoch != currEpoch {
			eloss := Mean(epochLoss)
			losses = append(losses, eloss)
			epochLoss = []float64{}
			epochLoss = append(epochLoss, loss)
			currEpoch = epoch
		} else {
			epochLoss = append(epochLoss, loss)
		}
	}

	return losses
}

/*

type EpochTicks struct{}

// Ticks returns Ticks in the specified range.
func (EpochTicks) Ticks(min, max float64) []plot.Tick {
	if max <= min {
		panic("illegal range")
	}
	var ticks []plot.Tick

	// label every 10 unit
	for i := min; i <= max; i++ {
		if int(i)%5 == 0 {
			ticks = append(ticks, plot.Tick{Value: i, Label: strconv.FormatFloat(i, 'f', 0, 64)})
		} else {
			ticks = append(ticks, plot.Tick{Value: i, Label: ""})
		}
	}
	return ticks
}

*/
