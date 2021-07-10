package lab

import (
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/dutil"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

type EvalOptions struct{
	Threshold float64
	CUDA bool
	Debug bool
	LabelsAvailable bool
	EarlyStopping int
}

type EvalOption func(*EvalOptions)
func defaultEvalOptions() *EvalOptions{
	return &EvalOptions{
		Threshold: 0.5,
		CUDA: true,
		Debug: false,
		LabelsAvailable: true,
		EarlyStopping: math.MaxInt32,
	}
}

func WithEvalThreshold(val float64) EvalOption{
	return func(o *EvalOptions){
		o.Threshold = val
	}
}

func WithEvalCUDA(cuda bool) EvalOption{
	return func(o *EvalOptions){
		o.CUDA = cuda
	}
}

func WithEvalDebug(debug bool) EvalOption{
	return func(o *EvalOptions){
		o.Debug = debug
	}
}

func WithEvalLabelsAvailable(l bool) EvalOption{
	return func(o *EvalOptions){
		o.LabelsAvailable = l
	}
}

func WithEvalEarlyStopping(v int) EvalOption{
	return func(o *EvalOptions){
		o.EarlyStopping = v
	}
}

func(e *Evaluator) evaluate(model ts.ModuleT, criterion LossFunc, epoch int)(map[string]float64, float64, float64){
	e.Epoch = epoch	

	metrics := make(map[string][]float64, 0)
	validMetrics := make([]float64, 0)
	losses := make([]float64, 0)

	count := 0
	e.Loader.Reset()
	for e.Loader.HasNext(){
		dataItem, err := e.Loader.Next()
		if err != nil{
			err = fmt.Errorf("Predictor - Fetch data failed: %w\n", err)
			log.Fatal(err)
		}

		device := gotch.CPU
		if e.CUDA{
			device = gotch.CudaIfAvailable()
		}

		// NOTE. For now, batchSize = 1, dataItem has type []ts.Tensor
		input := dataItem.([]ts.Tensor)[0].MustUnsqueeze(0, true).MustTo(device, true)
		target := dataItem.([]ts.Tensor)[1].MustTo(device, true)

		logits := model.ForwardT(input, false).MustDetach(true).MustSqueeze(true)

		// loss
		loss := criterion(logits, target)
		lossVal := loss.Float64Values()[0]	
		losses = append(losses, lossVal)

		// metrics
		stepMetrics, stepValidMetric, err := e.calculateMetrics(logits, target)
		for k, v := range stepMetrics{
			metrics[k] = append(metrics[k], v)
		}
		validMetrics = append(validMetrics, stepValidMetric)

		input.MustDrop()
		target.MustDrop()
		logits.MustDrop()
		loss.MustDrop()	

		count++
	} // inf. for loop

	avgMetrics := make(map[string]float64, 0)
	for k, v := range metrics{
		avgMetrics[k] = mean(v)
	}
	avgValidMetric := mean(validMetrics)
	loss := mean(losses)
	avgMetrics["loss"] = loss

	return avgMetrics, avgValidMetric, loss
}

type Evaluator struct {
	// Predictor *Predictor
	Loader *dutil.DataLoader
	LabelsAvailable bool
	CUDA bool
	Debug bool
	Epoch int
	// List of strings corresponding to desired metrics
	// These strings should correspond to function names defined in file `metrics.go`
	Metrics []Metric
	// ValidMetric should be included within metrics
	// This specifies which metric we should track for validation improvement
	ValidMetric Metric
	// Mode should be one of ['min', 'max']
	// This determines whether a lower (min) or higher (max)
	// valid_metric is considered to be better
	Mode string // TODO. change to fixed options
	// This determines by how much the valid_metric needs to improve
	// to be considered an improvement
	ImproveThresh float64
	// Specifies part of the model name
	Prefix string
	SaveCheckpointDir string
	// SaveBest = True, overwrite checkpoints if score improves
	// If False, save all checkpoints
	SaveBest bool
	MetricsFile string
	// How many epochs of no improvement do we wait before stopping training?
	EarlyStopping int
	Stopping int

	Threshold float64
	History []map[string]float64

	BestModel string
	BestScore float64

	Logger *Logger
}

func (e *Evaluator) ResetBest() {
	e.BestModel = ""
	e.BestScore = math.Inf(-1)
}

func (e *Evaluator) SetLogger(logger *Logger){
	e.Logger = logger
}

func(e *Evaluator) generateMetricsDf() dataframe.DataFrame{
	df := dataframe.New(series.New(e.History, series.String, "metrics" ))
	return df
}

func (e *Evaluator) CheckStopping() bool {
	return e.Stopping >= e.EarlyStopping
}

func(e *Evaluator) checkImprovement(score float64) bool{
	// If mode is "min", make score negative, then higher score is
	// better (i.e., -0.01 > -0.02)
	if e.Mode == "min"{
		score = -score
	}
	improved := score >= (e.BestScore + e.ImproveThresh)

	switch improved{
	case true:
		e.Stopping = 0
		e.BestScore = score
	case false:
		e.Stopping += 1
	}

	return improved
}

func(e *Evaluator) saveCheckpoint(weights *nn.VarStore, validMetric float64) error{
	saveFile := fmt.Sprintf("%s_%03d_VM-%0.4f.bin", e.Prefix, e.Epoch, validMetric)
	saveFile = strings.ToUpper(saveFile)
	saveFile = fmt.Sprintf("%s/%s", e.SaveCheckpointDir, saveFile)

	switch e.SaveBest{
	case true:
		if e.checkImprovement(validMetric){
			// delete previous saved best model
			if e.BestModel != ""{
				err := os.Remove(e.BestModel)
				if err != nil{
					err = fmt.Errorf("SaveCheckpoint - Remove old best model failed: %w\n", err)
					return err
				}
			}

			// save a new best model
			err := weights.Save(saveFile)
			if err != nil{
				err = fmt.Errorf("SaveCheckpoint - Save model failed: %w\n", err)
				return err
			}
		}
	case false:
			// save a new best model
			err := weights.Save(saveFile)
			if err != nil{
				err = fmt.Errorf("SaveCheckpoint - Save model failed: %w\n", err)
				return err
			}
	}

	return nil
}

func(e *Evaluator) calculateMetrics(logits, target *ts.Tensor) (map[string]float64, float64,error){
	// map of metric name and its value
	metrics := make(map[string]float64, 0)

	for _, m := range e.Metrics{
		l := m.Calculate(logits, target, WithMetricThreshold(e.Threshold))
		n := m.Name()
		metrics[n] = l
	}

	validMetric := metrics[e.ValidMetric.Name()]
	metrics["vm"] = validMetric

	return metrics, validMetric, nil
}

// Validate validates model and returns valid metric and loss values.
func (e *Evaluator) Validate(model *Model, criterion LossFunc, currentEpoch int) (float64, float64, error) {
	metrics, validMetric, loss := e.evaluate(model.Module, criterion, currentEpoch)

	// Log results
	var keys []string
	for k := range metrics{
		keys  = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys{
		v := metrics[k]
		msg := fmt.Sprintf("%-60s| %0.4f\n", k, v)
		e.Logger.Printf(msg)
		e.Logger.SendSlack(msg)
	}

	metrics["epoch"] = float64(e.Epoch)
	e.History = append(e.History, metrics)

	err := e.saveCheckpoint(model.Weights, validMetric)
	if err != nil{
		return -1, -1, err
	}

	return validMetric, loss,  nil
}

func NewEvaluator(cfg *Config, loader *dutil.DataLoader, metrics []Metric, validMetric Metric, opts ...EvalOption) (*Evaluator, error){
	options := defaultEvalOptions()
	for _, o := range opts{
		o(options)
	}

	saveCheckpointDir := cfg.Evaluation.Params.SaveCheckpointDir
	prefix := cfg.Evaluation.Params.Prefix
	saveBest := cfg.Evaluation.Params.SaveBest

	metricsFile := fmt.Sprintf("%s/%s", saveCheckpointDir, "metrics.csv")

	eval := &Evaluator{
		Loader: loader,
		LabelsAvailable: options.LabelsAvailable,
		CUDA: options.CUDA,
		Debug: options.Debug,
		Epoch: 0,
		Metrics: metrics,
		ValidMetric: validMetric,
		Mode: cfg.Evaluation.Params.Mode,
		Prefix: prefix,
		SaveCheckpointDir: saveCheckpointDir,
		SaveBest: saveBest,
		MetricsFile: metricsFile,
		EarlyStopping: options.EarlyStopping,
		Threshold: options.Threshold,
		History: nil,
		BestModel: "",
		BestScore: math.Inf(-1),
		Logger: nil,
	}

	// Create SaveCheckPointDir
	if eval.SaveCheckpointDir != ""{
		err := MakeDir(eval.SaveCheckpointDir)
		if err != nil{
			return nil, err
		}
	}

	eval.ResetBest()
	return eval, nil
}

func mean(vals []float64) float64{
	n := len(vals)
	var cum float64
	for _, v := range vals{
		// skip NaN value
		if math.IsNaN(v){
			n = n - 1
			continue
		}
		cum += v
	}

	if n <= 0{
		return 0
	}
	res := cum/float64(n)
	return res
}

