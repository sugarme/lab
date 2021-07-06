package lab

import (
	"encoding/gob"
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
	Thresholds []float64
	CUDA bool
	Debug bool
	LabelsAvailable bool
	EarlyStopping int
}

type EvalOption func(*EvalOptions)
func defaultEvalOptions() *EvalOptions{
	return &EvalOptions{
		Thresholds: []float64{0.05, 1.05, 0.05},
		CUDA: true,
		Debug: false,
		LabelsAvailable: true,
		EarlyStopping: math.MaxInt32,
	}
}

func WithEvalThresholds(vals []float64) EvalOption{
	return func(o *EvalOptions){
		o.Thresholds = vals
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


type Predictor struct{
	Loader *dutil.DataLoader
	LabelsAvailable bool
	CUDA bool
	Debug bool
	Epoch int
}

func NewPredictor(loader *dutil.DataLoader, opts ...EvalOption) *Predictor{
	options := defaultEvalOptions()
	for _, o := range opts{
		o(options)
	}

	return &Predictor{
		Loader: loader,
		LabelsAvailable: options.LabelsAvailable,
		CUDA: options.CUDA,
		Debug: options.Debug,
		Epoch: 0,
	}
}

func(p *Predictor) Predict(model ts.ModuleT, criterion LossFunc, epoch int)(yTrue []float64, yPred [][]float64, losses []float64){
	p.Epoch = epoch	

	count := 0
	p.Loader.Reset()
	for p.Loader.HasNext(){
		if p.Debug && count > 10{
			yTrue[0] = 1.0
			yTrue[1] = 0.0
			break
		}

		dataItem, err := p.Loader.Next()
		if err != nil{
			err = fmt.Errorf("Predictor - Fetch data failed: %w\n", err)
			log.Fatal(err)
		}

		device := gotch.CPU
		if p.CUDA{
			device = gotch.CudaIfAvailable()
		}

		// NOTE. For now, batchSize = 1, dataItem has type []ts.Tensor
		input := dataItem.([]ts.Tensor)[0].MustUnsqueeze(0, true).MustTo(device, true)
		target := dataItem.([]ts.Tensor)[1].MustTo(device, true)

		output := model.ForwardT(input, false).MustDetach(true)
		loss := criterion(output, target)
		lossVal := loss.Float64Values()[0]	
		losses = append(losses, lossVal)

		yTrue = append(yTrue, target.Float64Values()[0])

		pvalues := output.MustSoftmax(1, gotch.Float, true)
		yPred = append(yPred, pvalues.Float64Values())

		loss.MustDrop()	
		pvalues.MustDrop()
		input.MustDrop()
		target.MustDrop()

		count++
	} // inf. for loop

	return yTrue, yPred, losses
}


type Evaluator struct {
	Predictor *Predictor
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
	// thresholds=np.arange(0.05, 1.05, 0.05),
	Thresholds []float64
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

func(e *Evaluator) CheckImprovement(score float64) bool{
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

func(e *Evaluator) SaveCheckpoint(weights *nn.VarStore, validMetric float64, yTrue []float64, yPred [][]float64) error{
	saveFile := fmt.Sprintf("%s_%03d_VM-%0.4f.bin", e.Prefix, e.Predictor.Epoch, validMetric)
	saveFile = strings.ToUpper(saveFile)
	saveFile = fmt.Sprintf("%s/%s", e.SaveCheckpointDir, saveFile)

	switch e.SaveBest{
	case true:
		if e.CheckImprovement(validMetric){
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

			// save prediction
			data := struct{
				yTrue []float64 `json:"y_true"`
				yPred [][]float64 `json:"y_pred"`
			}{
				yTrue: yTrue,
				yPred: yPred,
			}
			predFile := fmt.Sprintf("%s/%s", e.SaveCheckpointDir, "valid_prediction.bin")
			f, err := os.Create(predFile)
			if err != nil{
				err = fmt.Errorf("SaveCheckpoint - Create validation file to save failed: %w\n", err)
				return err
			}
			enc := gob.NewEncoder(f)
			enc.Encode(data)
			f.Close()
		}
	case false:
			// save a new best model
			err := weights.Save(saveFile)
			if err != nil{
				err = fmt.Errorf("SaveCheckpoint - Save model failed: %w\n", err)
				return err
			}

			// save prediction
			data := struct{
				yTrue []float64 `json:"y_true"`
				yPred [][]float64 `json:"y_pred"`
			}{
				yTrue: yTrue,
				yPred: yPred,
			}
			predFile := fmt.Sprintf("%s/%s", e.SaveCheckpointDir, "valid_prediction.bin")
			f, err := os.Create(predFile)
			if err != nil{
				err = fmt.Errorf("SaveCheckpoint - Create validation file to save failed: %w\n", err)
				return err
			}
			enc := gob.NewEncoder(f)
			enc.Encode(data)
			f.Close()
	}

	return nil
}

// CalculateMetrics calculates metrics specified in slice of metrics property.
// It logs calculated metrics to stdout and log file and returns valid metric.
func(e *Evaluator) CalculateMetrics(yTrue []float64, yPred [][]float64, losses []float64) (float64, error){
	// map of metric name and its value
	metrics := make(map[string]float64, 0)
	metrics["loss"] = mean(losses)

	for _, m := range e.Metrics{
		l := m.Calculate(yTrue, yPred, e.Thresholds)
		n := m.Name()
		metrics[n] = l
	}

	validMetric := metrics[e.ValidMetric.Name()]
	metrics["vm"] = validMetric
	
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

	metrics["epoch"] = float64(e.Predictor.Epoch)
	e.History = append(e.History, metrics)

	return validMetric, nil
}

// Validate validates model and return valid metric.
func (e *Evaluator) Validate(model *Model, criterion LossFunc, currentEpoch int) (validMetric, avgLoss float64, err error) {
	yTrue, yPred, losses := e.Predictor.Predict(model.Module, criterion, currentEpoch)

	validMetric, err = e.CalculateMetrics(yTrue, yPred, losses)
	if err != nil{
		return -1, -1, err
	}

	err = e.SaveCheckpoint(model.Weights, validMetric, yTrue, yPred)
	if err != nil{
		return -1, -1,err
	}

	avgLoss = mean(losses)
	return validMetric, avgLoss,  nil
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
		Predictor: NewPredictor(loader, opts...),
		Metrics: metrics,
		ValidMetric: validMetric,
		Mode: cfg.Evaluation.Params.Mode,
		Prefix: prefix,
		SaveCheckpointDir: saveCheckpointDir,
		SaveBest: saveBest,
		MetricsFile: metricsFile,
		EarlyStopping: options.EarlyStopping,
		Thresholds: options.Thresholds,
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

