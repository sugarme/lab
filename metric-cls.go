package lab

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// MatIndex index 2D tensor.
func MatIndex(mat *ts.Tensor, index []int) (*ts.Tensor, error) {
	y := int64(index[0])
	x := int64(index[1])
	size := mat.MustSize()
	if len(size) != 2 {
		err := fmt.Errorf("MatIndex - Expected tensor is 2D got %v\n", size)
		return nil, err
	}
	if y < 0 || y >= size[0] {
		err := fmt.Errorf("MatIndex - Col index is out of range. Expected [0, %v) got %v\n", size[0], y)
		return nil, err
	}
	if x < 0 || x >= size[1] {
		err := fmt.Errorf("MatIndex - Row index is out of range. Expected [0, %v) got %v\n", size[1], x)
		return nil, err
	}

	selY := mat.MustSelect(0, y, false)
	selXY := selY.MustNarrow(0, x, 1, true)
	return selXY, nil
}

// ConfusionMatrix generates confusion matrix with colums are ground truth and rows are prediction.
func ConfusionMatrix(logits, target *ts.Tensor, nclasses int64) (*ts.Tensor, error) {
	dtype := gotch.Int64
	device := target.MustDevice()
	bsize := target.MustSize()[0]

	targetDims := target.MustSize()
	logitsDims := logits.MustSize()

	if len(targetDims) != 1 {
		err := fmt.Errorf("Expected target dim = 1. Got: %v\n", targetDims)
		return nil, err
	}

	if logitsDims[0] != targetDims[0] {
		err := fmt.Errorf("Expected dim 0 of logits and target equal. Got logits %+v - target %+v", logitsDims, targetDims)
		return nil, err
	}

	if len(logitsDims) > 2 {
		err := fmt.Errorf("Expected logits dim <= 2. Got %v\n", logitsDims)
		return nil, err
	}

	yTrues := target.Int64Values()
	var yPreds []int64
	if len(logitsDims) == 2 {
		pred := logits.MustArgmax([]int64{1}, false, false)
		yPreds = pred.Int64Values()
		pred.MustDrop()
	} else {
		yPreds = logits.Int64Values()
	}

	if len(yTrues) != len(yPreds) {
		err := fmt.Errorf("ConfusionMatrix - Expected Ytrues and yPreds are same. Got %v and %v\n", len(yTrues), len(yPreds))
		return nil, err
	}

	mat := ts.MustZeros([]int64{nclasses, nclasses}, dtype, device)

	// iterate over stack pair
	for i := 0; i < int(bsize); i++ {
		yTrue := int(yTrues[i])
		yPred := int(yPreds[i])
		matView, err := MatIndex(mat, []int{yTrue, yPred})
		if err != nil {
			err = fmt.Errorf("ConfusionMatrix - Indexing failed: %w\n", err)
			return nil, err
		}
		matView.MustAddScalar_(ts.IntScalar(1))
		matView.MustDrop()
	}

	return mat, nil
}

// MultiClassMeter is a struct to calculate multi-class classification metrics.
type MultiClassMeter struct {
	Matrix     *ts.Tensor // Confusion matrix
	NumClasses int64      // number of classes
	Eps        float64

	// metrics for each class
	precisions []float64
	recalls    []float64
	f1s        []float64

	// average metrics
	accuracy          float64
	macroPrecision    float64
	macroRecall       float64
	macroF1           float64
	weightedPrecision float64
	weightedRecall    float64
	weightedF1        float64
}

// NewMultiClassMeter creates MultiClassMeter.
// -yTrue is 1D tensor  of int64 class id of ground truth
// -yPred is 1D tensor of int64 class id of prediction
func NewMultiClassMeter(yTrue, yPred *ts.Tensor, nclasses int64, epsOpt ...float64) (*MultiClassMeter, error) {
	eps := 1e-7
	if len(epsOpt) > 0 && epsOpt[0] < 1 {
		eps = epsOpt[0]
	}
	m, err := ConfusionMatrix(yPred, yTrue, nclasses)
	if err != nil {
		err := fmt.Errorf("NewMultiClassMeter failed: %w\n", err)
		return nil, err
	}

	meter := &MultiClassMeter{
		Matrix:     m,
		NumClasses: nclasses,
		Eps:        eps,
		precisions: nil,
		recalls:    nil,
		f1s:        nil,

		accuracy:          -1,
		macroPrecision:    -1,
		macroRecall:       -1,
		macroF1:           -1,
		weightedPrecision: -1,
		weightedRecall:    -1,
		weightedF1:        -1,
	}

	// calculate all metrics.
	err = meter.calculate()
	if err != nil {
		err = fmt.Errorf("NewMultiClassMeter - calculate metrics failed: %w\n", err)
		return nil, err
	}

	return meter, nil
}

// calculate calculates all classification metrics.
func (m *MultiClassMeter) calculate() error {
	diagTs := m.Matrix.MustDiag(0, false)
	diag := diagTs.Float64Values()
	diagTs.MustDrop()
	sum0 := m.Matrix.MustSumDimIntlist([]int64{0}, false, gotch.Int64, false)
	sumCol := sum0.Float64Values()
	sum0.MustDrop()
	sum1 := m.Matrix.MustSumDimIntlist([]int64{1}, false, gotch.Int64, false)
	sumRow := sum1.Float64Values()
	sum1.MustDrop()

	var (
		precisions []float64 = make([]float64, m.NumClasses)
		recalls    []float64 = make([]float64, m.NumClasses)
		f1s        []float64 = make([]float64, m.NumClasses)
	)

	for i := 0; i < int(m.NumClasses); i++ {
		precisions[i] = diag[i] / (sumCol[i] + m.Eps)
		recalls[i] = diag[i] / (sumRow[i] + m.Eps)
		f1s[i] = (2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i] + m.Eps)
	}

	macroPrecision := sumFloat64(precisions) / float64(m.NumClasses)
	macroRecall := sumFloat64(recalls) / float64(m.NumClasses)
	macroF1 := sumFloat64(f1s) / float64(m.NumClasses)

	weightedPrecision, err := weightedMetric(precisions, sumRow)
	if err != nil {
		return err
	}
	weightedRecall, err := weightedMetric(recalls, sumRow)
	if err != nil {
		return err
	}

	weightedF1, err := weightedMetric(f1s, sumRow)
	if err != nil {
		return err
	}

	// accuracy = microF1 = microPrecision = microRecall
	// In multi-class, we consider all the correctly predicted samples to be True Positive (TP)
	// False Positive (FP) is total number of prediction errors (all non-diagonal cells) in confusion matrix.
	// False Negative (FN) = False Positive. Because each prediction error (x is misclassified as Y) is False Positive for Y
	// and a False Negative for X.
	sumAllTs := m.Matrix.MustSum(gotch.Int64, false)
	sumAll := sumAllTs.Float64Values()[0]
	sumAllTs.MustDrop()
	tp := sumFloat64(diag)
	fp := sumAll - tp
	accuracy := tp / (tp + fp)

	m.precisions = precisions
	m.recalls = recalls
	m.f1s = f1s
	m.accuracy = accuracy
	m.macroPrecision = macroPrecision
	m.macroRecall = macroRecall
	m.macroF1 = macroF1
	m.weightedPrecision = weightedPrecision
	m.weightedRecall = weightedRecall
	m.weightedF1 = weightedF1

	return nil
}

// PrintStats prints out classification metrics.
func (m *MultiClassMeter) PrintStats() error {
	sum0 := m.Matrix.MustSumDimIntlist([]int64{0}, false, gotch.Int64, false)
	sumCol := sum0.Float64Values()
	sum0.MustDrop()
	sum1 := m.Matrix.MustSumDimIntlist([]int64{1}, false, gotch.Int64, false)
	sumRow := sum1.Float64Values()
	sum1.MustDrop()

	// Header
	fmt.Printf("%20s %10s %10s %10s %10s\n", "", "precision", "recall", "f1-score", "support")
	// For individual classes
	for i := 0; i < int(m.NumClasses); i++ {
		fmt.Printf("%20d %10.3f %10.3f %10.3f %10d\n", i, m.precisions[i], m.recalls[i], m.f1s[i], int(sumRow[i]))
	}

	// Overall
	total := sumFloat64(sumCol)
	fmt.Println()
	fmt.Printf("%20s %10s %10s %10.3f %10.0f\n", "accuracy", "", "", m.accuracy, total)
	fmt.Printf("%20s %10.3f %10.3f %10.3f %10.0f\n", "macro avg", m.macroPrecision, m.macroRecall, m.macroF1, total)
	fmt.Printf("%20s %10.3f %10.3f %10.3f %10.0f\n", "weighted avg", m.weightedPrecision, m.weightedRecall, m.weightedF1, total)

	return nil
}

func sumFloat64(vals []float64) float64 {
	var sum float64
	for _, v := range vals {
		sum += v
	}

	return sum
}

func weightedMetric(vals []float64, counts []float64) (float64, error) {
	if len(vals) != len(counts) {
		err := fmt.Errorf("weightedMetric - Expected equal vals and counts. Got: %v and %v\n", len(vals), len(counts))
		return -1, err
	}

	var sum float64
	for i := 0; i < len(counts); i++ {
		sum += vals[i] * counts[i]
	}

	return sum / sumFloat64(counts), nil
}

// ConfusionMatrix prints out confusion matrix.
func (m *MultiClassMeter) ConfusionMatrix() {
	mat := m.Matrix

	// Header
	header := fmt.Sprintf("%20s ", "")
	for i := 0; i < int(m.NumClasses); i++ {
		hcell := fmt.Sprintf("%10d ", i)
		header = header + hcell
	}

	// Sum
	hSum := fmt.Sprintf("%10s", "total")
	header = header + hSum
	fmt.Println(header)
	fmt.Println()

	// Body
	for i := 0; i < int(m.NumClasses); i++ {
		row := mat.MustSelect(0, int64(i), false)
		rowVals := row.Int64Values()
		row.MustDrop()
		printRow(fmt.Sprintf("%v", i), rowVals)
	}

	colSumsTs := mat.MustSumDimIntlist([]int64{0}, false, gotch.Int64, false)
	colSums := colSumsTs.Int64Values()
	colSumsTs.MustDrop()

	// Footer
	fmt.Println()
	printRow("total", colSums)
}

func printRow(rname string, vals []int64) {
	var total int64
	row := fmt.Sprintf("%20s ", rname)
	for i := 0; i < len(vals); i++ {
		hcell := fmt.Sprintf("%10d ", vals[i])
		row = row + hcell
		total += vals[i]
	}

	sum := fmt.Sprintf("%10d", total)
	row = row + sum

	fmt.Println(row)
}

// Precision:
// ==========

type PrecisionMeter struct {
	Classes int64 // number of classes
}

func NewPrecisionMeter(n int) Metric {
	return &PrecisionMeter{
		Classes: int64(n),
	}
}

// Calculate implements Metric interface.
func (m *PrecisionMeter) Calculate(logits, target *ts.Tensor, opts ...MetricOption) float64 {

	options := defaultMetricOptions()
	for _, o := range opts {
		o(options)
	}

	meter, err := NewMultiClassMeter(target, logits, m.Classes)
	if err != nil {
		err = fmt.Errorf("PrecisionMeter - Calculate metrics failed: %w\n", err)
		log.Fatal(err)
	}

	var retVal float64
	switch options.AverageMode {
	case "micro":
		retVal = meter.accuracy
	case "macro":
		retVal = meter.macroPrecision
	case "weighted":
		retVal = meter.weightedPrecision
	default:
		retVal = meter.macroPrecision
	}

	// Delete tensor
	meter.Matrix.MustDrop()
	meter = nil

	return retVal
}

func (m *PrecisionMeter) Name() string {
	return "precision"
}

// Recall (Sensivity):
// ===================

type RecallMeter struct {
	Classes int64 // number of classes
}

func NewRecallMeter(n int) Metric {
	return &RecallMeter{
		Classes: int64(n),
	}
}

// Calculate implements Metric interface.
func (m *RecallMeter) Calculate(logits, target *ts.Tensor, opts ...MetricOption) float64 {

	options := defaultMetricOptions()
	for _, o := range opts {
		o(options)
	}

	meter, err := NewMultiClassMeter(target, logits, m.Classes)
	if err != nil {
		err = fmt.Errorf("PrecisionMeter - Calculate metrics failed: %w\n", err)
		log.Fatal(err)
	}

	var retVal float64
	switch options.AverageMode {
	case "micro":
		retVal = meter.accuracy
	case "macro":
		retVal = meter.macroRecall
	case "weighted":
		retVal = meter.weightedRecall
	default:
		retVal = meter.macroRecall
	}

	// Delete tensor
	meter.Matrix.MustDrop()
	meter = nil

	return retVal
}

func (m *RecallMeter) Name() string {
	return "recall"
}

// F1-Score:
// =========

type F1Meter struct {
	Classes int64 // number of classes
}

func NewF1Meter(n int) Metric {
	return &F1Meter{
		Classes: int64(n),
	}
}

// Calculate implements Metric interface.
func (m *F1Meter) Calculate(logits, target *ts.Tensor, opts ...MetricOption) float64 {
	options := defaultMetricOptions()
	for _, o := range opts {
		o(options)
	}

	meter, err := NewMultiClassMeter(target, logits, m.Classes)
	if err != nil {
		err = fmt.Errorf("PrecisionMeter - Calculate metrics failed: %w\n", err)
		log.Fatal(err)
	}

	var retVal float64
	switch options.AverageMode {
	case "micro":
		retVal = meter.accuracy
	case "macro":
		retVal = meter.macroF1
	case "weighted":
		retVal = meter.weightedF1
	default:
		retVal = meter.macroF1
	}

	// Delete tensor
	meter.Matrix.MustDrop()
	meter = nil

	return retVal
}

func (m *F1Meter) Name() string {
	return "f1"
}

// Accuracy:
// =========

type AccuracyMeter struct {
	Classes int64 // number of classes
}

func NewAccuracyMeter(n int) Metric {
	return &AccuracyMeter{
		Classes: int64(n),
	}
}

// Calculate implements Metric interface.
func (m *AccuracyMeter) Calculate(logits, target *ts.Tensor, opts ...MetricOption) float64 {
	meter, err := NewMultiClassMeter(target, logits, m.Classes)
	if err != nil {
		err = fmt.Errorf("PrecisionMeter - Calculate metrics failed: %w\n", err)
		log.Fatal(err)
	}

	// Delete tensor
	retVal := meter.accuracy
	meter.Matrix.MustDrop()
	meter = nil

	return retVal
}

func (m *AccuracyMeter) Name() string {
	return "accuracy"
}

// Loss:
// =====

type LossMeter struct {
	lossFunc LossFunc
}

func NewLossMeter(lossFunc LossFunc) Metric {
	return &LossMeter{lossFunc}
}

func (m *LossMeter) Calculate(logits, target *ts.Tensor, opts ...MetricOption) float64 {
	lossTs := m.lossFunc(logits, target)
	loss := lossTs.Float64Values()[0]
	lossTs.MustDrop()
	return loss
}

func (m *LossMeter) Name() string {
	return "valid_loss"
}
