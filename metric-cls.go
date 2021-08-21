package lab

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// ClassificationMetrics returns precision, recall, f1-score.
// logits shape: [batch_size, n_classes]
// target shape: [batch_size]
// Ref.
// 1. https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
// 2. https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
func ClassificationMetrics(logits, target *ts.Tensor, nclasses int64) (precisionVal, recallVal, f1Val, accuracyVal float64, err error) {
	yTrueDims := target.MustSize()
	yPredDims := logits.MustSize()

	if yTrueDims[0] != yPredDims[0] {
		err = fmt.Errorf("Expected dim 0 of logits and target equal. Got logits %+v - target %+v", yPredDims, yTrueDims)
		return -1, -1, -1, -1, err
	}

	yTrue, yPred, err := YtrueYpred(logits, target, nclasses)
	if err != nil {
		return -1, -1, -1, -1, err
	}

	dtype := gotch.Float
	device := target.MustDevice()
	oneTs := ts.MustOnes([]int64{1}, dtype, device).MustTo(device, true)
	epsilon := ts.MustOfSlice([]float64{1e-7}).MustTotype(dtype, true).MustTo(device, true)
	// 1 - yTrue
	yTrue1 := oneTs.MustSub(yTrue, false)
	// 1 - yPred
	yPred1 := oneTs.MustSub(yPred, false)

	tp := yTrue.MustMul(yPred, false).MustSum(dtype, true)
	tn := yTrue1.MustMul(yPred1, false).MustSum(dtype, true)
	fp := yTrue1.MustMul(yPred, false).MustSum(dtype, true)
	fn := yTrue.MustMul(yPred1, false).MustSum(dtype, true)

	fmt.Printf("TP: %v\n", tp.Int64Values()[0])
	fmt.Printf("TN: %v\n", tn.Int64Values()[0])
	fmt.Printf("FP: %v\n", fp.Int64Values()[0])
	fmt.Printf("FN: %v\n", fn.Int64Values()[0])

	// Precision = TP/(TP + FP)
	precisionDiv := tp.MustAdd(fp, false).MustAdd(epsilon, true)
	precision := tp.MustDiv(precisionDiv, false)
	precisionDiv.MustDrop()
	precisionVal = precision.Float64Values()[0]

	// Recall = TP/(TP + FN)
	recallDiv := tp.MustAdd(fn, false).MustAdd(epsilon, true)
	recall := tp.MustDiv(recallDiv, false)
	recallVal = recall.Float64Values()[0]

	// F1 = 2 x (Precision x Recall)/(Precision + Recall)
	f1Div := precision.MustAdd(recall, false).MustAdd(epsilon, true)
	f1 := precision.MustMul(recall, false).MustMulScalar(ts.FloatScalar(2), true).MustDiv(f1Div, true)
	f1Val = f1.Float64Values()[0]

	// Accuracy = (TP + TN)/(P + N) = (TP + TN)/(TP + TN + FP + FN)
	accuracyDiv := tp.MustAdd(tn, false).MustAdd(fp, true).MustAdd(fn, true)
	accuracy := tp.MustAdd(tn, false).MustDiv(accuracyDiv, true)
	accuracyDiv.MustDrop()
	accuracyVal = accuracy.Float64Values()[0]

	// Delete intermediate tensors
	yPred.MustDrop()
	yTrue.MustDrop()
	oneTs.MustDrop()
	epsilon.MustDrop()
	yTrue1.MustDrop()
	yPred1.MustDrop()
	tp.MustDrop()
	fp.MustDrop()
	fn.MustDrop()
	precision.MustDrop()
	recall.MustDrop()
	f1.MustDrop()
	accuracy.MustDrop()

	return precisionVal, recallVal, f1Val, accuracyVal, nil
}

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

// convert logits and target tensors to 2D one-hot tensors.
//
// Outputs will have shape: [batch size, number of classes]
func YtrueYpred(logits, target *ts.Tensor, nclasses int64) (*ts.Tensor, *ts.Tensor, error) {
	dtype := gotch.Int64
	device := target.MustDevice()
	bsize := target.MustSize()[0]

	tvals := target.Int64Values()
	var pred *ts.Tensor
	if len(logits.MustSize()) == 2 {
		pred = logits.MustArgmax([]int64{1}, false, false)
	} else {
		pred = logits.MustShallowClone()
	}
	pvals := pred.Int64Values()
	pred.MustDrop()

	yTrue := ts.MustZeros([]int64{bsize, nclasses}, dtype, device)
	yPred := ts.MustZeros([]int64{bsize, nclasses}, dtype, device)
	oneTs := ts.MustOnes([]int64{1}, dtype, device)

	for i, idx := range tvals {
		view, err := MatIndex(yTrue, []int{i, int(idx)})
		if err != nil {
			return nil, nil, err
		}
		view.Copy_(oneTs)
		view.MustDrop()
	}

	for i, idx := range pvals {
		view, err := MatIndex(yPred, []int{i, int(idx)})
		if err != nil {
			return nil, nil, err
		}
		view.Copy_(oneTs)
		view.MustDrop()
	}

	oneTs.MustDrop()

	return yTrue, yPred, nil
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
	precision, _, _, _, err := ClassificationMetrics(logits, target, m.Classes)
	if err != nil {
		log.Fatal(err)
	}

	return precision
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
	_, recall, _, _, err := ClassificationMetrics(logits, target, m.Classes)
	if err != nil {
		log.Fatal(err)
	}

	return recall
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
	_, _, f1, _, err := ClassificationMetrics(logits, target, m.Classes)
	if err != nil {
		log.Fatal(err)
	}

	return f1
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
	_, _, _, accuracy, err := ClassificationMetrics(logits, target, m.Classes)
	if err != nil {
		log.Fatal(err)
	}

	return accuracy
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
}

// NewMultiClassMeter creates MultiClassMeter.
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

	return &MultiClassMeter{
		Matrix:     m,
		NumClasses: nclasses,
		Eps:        eps,
	}, nil
}

// PrintStats prints out classification metrics.
func (m *MultiClassMeter) PrintStats() error {
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

	// Header
	fmt.Printf("%20s %10s %10s %10s %10s\n", "", "precision", "recall", "f1-score", "support")
	// For individual classes
	for i := 0; i < int(m.NumClasses); i++ {
		fmt.Printf("%20d %10.3f %10.3f %10.3f %10d\n", i, precisions[i], recalls[i], f1s[i], int(sumRow[i]))
	}

	// Overall
	total := sumFloat64(sumCol)
	fmt.Println()
	fmt.Printf("%20s %10s %10s %10.3f %10.0f\n", "accuracy", "", "", accuracy, total)
	fmt.Printf("%20s %10.3f %10.3f %10.3f %10.0f\n", "macro avg", macroPrecision, macroRecall, macroF1, total)
	fmt.Printf("%20s %10.3f %10.3f %10.3f %10.0f\n", "weighted avg", weightedPrecision, weightedRecall, weightedF1, total)

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
