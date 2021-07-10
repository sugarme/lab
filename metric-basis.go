package lab

import (
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
)

// Accuracy calculates accuracy value.
func Accuracy(yTrue []float64, yPred []float64, nclasses int) float64{
	t := mat.NewDense(nclasses, 1, yTrue)
	p := mat.NewDense(nclasses, 1, yPred)
	return metrics.AccuracyScore(t, p, true, nil)
}

// Precision calculates precision value.
func Precision(yTrue []float64, yPred []float64, nclasses int) float64{
	t := mat.NewDense(nclasses, 1, yTrue)
	p := mat.NewDense(nclasses, 1, yPred)
	var sampleWeight []float64
	return metrics.PrecisionScore(t, p, "weighted", sampleWeight)
}

// Recall calculates recall (sensitivity) value.
func Recall(yTrue []float64, yPred []float64, nclasses int) float64{
	t := mat.NewDense(nclasses, 1, yTrue)
	p := mat.NewDense(nclasses, 1, yPred)
	var sampleWeight []float64
	return metrics.RecallScore(t, p, "weighted", sampleWeight)
}

// F1 calculates F1 score.
func F1(yTrue []float64, yPred []float64, nclasses int) float64{
	t := mat.NewDense(nclasses, 1, yTrue)
	p := mat.NewDense(nclasses, 1, yPred)
	var sampleWeight []float64
	return metrics.F1Score(t, p, "weighted", sampleWeight)
}

// AUC computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
// 
// - yTrue: an array of labels or binary label indicators. Binary and multiclass cases expect labels of 
// shape = [n_samples] while the multilabel case expect binary label indicators with shape = [n_samples, n_classes].
// - yScore: an array of target scores with shape = [n_samples] or [n_samples, n_classes].
// In binary case, it's a slice of [n_samples]. Both probability estimate and non-thresholded decision 
// values can be provided. The probability estimates correspond to the PROBABILITY OF THE CLASS WITH THE GREATER LABEL. 
// In multiclass case, it corresponds to an array of shape [n_samples, n_classes] of probability estimates. 
// The probability estimates must sum to 1 accross the possible classes. In addition, the order of 
// the class scores must correspond to the order of labels, if provided, or else to the numerical or 
// lexicographical order of the labels in yTrue array, shape = [n_samples] target scores, can either be 
// probability estimates of the positive class, confidence values, or non-thresholded measure of decisions  
// (as returned by "decision_function" on some classifiers).
// In the multilabel case, it corresponds to an array of shape (n_samples, n_classes). Probability estimates 
// are provided by the predict_proba method and the non-thresholded decision values by the decision_function method. 
// The probability estimates correspond to the PROBABILITY OF THE CLASS WITH THE GREATER LABEL for each 
// output of the classifier.
// - posLabel: float64, Label considered as positive and others are considered negative. 
func AUC(yTrueVals []float64, yScoreVals []float64, posLabel float64) float64{
	n := len(yTrueVals)
	yTrue := mat.NewDense(n, 1, yTrueVals)
	yScore := mat.NewDense(n, 1, yScoreVals)
	// false positive rate (1 - specificity), true positive rate (sensitivity)
	fpr, tpr, _ := metrics.ROCCurve(yTrue, yScore, posLabel, nil) // omit threshold return

	return metrics.AUC(fpr, tpr)
}
