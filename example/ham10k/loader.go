package main

import (
	"fmt"
	"os"
	"reflect"
	"strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/dutil"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
	"github.com/sugarme/gotch/vision/aug"
	"github.com/sugarme/lab"
)

type Sample struct{
	Name string
	Input string
	Target string
}

// Make train, valid datasets
func makeDatasets(dataDir string) ([]Sample, []Sample, error){
	inputDir := fmt.Sprintf("%s/trainx", dataDir)
	targetDir := fmt.Sprintf("%s/trainy", dataDir)

	var data []Sample

	f, err := os.Open(inputDir)
	if err != nil{
		err = fmt.Errorf("Opening input directory failed: %w\n", err)
		return nil, nil, err
	}

	entries, err := f.Readdir(0)
	if err != nil{
		err = fmt.Errorf("Browsing input directory failed: %w\n", err)
		return nil, nil, err
	}


	for _, entry := range entries{
		if !entry.IsDir(){
			filename := entry.Name()
			imageStem := strings.TrimPrefix(filename, "X_")
			imageName := strings.TrimSuffix(imageStem, ".bmp")

			// lookup target, if not found, skip this sample
			targetFile := fmt.Sprintf("%s/Y_%s", targetDir, imageStem)
			if _, err := os.Stat(targetFile); os.IsNotExist(err){
				fmt.Printf("Target file %q does not exist. Skip this sample %q\n", targetFile, imageName)
			}
			sample := Sample{
				Name: imageName, 
				Input: fmt.Sprintf("%s/%s", inputDir, filename),
				Target: targetFile,
			}
			data = append(data, sample)
		}
	}

	fmt.Printf("total samples: %d\n", len(data))

	// Split train:test = 3:1 (test samples = 25%)
	kf, err := dutil.NewKFold(len(data), dutil.WithNFolds(4), dutil.WithKFoldShuffle(true))
	if err != nil{
		err = fmt.Errorf("Split train/valid data  failed: %w\n", err)
		return nil, nil, err
	}

	folds := kf.Split()

	// Just take fold 0
	trainIds := folds[0].Train
	validIds := folds[0].Test

	var(
		trainData []Sample
		validData []Sample
	)

	for _, id := range trainIds{
		s := data[id]
		trainData = append(trainData, s)
	}
	for _, id := range validIds{
		s := data[id]
		validData = append(validData, s)
	}

	fmt.Printf("train samples: %d\n", len(trainData))
	fmt.Printf("valid samples: %d\n", len(validData))

	return trainData, validData, nil
}

type HamDataset struct {
	Samples []Sample
	Config lab.Transform
	Transformer aug.Transformer
	IsTrain bool
}

func (m *HamDataset) Item(idx int) (interface{}, error) {
	sample := m.Samples[idx]
	var(
		imgTs *ts.Tensor
		maskTs *ts.Tensor
		err error
	) 
	
	resize := m.Config.ResizeTo
	switch len(resize){
	case 2:
		imgTs, err = vision.LoadAndResize(sample.Input, resize[0], resize[1])
		if err != nil {
			err := fmt.Errorf("HamDataset.Item load image failed: %w\n", err)
			return nil, err
		}
		maskTs, err = vision.LoadAndResize(sample.Target, resize[0], resize[1])
		if err != nil {
			err := fmt.Errorf("HamDataset.Item load mask failed: %w\n", err)
			return nil, err
		}

	case 1:
		imgTs, err = vision.LoadAndResize(sample.Input, resize[0], resize[0])
		if err != nil {
			err := fmt.Errorf("HamDataset.Item load image failed: %w\n", err)
			return nil, err
		}
		maskTs, err = vision.LoadAndResize(sample.Target, resize[0], resize[1])
		if err != nil {
			err := fmt.Errorf("HamDataset.Item load mask failed: %w\n", err)
			return nil, err
		}

	default:
		imgTs, err = vision.Load(sample.Input)
		if err != nil {
			err := fmt.Errorf("HamDataset.Item load image failed: %w\n", err)
			return nil, err
		}
		maskTs, err = vision.Load(sample.Target)
		if err != nil {
			err := fmt.Errorf("HamDataset.Item load mask failed: %w\n", err)
			return nil, err
		}
	}

	// image
	imgTs0 := imgTs.MustTo(gotch.CudaIfAvailable(), true)
	var img *ts.Tensor
	if m.IsTrain && m.Transformer != nil{
		img = m.Transformer.Transform(imgTs0)
	} else {
		img = imgTs0.MustShallowClone()
	}
	imgTs0.MustDrop()

	// mask
	maskTs0 := maskTs.MustTo(gotch.CudaIfAvailable(), true)
	var mask *ts.Tensor
	if m.IsTrain && m.Transformer != nil{
		mask = m.Transformer.Transform(maskTs0)
	} else {
		mask = maskTs0.MustShallowClone()
	}
	maskTs0.MustDrop()


	imgTs1 := img.MustDiv1(ts.FloatScalar(255.0), true).MustTotype(gotch.Float, true)
	maskTs1 := mask.MustDiv1(ts.FloatScalar(255.0), true).MustTotype(gotch.Float, true)

	return []ts.Tensor{*imgTs1, *maskTs1}, nil
}

func (m *HamDataset) Len() int {
	return len(m.Samples)
}

func (m *HamDataset) DType() reflect.Type {
	return reflect.TypeOf(m.Samples)
}

func NewHamDataset(data []Sample, cfg *lab.Config, isTrain bool) (dutil.Dataset, error){
	var transformer aug.Transformer
	var err error
	if isTrain{
		switch cfg.Transform.Augment{
		case "RandAugment":
			transformer, err = lab.NewRandomAugment(lab.WithRandomAugmentNval(cfg.Transform.Params.N), lab.WithRandomAugmentMval(cfg.Transform.Params.M))
			if err != nil {
				return nil, err
			}

			// TODO: continue
		default:
			fmt.Printf("No transform method found...\n")
		}
	}

	return &HamDataset{
		Samples: data,
		Config: cfg.Transform,
		Transformer: transformer,
		IsTrain: isTrain,
	}, nil
}

func checkLoader(cfg *lab.Config) error{
	trainData, _, err := makeDatasets()
	if err != nil{
		return err
	}

	ds, err := NewHamDataset(trainData, cfg, true)
	if err != nil{
		return err
	}

	s, err := dutil.NewBatchSampler(ds.Len(), int(cfg.Train.BatchSize), true, true)
	if err != nil{
		return err
	}

	loader, err := dutil.NewDataLoader(ds, s)	
	if err != nil{
		return err
	}

	epochs := 6
	for e := 0; e < epochs; e++{
		count := 0
		for loader.HasNext(){
			dataItem, err := loader.Next()
			if err != nil {
				err = fmt.Errorf("fetchData failed: %w\n", err)
				return err
			}

			if count % 10 == 0 && count > 0{
				fmt.Printf("Fetched %4d batches.\n", count)
			}

			batchSize := len(dataItem.([][]ts.Tensor))
			count++


			var (
				batch  []ts.Tensor
				labels []ts.Tensor
			)
			for i := 0; i < int(batchSize); i++ {
				batch = append(batch, dataItem.([][]ts.Tensor)[i][0])
				labels = append(labels, dataItem.([][]ts.Tensor)[i][1])
			}
			batchTs := ts.MustStack(batch, 0)

			// labelTs := ts.MustStack(labels, 0).MustSqueeze(true)
			labelTs := ts.MustStack(labels, 0)

			for i := 0; i < len(batch); i++ {
				batch[i].MustDrop()
				labels[i].MustDrop()
			}

			batchTs.MustDrop()
			labelTs.MustDrop()
		}

		loader.Reset()
		// fmt.Printf("data loader: %v\n", loader)
	}

	return nil
}
