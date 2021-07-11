package main

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"github.com/sugarme/gotch/dutil"
	"github.com/sugarme/lab"
)

type SkinDz struct{
	ID int
	Class string
	ClassID int
	File string
}

type Class struct{
	ID int
	Name string
	Count int
}

func preprocess(cfg *lab.Config)([]SkinDz, []dutil.Fold, error){
	dataDir := cfg.Dataset.DataDir[0]
	outDir := cfg.Evaluation.Params.SaveCheckpointDir
	csvFile := cfg.Dataset.CSVFilename


	f, err := os.Open(csvFile)
	if err != nil{
		err = fmt.Errorf("Open CSV file failed: %w\n", err)
		return nil, nil, err
	}
	defer f.Close()
	df := dataframe.ReadCSV(f)
	labelFunc := func(s series.Series) series.Series{
		name := s.Records()[0]
		labels := s.Records()[1:]
		var idx int
		for i, v := range labels{
			val, err := strconv.ParseFloat(v, 64)
			if err != nil{
				log.Fatal(err)
			}
			if val == 1 {
				idx = i
				break
			}
		}
		s = s.Subset([]int{0})
		s.Append(idx)
		imageFile := fmt.Sprintf("%s/%s.jpg", dataDir, name)
		s.Append(imageFile)
		return s
	}

	df = df.Rapply(labelFunc)
	classes := []string{
		"MEL",
		"NV",
		"BCC",
		"AKIEC",
		"BKL",
		"DF",
		"VASC",
	}

	var ds []SkinDz
	lines := df.Records()
	for i, line := range lines{
		// skip header
		if i == 0{
			continue
		}
		// name := line[0]
		label, err := strconv.Atoi(line[1])
		if err != nil{
			return nil, nil, err
		}
		file := line[2]
		ds = append(ds, SkinDz{
			ID: i - 1,
			Class: classes[label],
			ClassID: label,
			File: file,
		})
	}

	folds, err := makeFolds(ds, classes)
	if err != nil{
		return nil, nil, err
	}

	saveFile := fmt.Sprintf("%s/data.csv", outDir)
	err = saveToCSV(ds, saveFile)
	if err != nil{
		return nil, nil, err
	}

	// Print statitics
	printStat(ds, classes)

	trainSet := getSet(folds[0].Train, ds)
	validSet := getSet(folds[0].Test, ds)
	printStat(trainSet, classes)
	printStat(validSet, classes)

	return ds, folds, nil
}

func saveClassToCSV(classes map[string]Class, file string) error{
	f, err := os.Create(file)
	if err != nil{
		err = fmt.Errorf("Create csv file failed: %w\n", err)
		return err
	}
	defer f.Close()

	headers := []string{"id","class","count\n"}
	_, err = f.WriteString(strings.Join(headers, ","))
	if err != nil{
		return err
	}

	// sorting
	var classnames []string
	for n := range classes{
		classnames = append(classnames, n)
	} 
	sort.Strings(classnames)

	for _, n := range classnames{
		v := classes[n]
		l := fmt.Sprintf("%d,%s,%d\n", v.ID,n, v.Count)
		_, err = f.WriteString(l)
		if err != nil{
			return err
		}
	}

	return nil
}

func saveToCSV(data []SkinDz, file string) error{
	f, err := os.Create(file)
	if err != nil{
		err = fmt.Errorf("Create csv file failed: %w\n", err)
		return err
	}
	defer f.Close()

	headers := []string{"id","class","class_id","file\n"}
	_, err = f.WriteString(strings.Join(headers, ","))
	if err != nil{
		return err
	}

	for i:= 0; i < len(data); i++{
		d := data[i]		
		l := fmt.Sprintf("%d,%s,%d,%s\n",d.ID,d.Class, d.ClassID, d.File)
		_, err = f.WriteString(l)
		if err != nil{
			return err
		}
	}

	return nil
}

func makeFolds(data []SkinDz, classnames []string) ([]dutil.Fold, error){
	// make map of classes
	classes := make(map[string][]SkinDz, len(classnames))
	for _, name := range classnames{
		for _, d := range data{
			if d.Class == name{
				if old, ok := classes[name]; ok{
					new := append(old, d)
					classes[name] = new
				} else{
					classes[name] = []SkinDz{d}
				}
			}
		}
	}

	// make 5 folds
	nfolds := 5
	folds := make([]dutil.Fold, nfolds)
	for _, c := range classes{
		f, err := makeClassFolds(c)
		if err != nil{
			return nil, err
		}
		for i := 0; i < nfolds; i++{
			folds[i].Train = append(folds[i].Train, f[i].Train...)
			folds[i].Test = append(folds[i].Test, f[i].Test...)
		}
	}

	return folds, nil
}

func makeClassFolds(data []SkinDz)([]dutil.Fold, error){
	nfolds := 5
	kf, err := dutil.NewKFold(len(data), dutil.WithKFoldShuffle(true))
	if err != nil{
		return nil, err
	}

	folds := kf.Split()
	for i := 0; i < nfolds; i++{
		f := folds[i]
		var(
			trainIds []int
			testIds []int
		)

		for _, idx := range f.Train{
			id := data[idx].ID
			trainIds = append(trainIds, id)
		}
		for _, idx := range f.Test{
			id := data[idx].ID
			testIds = append(testIds, id)
		}

		folds[i].Train = trainIds
		folds[i].Test = testIds
	}

	return folds, nil
}

func getSet(indices []int, data []SkinDz) []SkinDz{
	ds := make([]SkinDz, len(indices))
	for i, idx := range indices{
		ds[i] = data[idx]
	}

	return ds
}

func printStat(ds []SkinDz, classNames []string){
	classes := make(map[string]int, len(classNames))
	n := len(ds)
	for _, v := range ds{
		name := v.Class
		if _, ok := classes[name]; ok{
			classes[name] += 1
		} else {
			classes[name] = 1
		}
	}

	for _, name := range classNames{
		count := classes[name]
		freq := float64(count)/float64(n)
		fmt.Printf("%-20s\t%0.4f(%4d/%d)\n", name, freq, count, n)
	}
	fmt.Println()
}

func classWeights(ds []SkinDz, classNames []string) []float64{
	classes := make(map[string]int, len(classNames))
	for _, v := range ds{
		name := v.Class
		if _, ok := classes[name]; ok{
			classes[name] += 1
		} else {
			classes[name] = 1
		}
	}

	weights := make([]float64, len(classNames))
	classWeights := lab.ClassWeights(classes)
	for i := 0; i < len(classNames); i++{
		name := classNames[i]
		weights[i] = classWeights[name]
	}

	return weights
}
