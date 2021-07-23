package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

// NOTE: in HAM10000 dataset, each lesion ("lesion_id") can have more than 1 images (image_id).
// We will split data to train and valid sets. Valid set will contain only lesions with single image
// to avoid leaking from train set.
func makeTrainValid(csvFile, dataDir string, balancing bool) (trainSet, validSet []SkinDz, err error){
	// csvFile := "data/HAM10000_metadata.csv"
	f, err := os.Open(csvFile)
	if err != nil{
		err := fmt.Errorf("Read metadata file failed: %w\n", err)
		return nil, nil, err
	}
	defer f.Close()
	originDF := dataframe.ReadCSV(f, dataframe.HasHeader(true))



	// 1. Count number of images for each lesion
	headers := originDF.Names()
	var countDF  dataframe.DataFrame
	isFirst := true
	for _, n := range headers{
		if n != "lesion_id"{
			group := originDF.GroupBy("lesion_id").Aggregation([]dataframe.AggregationType{6}, []string{n})
			if isFirst {
				countDF = group
				isFirst = false
			} else {
				countDF = countDF.LeftJoin(group, "lesion_id")
			}
		}
	}

	order := dataframe.Sort("lesion_id")
	countDF.SetNames(headers...)
	countDF = countDF.Arrange(order)

	// 2. Insert a column "unique" into originDF to mark whether lesion have single or multiple images.
	uniqueCol := originDF.Select("lesion_id")
	uniqueCol.SetNames("unique")
	uniqueMarkedDF := originDF.CBind(uniqueCol)

	// 3. Filter a dataset with only lesions with single image.
	dupDF := uniqueMarkedDF.Rapply(func(row series.Series) series.Series{
		lesionID := row.Elem(0).Val()
		if isUnique(countDF, lesionID.(string)){
			row.Elem(row.Len() - 1).Set(true)
		} else {
			row.Elem(row.Len() - 1).Set(false)
		}

		return row
	})

	headers = append(headers, "unique")
	dupDF.SetNames(headers...)

	uniqueDF := dupDF.Filter(dataframe.F{
		Colidx: 0,
		Colname: "unique",
		Comparator: series.Eq,
		Comparando: true,
	})

	fmt.Printf("Origin dataset: %d\n", originDF.Nrow())
	fmt.Printf("unique dataset: %d\n", uniqueDF.Nrow())
	dxCol := uniqueDF.Col("dx")
	classes := make(map[string]int)
	for i := 0; i < dxCol.Len(); i++{
		dx := dxCol.Elem(i).Val().(string)
		if _, ok := classes[dx]; !ok{
			classes[dx] = 1
		} else {
			classes[dx] += 1
		}
	}
	for k, v := range classes{
		fmt.Printf("%s\t%4d\n", k, v)
	}


	// 4. Split uniqueDF to train, valid
	lines := uniqueDF.Select([]string{"image_id","dx"}).Records()[1:]
	var ds []SkinDz
	for i, l := range lines{
		imageId := l[0]
		dx := l[1]
		ds = append(ds, SkinDz{
			ID: i,
			ClassID: classes[dx],
			Class: dx,
			File: imageId,
		})
	}

	var classnames []string
	for k  := range classes{
		classnames = append(classnames, k)
	}
	// Make 5folds for each class and merge them
	folds, err := makeFolds(ds, classnames)
	if err != nil{
		return nil, nil, err
	}

	validIds := folds[0].Test
	validDF := uniqueDF.Subset(validIds).Select(originDF.Names())
	validLesionIds := validDF.Select(0)
	validLesionIds.SetNames("lesion_id")

	validCol := uniqueCol.Copy()
	validCol.SetNames("is_valid")
	validMarkedDF := originDF.CBind(validCol)
	validMarkedDF = validMarkedDF.Rapply(func(row series.Series) series.Series{
		lesionID := row.Elem(0).Val()
		if inValidSet(validDF, lesionID.(string)){
			row.Elem(row.Len() - 1).Set(true)
		} else{
			row.Elem(row.Len() - 1).Set(false)
		}

		return row
	})

	headers = originDF.Names()
	headers = append(headers, "is_valid")
	validMarkedDF.SetNames(headers...)

	ncols := validMarkedDF.Ncol()
	trainDF := validMarkedDF.Filter(
		dataframe.F{
			Colidx: ncols - 1,
			Colname: "is_valid", 
			Comparator: series.Eq, 
			Comparando: false,
		},
	).Select(originDF.Names())

	valid, err := makeDataset(validDF, dataDir)
	if err != nil{
		return nil, nil, err
	}

	train, err := makeDataset(trainDF, dataDir)
	if err != nil{
		return nil, nil, err
	}

	if !balancing{
		return train, valid, nil
	}

	balancedTrain := balancedSampling(train, classnames)

	return balancedTrain, valid, nil
}

func saveData(data []SkinDz, file string) error{
	f, err := os.Create(file)
	if err != nil{
		err = fmt.Errorf("saveData - create file failed: %w\n", err)
		return err
	}
	defer f.Close()
	// header
	h := fmt.Sprintf("id,class,class_id,file\n")
	_, err = f.WriteString(h)
	if err != nil{
		err = fmt.Errorf("saveData - write header failed: %w\n", err)
		return err
	}

	for _, item := range data{
		line := fmt.Sprintf("%d,%s,%d,%s\n", item.ID, item.Class, item.ClassID, item.File)
		_, err := f.WriteString(line)
		if err != nil{
			err = fmt.Errorf("saveData - write line failed: %w\n", err)
			return err
		}
	}

	return nil
}

// load data from csv file.
func loadData(file string)([]SkinDz, error){
	f, err := os.Open(file)
	if err != nil{
		err = fmt.Errorf("loadData - open file failed: %w\n", err)
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	var isHeader bool = true
	var data []SkinDz
	for sc.Scan(){
		// skip header
		if isHeader{
			_ = sc.Text()
			isHeader = false
			continue
		}
		line := sc.Text()
		fields := strings.Split(line, ",")
		id, err := strconv.Atoi(fields[0])
		if err != nil{
			err = fmt.Errorf("Convert string ID failed: %w\n", err)
			return nil, err
		}
		classId, err := strconv.Atoi(fields[2])
		if err != nil{
			err = fmt.Errorf("Convert class ID failed: %w\n", err)
			return nil, err
		}

		dz := SkinDz{
			ID: id,
			Class: fields[1],
			ClassID: classId,
			File: fields[3],
		}
		data = append(data, dz)
	}

	return data, nil
}

func isUnique(df dataframe.DataFrame, lesionID string) bool{
	fil := df.Filter(dataframe.F{
		Colidx: 0,
		Colname: "lesion_id", 
		Comparator: series.Eq, 
		Comparando: lesionID,
	})
	if fil.Nrow() == 0{
		err := fmt.Errorf("lesionID (%q) not found in the dataframe.\n", lesionID)
		log.Fatal(err)
	}

	// "image_id" index = 1
	imageCount := fil.Elem(0, 1).Val()
	if imageCount == 1{
		return true
	}
	return false
}

// returns whether lesionID in valid set.
func inValidSet(validDF dataframe.DataFrame, lesionID string) bool{
	fil := validDF.Filter(dataframe.F{
		Colidx: 0,
		Colname: "lesion_id", 
		Comparator: series.Eq, 
		Comparando: lesionID,
	})

	if fil.Nrow() > 0{
		return true
	}

	return false
}

func makeDataset(df dataframe.DataFrame, dataDir string) ([]SkinDz, error){
	classes := map[string]int{
		"mel": 0,
		"nv": 1,
		"bcc": 2,
		"akiec": 3,
		"bkl": 4,
		"df": 5,
		"vasc": 6,
	}

	var ds []SkinDz
	lines := df.Select([]string{"dx", "image_id"}).Records()
	for i, line := range lines{
		// skip header
		if i == 0{
			continue
		}
		dx := line[0]
		imageId := line[1]
		file := fmt.Sprintf("%s/%s.jpg", dataDir, imageId)
		ds = append(ds, SkinDz{
			ID: i - 1,
			Class: dx,
			ClassID: classes[dx],
			File: file,
		})
	}

	return ds, nil
}

// balancedSampling makes balanced classes in dataset by upsampling.
func balancedSampling(ds []SkinDz, classNames []string) []SkinDz{
	classes := make(map[string]int, len(classNames))
	for _, v := range ds{
		name := v.Class
		if _, ok := classes[name]; ok{
			classes[name] += 1
		} else {
			classes[name] = 1
		}
	}

	maxCount := 0
	for _, c := range classes{
		if c > maxCount {
			maxCount = c
		}
	}

	weights := make(map[string]int, len(classNames))
	for i := 0; i < len(classNames); i++{
		name := classNames[i]
		count := classes[name]
		w := float64(maxCount)/float64(count)
		weights[name] = int(w)
		// fmt.Printf("%-20s\t %4d(%0.4f)", name, int(float64(count) * w), w)
	}

	data := make(map[string][]SkinDz, len(classes))
	for _, item := range ds{
		name := item.Class
		data[name] = append(data[name], item)
	} 
	for name, weight := range weights{
		// weight - 1 : duplication times of subset
		dupTimes := weight - 1
		if dupTimes > 0{
			originSubset := data[name]
			for i := 0; i < dupTimes; i++{
				data[name] = append(data[name], originSubset...)
			}
		}
	}

	var balancedDs []SkinDz
	for _, d := range data{
		balancedDs = append(balancedDs, d...)
	}
	return balancedDs
}
