package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/sugarme/lab"
)


func checkData(cfg *lab.Config){

	fmt.Printf("Data dir: %q\n", cfg.Dataset.DataDir)

	trainDir := cfg.Dataset.DataDir[1]
	var trainFiles []string
	err := filepath.Walk(trainDir,
    func(path string, info os.FileInfo, err error) error {
    if err != nil {
        return err
    }

		file := info.Name()
		trainFiles = append(trainFiles, file)
    return nil
	})
	if err != nil {
			log.Println(err)
	}

	fmt.Printf("Num of train files: %d\n", len(trainFiles))

}

// read all images from subdirs.
func readFiles(rootDir string) {
	files, err := ioutil.ReadDir(rootDir)
	if err != nil{
		log.Fatal(err)
	}

	fmt.Printf("Num of files: %d\n", len(files))
}
