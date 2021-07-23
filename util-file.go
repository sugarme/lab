package lab

import (
	"fmt"
	"os"
)

// MakeDir creates dir if not existing
func MakeDir(dir string) error{
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err := os.MkdirAll(dir, 0755); err != nil {
			err = fmt.Errorf("MakeDir failed: %w\n", err)
			return err
		}
	}

	return nil
}

// IsFileExist checks whether file exists with input file path.
func IsFileExist(file string) bool{
	if _, err := os.Stat(file); err == nil {
		return true
	} else if os.IsNotExist(err) {
		return false
	} else {
		// Something wrong and we want to stop now.
		err := fmt.Errorf("IsFileExist - check file exist failed: %w\n", err)
		panic(err)
	}
}
