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
