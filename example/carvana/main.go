package main

import (
	"fmt"

	"github.com/sugarme/lab"
)

func main() {

	fmt.Println("hello world!")

	b := lab.NewBuilder(nil)

	fmt.Print(b)
}
