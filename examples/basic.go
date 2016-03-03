package main

import (
	"fmt"
	"github.com/ohheydom/neuralnetwork"
)

func main() {
	x := []float64{4, 5, 6, 7}
	w := []float64{1, 2, 1, 2, 2}
	v := neuralnetwork.NetInput(x, w)
	fmt.Println(neuralnetwork.Activate(v))
}
