package main

import (
	"fmt"
	"github.com/ohheydom/neuralnetwork"
)

func main() {
	// This will create a Neural Network with properly indexes biases and weights set to 0
	n := neuralnetwork.NewNetwork([]int{4, 6, 1})
	fmt.Println(n)
}
