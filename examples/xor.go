package main

import (
	"fmt"
	"github.com/ohheydom/neuralnetwork"
)

func main() {
	// Setup the Neural Network with proper weights and biases
	n := neuralnetwork.NewNetwork([]int{2, 2, 1})
	n.Biases = [][]float64{[]float64{-10, 30}, []float64{-30}}
	n.Weights = [][][]float64{[][]float64{[]float64{20, 20}, []float64{-20, -20}}, [][]float64{[]float64{20, 20}}}

	// Create the inputs
	w := []float64{1, 1}
	x := []float64{0, 0}
	y := []float64{1, 0}
	z := []float64{0, 1}

	// Print the results
	fmt.Printf("1 and 1 returns %.0f\n", n.FeedForward(w)[0])
	fmt.Printf("0 and 0 returns %.0f\n", n.FeedForward(x)[0])
	fmt.Printf("1 and 0 returns %.0f\n", n.FeedForward(y)[0])
	fmt.Printf("0 and 1 returns %.0f\n", n.FeedForward(z)[0])
}
