package neuralnetwork

import (
	"math"
)

// Network is a struct containing a Neural Network's:
// Weights - Weights for each connection between Neurons
// Biases - Biases for each neuron
// NLayers - Number of layers including input, hidden, and output
// Sizes - Slice of how many neurons in each layer
type Network struct {
	Weights [][][]float64
	Biases  [][]float64
	NLayers int
	Sizes   []int
}

// NewNetwork builds a Neural Network from the given sizes of each layer
func NewNetwork(sizes []int) *Network {
	network := new(Network)
	nSize := len(sizes)
	network.Sizes = sizes
	network.NLayers = nSize
	network.Weights = make([][][]float64, nSize-1)
	network.Biases = make([][]float64, nSize-1)
	for j := 0; j < nSize-1; j++ {
		network.Biases[j] = make([]float64, sizes[j+1])
		network.Weights[j] = make([][]float64, sizes[j+1])
		for k := 0; k < sizes[j+1]; k++ {
			network.Weights[j][k] = make([]float64, sizes[j])
		}
	}
	return network
}

func activate(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func netInput(x []float64, w []float64, b float64) float64 {
	n := len(x)
	var total float64
	for i := 0; i < n; i++ {
		total += x[i] * w[i]
	}
	return total + b
}

// FeedForward uses the given weights and biases to calculate the function of a
func (network *Network) FeedForward(a []float64) []float64 {
	totA := a
	n := network.NLayers - 1
	for i := 0; i < n; i++ {
		newA := make([]float64, network.Sizes[i+1])
		for j := 0; j < network.Sizes[i+1]; j++ {
			newA[j] += netInput(totA, network.Weights[i][j], network.Biases[i][j])
		}
		for bb := 0; bb < len(newA); bb++ {
			newA[bb] = activate(newA[bb])
		}
		totA = newA
	}
	return totA
}
