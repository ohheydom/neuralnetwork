package neuralnetwork

import (
	"log"
	"math"
	"math/rand"
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
	network.Weights, network.Biases = buildWeightsAndBiases(nSize, sizes)
	return network
}

func activate(v float64) float64 {
	return sigmoid(v)
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

// SGD (Stochastic Gradient Descent) updates the weights and biases of a model given training data (x) and labels (y)
func (network *Network) SGD(x [][]float64, y []float64, miniBatchSize int, nIter int, eta float64) {
	n := len(x)
	if miniBatchSize > n {
		log.Fatal("miniBatchSize must be smaller than the number of samples in your training data.")
	}
	batches := createBatchMarkers(n, miniBatchSize)
	for i := 0; i < nIter; i++ {
		newX, newY := randomShuffle(x, y, 4)
		for j := 0; j < len(batches)-1; j++ {
			s, e := batches[j], batches[j+1]
			network.updateWeights(newX[s:e], newY[s:e], eta)
		}
	}
}

func (network *Network) updateWeights(x [][]float64, y []float64, eta float64) {
	newWeights, newBiases := buildWeightsAndBiases(network.NLayers, network.Sizes)
	for i := 0; i < len(x); i++ {
		weightChange, biasChange := network.backPropagation(x[i], y[i])
		for j := 0; j < network.NLayers-1; j++ {
			for bc := 0; bc < len(newBiases[j]); bc++ {
				newBiases[j][bc] += biasChange[j][bc]
			}
			for k := 0; k < network.Sizes[j+1]; k++ {
				for bw := 0; bw < len(newWeights[j][k]); bw++ {
					newWeights[j][k][bw] += weightChange[j][k][bw]
				}
			}
		}
	}
	for i := 0; i < len(x); i++ {
		for j := 0; j < network.NLayers-1; j++ {
			for bc := 0; bc < len(newBiases[j]); bc++ {
				network.Biases[j][bc] -= (eta / float64(len(x))) * newBiases[j][bc]
			}
			for k := 0; k < network.Sizes[j+1]; k++ {
				for bw := 0; bw < len(newWeights[j][k]); bw++ {
					network.Weights[j][k][bw] -= (eta / float64(len(x))) * newWeights[j][k][bw]
				}
			}
		}
	}
}

func (network *Network) backPropagation(x []float64, y float64) ([][][]float64, [][]float64) {
	newWeights, newBiases := buildWeightsAndBiases(network.NLayers, network.Sizes)
	return newWeights, newBiases
}

// Helper Functions

func buildWeightsAndBiases(nLayers int, sizes []int) ([][][]float64, [][]float64) {
	weights := make([][][]float64, nLayers-1)
	biases := make([][]float64, nLayers-1)
	for j := 0; j < nLayers-1; j++ {
		biases[j] = make([]float64, sizes[j+1])
		weights[j] = make([][]float64, sizes[j+1])
		for k := 0; k < sizes[j+1]; k++ {
			weights[j][k] = make([]float64, sizes[j])
		}
	}
	return weights, biases
}

func createBatchMarkers(n, miniBatchSize int) []int {
	bSize := n / miniBatchSize
	if n%miniBatchSize != 0 {
		bSize++
	}
	batches := make([]int, bSize)
	batches = append(batches, n)
	for idx, j := 0, 0; j < n; idx, j = idx+1, j+miniBatchSize {
		batches[idx] = j
	}
	return batches
}

func randomShuffle(x [][]float64, y []float64, seed int64) ([][]float64, []float64) {
	n := len(x)
	rand.Seed(seed)
	newOrder := rand.Perm(n)
	newX, newY := make([][]float64, n), make([]float64, n)
	for i := 0; i < n; i++ {
		newX[i] = x[newOrder[i]]
		newY[i] = y[newOrder[i]]
	}
	return newX, newY
}

func sigmoid(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func sigmoidDerivative(v float64) float64 {
	return sigmoid(v) * (1 - sigmoid(v))
}
