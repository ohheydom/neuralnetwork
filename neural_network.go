package neuralnetwork

import (
	"log"
	"math"
	"math/rand"
	"time"
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
	network.Weights, network.Biases = initializeWeightsAndBiases(nSize, sizes, false)
	return network
}

// FeedForward uses the given weights and biases to calculate the function of a
func (network *Network) FeedForward(a []float64) []float64 {
	activation := a
	for i, n := 0, network.NLayers-1; i < n; i++ {
		tempA := make([]float64, network.Sizes[i+1])
		for j := 0; j < network.Sizes[i+1]; j++ {
			v := netInput(activation, network.Weights[i][j], network.Biases[i][j])
			tempA[j] = activate(v)
		}
		activation = tempA
	}
	return activation
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
	nablaW, nablaB := initializeWeightsAndBiases(network.NLayers, network.Sizes, true)
	for i := 0; i < len(x); i++ {
		weightChange, biasChange := network.BackPropagation(x[i], y[i])
		for j := 0; j < network.NLayers-1; j++ {
			for bc := 0; bc < len(biasChange[j]); bc++ {
				nablaB[j][bc] += biasChange[j][bc]
			}
			for k := 0; k < network.Sizes[j+1]; k++ {
				for bw := 0; bw < len(weightChange[j][k]); bw++ {
					nablaW[j][k][bw] += weightChange[j][k][bw]
				}
			}
		}
	}
	for i := 0; i < len(x); i++ {
		for j := 0; j < network.NLayers-1; j++ {
			for bc := 0; bc < len(nablaB[j]); bc++ {
				network.Biases[j][bc] += (eta / float64(len(x))) * nablaB[j][bc]
			}
			for k := 0; k < network.Sizes[j+1]; k++ {
				for bw := 0; bw < len(nablaW[j][k]); bw++ {
					network.Weights[j][k][bw] += (eta / float64(len(x))) * nablaW[j][k][bw]
				}
			}
		}
	}
}

func (network *Network) BackPropagation(x []float64, y float64) ([][][]float64, [][]float64) {
	newWeights, newBiases := initializeWeightsAndBiases(network.NLayers, network.Sizes, true)
	activations := [][]float64{x}
	sigmoidPrimes := [][]float64{}
	zValues := [][][]float64{}
	activation := x
	for i := 0; i < network.NLayers-1; i++ {
		newAcLayer := []float64{}
		newZLayer := [][]float64{}
		newSigmoidPrimeLayer := []float64{}
		for j := 0; j < network.Sizes[i+1]; j++ {
			var overallZ float64
			newZNode := []float64{}
			for k := 0; k < len(activation); k++ {
				zVal := activation[k]*network.Weights[i][j][k] + network.Biases[i][j]
				newZNode = append(newZNode, zVal)
				overallZ += zVal
			}
			activatedNode := activate(overallZ)
			newSigmoidPrimeLayer = append(newSigmoidPrimeLayer, sigmoidDerivative(activatedNode))
			newAcLayer = append(newAcLayer, activatedNode)
			newZLayer = append(newZLayer, newZNode)
		}
		activation = newAcLayer
		activations = append(activations, activation)
		zValues = append(zValues, newZLayer)
		sigmoidPrimes = append(sigmoidPrimes, newSigmoidPrimeLayer)
	}
	cost := costDerivative(activations[len(activations)-1], y)
	errorLastLayer := hadamardVector(cost, sigmoidPrimes[len(sigmoidPrimes)-1])
	newBiases[len(newBiases)-1] = errorLastLayer
	newWeights[len(newWeights)-1] = multiplyVectors(activations[len(activations)-2], errorLastLayer)
	for k := network.NLayers - 2; k > 0; k-- {
		nodeErrors := []float64{}
		for l := 0; l < len(sigmoidPrimes[k-1]); l++ {
			var nodeError float64
			for m := 0; m < len(errorLastLayer); m++ {
				nodeError += network.Weights[k-1][m][l] * errorLastLayer[m] * sigmoidPrimes[k-1][l]
			}
			nodeErrors = append(nodeErrors, nodeError)
		}
		errorLastLayer = nodeErrors
		newBiases[k-1] = errorLastLayer
		newWeights[k-1] = multiplyVectors(activations[k-1], errorLastLayer)
	}
	return newWeights, newBiases
}

// Helper Functions

func multiplyVectors(a, b []float64) [][]float64 {
	lenA, lenB := len(a), len(b)
	newF := [][]float64{}
	for i := 0; i < lenB; i++ {
		newNodeM := []float64{}
		for j := 0; j < lenA; j++ {
			newNodeM = append(newNodeM, a[j]*b[i])
		}
		newF = append(newF, newNodeM)
	}
	return newF
}

func hadamardVector(a, b []float64) []float64 {
	n := len(a)
	newS := make([]float64, n)
	for i, n := 0, len(a); i < n; i++ {
		newS[i] = a[i] * b[i]
	}
	return newS
}

func activate(v float64) float64 {
	return sigmoid(v)
}

func netInput(x []float64, w []float64, b float64) float64 {
	var total float64
	for i, n := 0, len(x); i < n; i++ {
		total += x[i] * w[i]
	}
	return total + b
}

func transpose(a [][]float64) [][]float64 {
	n := len(a[0])
	nn := len(a)
	transposedS := make([][]float64, n)
	for i := 0; i < n; i++ {
		tempS := make([]float64, nn)
		for j := 0; j < nn; j++ {
			tempS[j] = a[j][i]
		}
		transposedS[i] = tempS
	}
	return transposedS
}

func initializeWeightsAndBiases(nLayers int, sizes []int, zeroValued bool) ([][][]float64, [][]float64) {
	rand.Seed(time.Now().Unix())
	sqrtInputs := math.Sqrt(float64(sizes[0]))
	weights := make([][][]float64, nLayers-1)
	biases := make([][]float64, nLayers-1)
	for j := 0; j < nLayers-1; j++ {
		biases[j] = make([]float64, sizes[j+1])
		if zeroValued != true {
			for j1 := 0; j1 < len(biases[j]); j1++ {
				biases[j][j1] = rand.NormFloat64() / sqrtInputs
			}
		}
		weights[j] = make([][]float64, sizes[j+1])
		for k := 0; k < sizes[j+1]; k++ {
			weights[j][k] = make([]float64, sizes[j])
			if zeroValued != true {
				for k1 := 0; k1 < len(weights[j][k]); k1++ {
					weights[j][k][k1] = rand.NormFloat64() / sqrtInputs
				}
			}
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

func relu(v float64) float64 {
	if v <= 0 {
		return 0
	}
	return v
}

func sigmoidDerivative(v float64) float64 {
	return sigmoid(v) * (1 - sigmoid(v))
}

func costDerivative(activations []float64, y float64) []float64 {
	errors := make([]float64, len(activations))
	for i := 0; i < len(activations); i++ {
		errors[i] = activations[i] - y
	}
	return errors
}
