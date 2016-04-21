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
	Weights           [][][]float64
	WeightsSum        float64
	WeightsSumSquared float64
	Biases            [][]float64
	NLayers           int
	Sizes             []int
	RegStrength       float64
	weightCosts       [][][]float64
	biasCosts         [][]float64
	loss              float64
}

// NewNetwork builds a Neural Network from the given sizes of each layer. Also takes in a regularization strength to be used during the loss and gradient calculations.
func NewNetwork(sizes []int, regStrength float64) *Network {
	network := new(Network)
	nSize := len(sizes)
	network.Sizes = sizes
	network.NLayers = nSize
	network.RegStrength = regStrength
	network.Weights, network.Biases, network.WeightsSum, network.WeightsSumSquared = initializeWeightsAndBiases(nSize, sizes, false)
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
func (network *Network) SGD(x [][]float64, y []float64, miniBatchSize int, nIter int, eta, lrDecay float64) {
	n := len(x)
	if miniBatchSize > n {
		log.Fatal("miniBatchSize must be smaller than the number of samples in your training data.")
	}
	batches := createBatchMarkers(n, miniBatchSize)
	for i := 0; i < nIter; i++ {
		newX, newY := randomShuffle(x, y, time.Now().Unix())
		for j := 0; j < len(batches)-1; j++ {
			s, e := batches[j], batches[j+1]
			network.updateWeights(newX[s:e], newY[s:e], eta)
		}
		eta *= lrDecay
	}
}

func (network *Network) updateWeights(x [][]float64, y []float64, eta float64) {
	network.weightCosts, network.biasCosts, _, _ = initializeWeightsAndBiases(network.NLayers, network.Sizes, true)
	network.loss = 0
	regTermSquared := 0.5 * network.RegStrength * network.WeightsSumSquared
	for i := 0; i < len(x); i++ {
		network.backPropagation(x[i], y[i])
	}
	network.loss /= float64(len(y))
	network.loss += regTermSquared
	regTerm := network.RegStrength * network.WeightsSum
	newWeightSum, newWeightSumSquared := 0.0, 0.0
	for i := 0; i < len(x); i++ {
		for j := 0; j < network.NLayers-1; j++ {
			for k := 0; k < network.Sizes[j+1]; k++ {
				network.Biases[j][k] -= (eta/float64(len(x)))*network.biasCosts[j][k] + regTerm
				for wc := 0; wc < len(network.weightCosts[j][k]); wc++ {
					network.Weights[j][k][wc] -= ((eta / float64(len(x))) * network.weightCosts[j][k][wc]) + regTerm
					newWeightSum += network.Weights[j][k][wc]
					newWeightSumSquared += network.Weights[j][k][wc] * network.Weights[j][k][wc]
				}
			}
		}
	}
	network.WeightsSum = newWeightSum
	network.WeightsSumSquared = newWeightSumSquared
}

// backPropagation feeds forward through the network then backpropagates and calculates all the changes in the cost function with respect to each weight and bias
func (network *Network) backPropagation(x []float64, y float64) {
	activations := [][]float64{x}
	sigmoidPrimes := [][]float64{}
	activation := x
	for i := 0; i < network.NLayers-1; i++ {
		newAcLayer := []float64{}
		newSigmoidPrimeLayer := []float64{}
		for j := 0; j < network.Sizes[i+1]; j++ {
			var overallZ float64
			for k := 0; k < len(activation); k++ {
				overallZ += activation[k] * network.Weights[i][j][k]
			}
			preAc := overallZ + network.Biases[i][j]
			newSigmoidPrimeLayer = append(newSigmoidPrimeLayer, activatePrime(preAc))
			newAcLayer = append(newAcLayer, activate(preAc))
		}
		activation = newAcLayer
		activations = append(activations, activation)
		sigmoidPrimes = append(sigmoidPrimes, newSigmoidPrimeLayer)
	}
	cost, loss := costDerivative(activations[len(activations)-1], y)
	network.loss += loss
	nLast := len(sigmoidPrimes) - 1
	errorLastLayer := network.hadamardVector(nLast, cost, sigmoidPrimes[nLast])
	network.multiplyVectors(nLast, activations[len(activations)-2], errorLastLayer)
	for i := 2; i < network.NLayers; i++ {
		n := len(network.biasCosts) - i
		nodeErrors := []float64{}
		for j := 0; j < len(sigmoidPrimes[n]); j++ {
			var nodeError float64
			for k := 0; k < len(errorLastLayer); k++ {
				nodeError += network.Weights[n+1][k][j] * errorLastLayer[k] * sigmoidPrimes[n][j]
				network.biasCosts[k][j] += nodeError + 0.5*network.RegStrength*network.WeightsSumSquared
			}
			nodeErrors = append(nodeErrors, nodeError)
		}
		errorLastLayer = nodeErrors
		network.multiplyVectors(n, activations[len(activations)-i-1], errorLastLayer)
	}
}

// Helper Functions

func activate(v float64) float64 {
	return sigmoid(v)
}

func activatePrime(v float64) float64 {
	return sigmoidPrime(v)
}

// multiplyVectors calculates a straight multiplication of vectors and adds values to the weight costs
func (network *Network) multiplyVectors(idx int, a, b []float64) {
	lenA, lenB := len(a), len(b)
	newF := make([][]float64, lenB)
	for i := 0; i < lenB; i++ {
		newNodeM := make([]float64, lenA)
		for j := 0; j < lenA; j++ {
			newNodeM[j] = a[j] * b[i]
			network.weightCosts[idx][i][j] += (a[j] * b[i]) + 0.5*network.RegStrength*network.WeightsSumSquared
		}
		newF[i] = newNodeM
	}
}

// hadamardVector calculates the hadamard product and adds value to the bias costs
func (network *Network) hadamardVector(idx int, a, b []float64) []float64 {
	n := len(a)
	newS := make([]float64, n)
	for i, n := 0, len(a); i < n; i++ {
		newS[i] = a[i] * b[i]
		network.biasCosts[idx][i] += a[i] * b[i]
	}
	return newS
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

func initializeWeightsAndBiases(nLayers int, sizes []int, zeroValued bool) ([][][]float64, [][]float64, float64, float64) {
	rand.Seed(2) //time.Now().Unix())
	sqrtInputs := math.Sqrt(float64(sizes[0]))
	weights := make([][][]float64, nLayers-1)
	biases := make([][]float64, nLayers-1)
	weightsSum, weightsSumSquared := 0.0, 0.0
	for i := 0; i < nLayers-1; i++ {
		biases[i] = make([]float64, sizes[i+1])
		weights[i] = make([][]float64, sizes[i+1])
		for j := 0; j < sizes[i+1]; j++ {
			weights[i][j] = make([]float64, sizes[i])
			if zeroValued != true {
				for k := 0; k < len(weights[i][j]); k++ {
					w := rand.NormFloat64() / sqrtInputs
					weightsSum += w
					weightsSumSquared += w * w
					weights[i][j][k] = w
				}
			}
		}
	}
	return weights, biases, weightsSum, weightsSumSquared
}

func createBatchMarkers(n, miniBatchSize int) []int {
	bSize := n / miniBatchSize
	if n%miniBatchSize != 0 {
		bSize++
	}
	batches := make([]int, bSize)
	batches = append(batches, n)
	for idx, i := 0, 0; i < n; idx, i = idx+1, i+miniBatchSize {
		batches[idx] = i
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

func sigmoidPrime(v float64) float64 {
	return sigmoid(v) * (1 - sigmoid(v))
}

func reLU(v float64) float64 {
	if v <= 0 {
		return 0
	}
	return v
}

func reLUPrime(v float64) float64 {
	if v > 0.0 {
		return 1
	}
	return 0
}

func costDerivative(activations []float64, y float64) ([]float64, float64) {
	errors := make([]float64, len(activations))
	var loss float64
	for i := 0; i < len(activations); i++ {
		errors[i] = activations[i] - y
		loss += errors[i] * errors[i]
	}
	return errors, loss
}

// Subtract mean and divide by standard deviation
func normalize(X [][]float64) [][]float64 {
	n := len(X)
	nFeatures := len(X[0])
	stdDev := 0.0
	for i := 0; i < n; i++ {
		mean, squaredTotal := 0.0, 0.0
		for j := 0; j < nFeatures; j++ {
			mean += X[i][j]
		}
		nf := float64(nFeatures)
		mean /= nf
		for k := 0; k < nFeatures; k++ {
			v := X[i][k] - mean
			squaredTotal += v * v
			X[i][k] = v
		}
		stdDev = math.Sqrt(squaredTotal / nf)
		for k := 0; k < nFeatures; k++ {
			X[i][k] /= stdDev
		}
	}
	return X
}
