# Neural Network

A basic Neural Network written in golang.

## Usage

There are currently two ways of using the package. You can either input training data with labels to automatically calculate the weights and biases, or you can manually input the weights and biases.


### Automatically Calculating Weights and Biases

#### Import

```golang
import (
  "github.com/ohheydom/neuralnetwork"
)
```

#### Create Neural Network

```golang
// xor Neural Network
n := neuralnetwork.NewNetwork([]int{2, 2, 1}, 0)
```

#### Create samples and Run Stochastic Gradient Descent

```golang
// Create the training inputs and train
xTrain := [][]float64{[]float64{1, 1}, []float64{0, 1}, []float64{1, 0}, []float64{0, 0}}
yTrain := []float64{0.0, 1.0, 1.0, 0.0}
n.SGD(xTrain, yTrain, 1, 100000, 0.10, 1.0)

// Create the test inputs
w := []float64{1, 1}
x := []float64{0, 0}
y := []float64{1, 0}
z := []float64{0, 1}

// Print the results
fmt.Printf("1 and 1 returns %.0f\n", n.FeedForward(w)[0])
fmt.Printf("0 and 0 returns %.0f\n", n.FeedForward(x)[0])
fmt.Printf("1 and 0 returns %.0f\n", n.FeedForward(y)[0])
fmt.Printf("0 and 1 returns %.0f\n", n.FeedForward(z)[0])
```

### Manually Inputting Weights and Biases

#### Import

```golang
import (
  "github.com/ohheydom/neuralnetwork"
)
```

#### Create Neural Network

```golang
// xor Neural Network
n := neuralnetwork.NewNetwork([]int{2, 2, 1}, 0)
n.Biases = [][]float64{[]float64{-10, 30}, []float64{-30}}
n.Weights = [][][]float64{[][]float64{[]float64{20, 20}, []float64{-20, -20}}, [][]float64{[]float64{20, 20}}}
```

#### Create samples and FeedForward

```golang
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
```

Please see other examples in the examples folder.
