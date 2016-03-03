# Neural Network

A basic Neural Network written in golang.

## Usage

Right now, the package does not calculate the weights or biases, so it doesn't do much. If you know the weights and biases, you can use it as follows:

### Import

```golang
import (
  "github.com/ohheydom/linearregression"
)
```

### Create Neural Network

```golang
// xor Neural Network
n := neuralnetwork.NewNetwork([]int{2, 2, 1})
n.Biases = [][]float64{[]float64{-10, 30}, []float64{-30}}
n.Weights = [][][]float64{[][]float64{[]float64{20, 20}, []float64{-20, -20}}, [][]float64{[]float64{20, 20}}}
```

### Create samples and FeedForward

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

## Todo

Implement back propagation to calculate weights and biases.
