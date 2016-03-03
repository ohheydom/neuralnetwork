package neuralnetwork

import (
	"math"
)

func Activate(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func NetInput(x []float64, w []float64) float64 {
	n := len(x)
	var total float64
	for i := 0; i < n; i++ {
		total += x[i] * w[i+1]
	}
	return total + w[0]
}
