package neuralnetwork

import (
	"testing"
)

func TestnetInput(t *testing.T) {
	v := []float64{5, 10, 15, 20}
	w := []float64{2, 4, 6, 8}
	b := 20.0
	netInput := netInput(v, w, b)
	if netInput != 320 {
		t.Errorf("Should have received 320, instead received %v", netInput)
	}
}

func Testactivate(t *testing.T) {
	v := 0.0
	activatedValue := activate(v)
	if activatedValue != 0.5 {
		t.Errorf("Should have received 0.5, instead received %v", activatedValue)
	}
}

func TestFeedForwardXOR(t *testing.T) {
	var xors = []struct {
		input  []float64
		output int
	}{
		{[]float64{1, 1}, 0},
		{[]float64{0, 0}, 0},
		{[]float64{1, 0}, 1},
		{[]float64{0, 1}, 1},
	}

	n := NewNetwork([]int{2, 2, 1})
	n.Biases = [][]float64{[]float64{-10, 30}, []float64{-30}}
	n.Weights = [][][]float64{[][]float64{[]float64{20, 20}, []float64{-20, -20}}, [][]float64{[]float64{20, 20}}}
	for _, v := range xors {
		actual := int(n.FeedForward(v.input)[0] + 0.5) // Rounds down
		if actual != v.output {
			t.Errorf("Expect %v, received %v", v.output, actual)
		}
	}
}
