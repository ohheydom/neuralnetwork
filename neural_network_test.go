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

func TestActivate(t *testing.T) {
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
			t.Errorf("Expected %v, received %v", v.output, actual)
		}
	}
}

func TestRandomShuffle(t *testing.T) {
	x := [][]float64{[]float64{1}, []float64{2}, []float64{3}, []float64{4}, []float64{5}, []float64{6}}
	y := []float64{1, 2, 3, 4, 5, 6}
	xNew, _ := randomShuffle(x, y, 1)
	if xNew[0][0] == x[0][0] {
		t.Errorf("Expected the slice to be shuffled, but instead it was not.")
	}
}

func TestTranspose(t *testing.T) {
	x := [][]float64{[]float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}}
	actual := transpose(x)
	if len(actual) != 4 && actual[3][1] != 2 {
		t.Errorf("Should have received a transposed array of length 4, instead received %v", actual)
	}
}

func TestSGD(t *testing.T) {
	n := NewNetwork([]int{2, 2, 1})
	xTrain := [][]float64{[]float64{1, 1}, []float64{0, 1}, []float64{1, 0}, []float64{0, 0}, []float64{1, 1}, []float64{0, 1}, []float64{1, 0}, []float64{0, 0}, []float64{1, 1}, []float64{0, 1}, []float64{1, 0}, []float64{0, 0}}

	yTrain := []float64{0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0}
	n.SGD(xTrain, yTrain, 2, 50000, 0.14)

	var xors = []struct {
		input  []float64
		output int
	}{
		{[]float64{1, 1}, 0},
		{[]float64{0, 0}, 0},
		{[]float64{1, 0}, 1},
		{[]float64{0, 1}, 1},
	}

	for _, v := range xors {
		actual := int(n.FeedForward(v.input)[0] + 0.5)
		if actual != v.output {
			t.Errorf("Expected %v, received %v", v.output, actual)
		}
	}
}
