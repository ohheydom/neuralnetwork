package neuralnetwork

import (
	"testing"
)

func TestNetInput(t *testing.T) {
	v := []float64{5, 10, 15, 20}
	w := []float64{20, 2, 4, 6, 8}
	netInput := NetInput(v, w)
	if netInput != 320 {
		t.Errorf("Should have received 320, instead received %v", netInput)
	}
}

func TestActivate(t *testing.T) {
	v := 0.0
	activatedValue := Activate(v)
	if activatedValue != 0.5 {
		t.Errorf("Should have received 0.5, instead received %v", activatedValue)
	}
}
