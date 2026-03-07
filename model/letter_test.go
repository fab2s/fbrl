package model

import (
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

func TestLetterModelForward(t *testing.T) {
	B := int64(2)
	nClasses := 26
	nGlimpses := 3
	patchSize := 8
	nScales := 1
	latentDim := 32

	m := NewLetterModel(nClasses, nGlimpses, patchSize, nScales, latentDim)

	imgT, err := tensor.RandN([]int64{B, 1, 128, 128})
	if err != nil {
		t.Fatal(err)
	}
	caseT, err := tensor.Zeros([]int64{B, 1})
	if err != nil {
		t.Fatal(err)
	}

	img := autograd.NewVariable(imgT, false)
	caseLabel := autograd.NewVariable(caseT, false)

	result := m.Forward(img, caseLabel)

	reconShape := result.Recon.Data().Shape()
	if reconShape[0] != B || reconShape[1] != 1 || reconShape[2] != 128 || reconShape[3] != 128 {
		t.Errorf("recon shape: got %v, want [%d, 1, 128, 128]", reconShape, B)
	}

	letterShape := result.LetterLogits.Data().Shape()
	if letterShape[0] != B || letterShape[1] != int64(nClasses) {
		t.Errorf("letter logits shape: got %v, want [%d, %d]", letterShape, B, nClasses)
	}

	caseShape := result.CaseLogits.Data().Shape()
	if caseShape[0] != B || caseShape[1] != 2 {
		t.Errorf("case logits shape: got %v, want [%d, 2]", caseShape, B)
	}

	// nGlimpses + 1 locations (initial zeros + one per step)
	if len(result.Locations) != nGlimpses+1 {
		t.Errorf("locations count: got %d, want %d", len(result.Locations), nGlimpses+1)
	}

	t.Logf("param count: %d", len(m.Parameters()))
	t.Logf("graph DOT:\n%s", m.Graph.DOT())
}

func TestLetterModelBackward(t *testing.T) {
	B := int64(2)
	m := NewLetterModel(26, 2, 8, 1, 32)

	imgT, _ := tensor.RandN([]int64{B, 1, 128, 128})
	caseT, _ := tensor.Zeros([]int64{B, 1})
	img := autograd.NewVariable(imgT, false)
	caseLabel := autograd.NewVariable(caseT, false)

	result := m.Forward(img, caseLabel)

	loss := result.LetterLogits.Sum()
	_ = loss.Backward()

	gradCount := 0
	for _, p := range m.Parameters() {
		if p.Grad() != nil {
			gradCount++
		}
	}
	if gradCount == 0 {
		t.Error("no parameters received gradients")
	}
	t.Logf("%d/%d params got gradients", gradCount, len(m.Parameters()))
}
