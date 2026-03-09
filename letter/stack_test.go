package letter

import (
	"testing"

	"github.com/fab2s/goDl/autograd"
	_ "github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

func TestStackToDevice(t *testing.T) {
	if !tensor.CUDAAvailable() {
		t.Skip("CUDA not available")
	}

	// Test 1: direct ToDevice
	t1, _ := tensor.Ones([]int64{4, 1, 32, 32})
	t.Logf("t1 device: %v", t1.Device())
	m := t1.ToDevice(tensor.CUDA)
	if err := m.Err(); err != nil {
		t.Fatalf("Direct ToDevice: %v", err)
	}
	t.Log("Direct ToDevice OK")

	// Test 2: Reshape + ToDevice
	r := t1.Reshape([]int64{1, 4, 1, 32, 32})
	t.Logf("Reshaped device: %v shape: %v", r.Device(), r.Shape())
	m2 := r.ToDevice(tensor.CUDA)
	if err := m2.Err(); err != nil {
		t.Fatalf("Reshape+ToDevice: %v", err)
	}
	t.Log("Reshape+ToDevice OK")

	// Test 3: Cat + ToDevice
	t2, _ := tensor.Ones([]int64{4, 1, 32, 32})
	catted := t1.Cat(t2, 0)
	t.Logf("Catted device: %v shape: %v", catted.Device(), catted.Shape())
	m3 := catted.ToDevice(tensor.CUDA)
	if err := m3.Err(); err != nil {
		t.Fatalf("Cat+ToDevice: %v", err)
	}
	t.Log("Cat+ToDevice OK")

	// Test 4: Stack + ToDevice (the failing case)
	stacked := tensor.Stack([]*tensor.Tensor{t1, t2}, 0)
	t.Logf("Stacked device: %v shape: %v", stacked.Device(), stacked.Shape())
	m4 := stacked.ToDevice(tensor.CUDA)
	if err := m4.Err(); err != nil {
		t.Fatalf("Stack+ToDevice: %v", err)
	}
	t.Log("Stack+ToDevice OK")
}

func TestCUDAAfterSetDevice(t *testing.T) {
	if !tensor.CUDAAvailable() {
		t.Skip("CUDA not available")
	}

	// Regression test for goDl CUDA use-after-free (fixed in d0cbf66).
	ds, err := NewSyntheticDataset(64)
	if err != nil {
		t.Fatal(err)
	}

	cfg := DefaultLetterConfig()
	cfg.NGlimpses = 2
	cfg.PatchSize = 8
	cfg.LatentDim = 32
	cfg.BatchSize = 16
	cfg.Epochs = 1
	cfg.SaveDir = ""

	m := NewLetterModel(cfg.NClasses, cfg.NGlimpses, cfg.PatchSize, cfg.NScales, cfg.LatentDim)
	m.Graph.SetDevice(tensor.CUDA)
	t.Log("SetDevice(CUDA) OK")

	// nn.NewAdam(m.Parameters(), cfg.LR)
	m.SetTraining(true)
	t.Log("Training mode OK (no optimizer)")

	loader := NewLetterLoader(ds, cfg.BatchSize, true)
	loader.device = m.Graph.Device()
	t.Log("Loader created")

	if !loader.Next() {
		t.Fatal("No first batch")
	}
	t.Log("First batch loaded to CUDA OK")

	batch := loader.Batch()
	t.Logf("Batch image: shape=%v device=%v", batch.Image.Shape(), batch.Image.Device())

	// Try forward pass.
	imgVar := autograd.NewVariable(batch.Image, false)
	caseVar := autograd.NewVariable(batch.CaseLabel, false)
	result := m.Forward(imgVar, caseVar)
	if err := result.LetterLogits.Err(); err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	t.Log("Forward pass OK")
}
