package letter

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

func makeLocations(B int64, coords [][]float32) []*autograd.Variable {
	locs := make([]*autograd.Variable, len(coords))
	for i, c := range coords {
		data := make([]float32, B*2)
		for b := range B {
			data[b*2] = c[0]
			data[b*2+1] = c[1]
		}
		t, _ := tensor.FromFloat32(data, []int64{B, 2})
		locs[i] = autograd.NewVariable(t, true)
	}
	return locs
}

func TestAttentionGuideLoss(t *testing.T) {
	B := int64(4)

	// All-ones image: guide values are positive everywhere.
	imgT, _ := tensor.Ones([]int64{B, 1, 32, 32})
	image := autograd.NewVariable(imgT, false)

	locs := makeLocations(B, [][]float32{
		{0, 0},     // initial (skipped)
		{0, 0},     // center
		{0.5, 0.5}, // off-center
	})

	loss := AttentionGuideLoss(image, locs, 0.16)
	if err := loss.Err(); err != nil {
		t.Fatal(err)
	}

	lossVal := loss.Item()
	// Loss should be negative (guide values are positive, then negated).
	if lossVal >= 0 {
		t.Errorf("expected negative loss, got %f", lossVal)
	}

	t.Logf("guide loss: %f", lossVal)
}

func TestAttentionGuideLossGrad(t *testing.T) {
	B := int64(2)
	imgT, _ := tensor.RandN([]int64{B, 1, 32, 32})
	image := autograd.NewVariable(imgT, false)

	locs := makeLocations(B, [][]float32{
		{0, 0},
		{0.3, -0.2},
		{-0.5, 0.4},
	})

	loss := AttentionGuideLoss(image, locs, 0.16)
	if err := loss.Err(); err != nil {
		t.Fatal(err)
	}

	loss.Backward()

	for i := 1; i < len(locs); i++ {
		g := locs[i].Grad()
		if g == nil {
			t.Errorf("location[%d] has no gradient", i)
		}
	}
}

func TestAttentionGuideLossTooFewLocations(t *testing.T) {
	B := int64(2)
	imgT, _ := tensor.Zeros([]int64{B, 1, 16, 16})
	image := autograd.NewVariable(imgT, false)

	locs := makeLocations(B, [][]float32{{0, 0}})
	loss := AttentionGuideLoss(image, locs, 0.16)
	val := loss.Item()
	if val != 0 {
		t.Errorf("expected 0 loss for single location, got %f", val)
	}
}

func TestFixationDiversityLoss(t *testing.T) {
	B := int64(4)

	clustered := makeLocations(B, [][]float32{
		{0, 0},      // initial (skipped)
		{0.1, 0.1},  // close together
		{0.12, 0.09},
		{0.08, 0.11},
	})

	spread := makeLocations(B, [][]float32{
		{0, 0},       // initial (skipped)
		{-0.8, -0.8}, // far apart
		{0.8, -0.8},
		{0.0, 0.8},
	})

	lossClustered := FixationDiversityLoss(clustered, 0.1, 1.0)
	lossSpread := FixationDiversityLoss(spread, 0.1, 1.0)

	if err := lossClustered.Err(); err != nil {
		t.Fatal(err)
	}
	if err := lossSpread.Err(); err != nil {
		t.Fatal(err)
	}

	vc := lossClustered.Item()
	vs := lossSpread.Item()

	if vc <= vs {
		t.Errorf("clustered loss (%f) should exceed spread loss (%f)", vc, vs)
	}

	t.Logf("clustered: %f, spread: %f", vc, vs)
}

func TestFixationDiversityLossVy(t *testing.T) {
	B := int64(2)

	// Fixations separated horizontally — vy should not matter.
	hLocs := makeLocations(B, [][]float32{
		{0, 0},
		{-0.1, 0},
		{0.1, 0},
	})

	// Fixations separated vertically — higher vy stretches y-axis,
	// making the same physical separation appear larger (less repulsion).
	vLocs := makeLocations(B, [][]float32{
		{0, 0},
		{0, -0.1},
		{0, 0.1},
	})

	hLossVy1 := FixationDiversityLoss(hLocs, 0.1, 1.0).Item()
	hLossVy3 := FixationDiversityLoss(hLocs, 0.1, 3.0).Item()

	// Horizontal separation: vy has no effect on x-only distance.
	if math.Abs(hLossVy1-hLossVy3) > 1e-6 {
		t.Errorf("horizontal: vy should not matter, got vy=1: %f, vy=3: %f", hLossVy1, hLossVy3)
	}

	vLossVy1 := FixationDiversityLoss(vLocs, 0.1, 1.0).Item()
	vLossVy3 := FixationDiversityLoss(vLocs, 0.1, 3.0).Item()

	// Vertical separation: higher vy → larger effective distance → less repulsion.
	if vLossVy3 >= vLossVy1 {
		t.Errorf("vertical: vy=3 should reduce repulsion, got vy=1: %f, vy=3: %f", vLossVy1, vLossVy3)
	}

	t.Logf("horizontal vy=1: %f, vy=3: %f | vertical vy=1: %f, vy=3: %f",
		hLossVy1, hLossVy3, vLossVy1, vLossVy3)
}

func TestFixationDiversityLossGrad(t *testing.T) {
	B := int64(2)
	locs := makeLocations(B, [][]float32{
		{0, 0},
		{0.3, 0.3},
		{-0.3, -0.3},
	})

	loss := FixationDiversityLoss(locs, 0.1, 1.0)
	loss.Backward()

	for i := 1; i < len(locs); i++ {
		g := locs[i].Grad()
		if g == nil {
			t.Errorf("location[%d] has no gradient", i)
		}
	}
}

func TestFixationDiversityLossTooFew(t *testing.T) {
	B := int64(2)
	locs := makeLocations(B, [][]float32{{0, 0}, {0.5, 0.5}})
	loss := FixationDiversityLoss(locs, 0.1, 1.0)
	val := loss.Item()
	if val != 0 {
		t.Errorf("expected 0 loss for single learned location, got %f", val)
	}
}

func TestFixationHitRate(t *testing.T) {
	B := int64(2)

	imgT, _ := tensor.Ones([]int64{B, 1, 16, 16})
	image := autograd.NewVariable(imgT, false)

	locs := makeLocations(B, [][]float32{
		{0, 0},
		{0, 0},
		{0.5, 0.5},
	})

	hr, mi := FixationHitRate(image, locs, 0.3)
	if hr != 1.0 {
		t.Errorf("expected 100%% hit rate on all-ones image, got %.2f", hr)
	}
	if math.Abs(mi-1.0) > 0.01 {
		t.Errorf("expected mean intensity ~1.0, got %.4f", mi)
	}

	zeroT, _ := tensor.Zeros([]int64{B, 1, 16, 16})
	zeroImg := autograd.NewVariable(zeroT, false)
	hr, mi = FixationHitRate(zeroImg, locs, 0.3)
	if hr != 0 {
		t.Errorf("expected 0%% hit rate on all-zeros image, got %.2f", hr)
	}
	if mi != 0 {
		t.Errorf("expected mean intensity 0, got %.4f", mi)
	}
}

func TestRecodeLoss(t *testing.T) {
	B := int64(2)
	aT, _ := tensor.RandN([]int64{B, 1, 8, 8})
	bT, _ := tensor.RandN([]int64{B, 1, 8, 8})

	a := autograd.NewVariable(aT, true)
	b := autograd.NewVariable(bT, false)

	loss := RecodeLoss(a, b)
	if err := loss.Err(); err != nil {
		t.Fatal(err)
	}

	val := loss.Item()
	if val < 0 {
		t.Errorf("MSE should be non-negative, got %f", val)
	}

	loss.Backward()
	if a.Grad() == nil {
		t.Error("expected gradient on a")
	}

	// Same tensor should give zero loss.
	same := RecodeLoss(autograd.NewVariable(aT, false), autograd.NewVariable(aT, false))
	sameVal := same.Item()
	if math.Abs(sameVal) > 1e-6 {
		t.Errorf("expected ~0 loss for identical tensors, got %f", sameVal)
	}
}
