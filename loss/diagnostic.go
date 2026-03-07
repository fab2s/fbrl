// Non-differentiable diagnostics for attention behavior.
package loss

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// FixationHitRate measures what fraction of fixations land on actual letter
// pixels. Returns (hitRate, meanIntensity). Both in [0, 1].
//
// image: [B, 1, H, W] clean image (sharp, unblurred).
// locations: fixation trajectory — [0] is skipped.
// threshold: intensity above which a sample counts as a hit. 0.3 is typical.
func FixationHitRate(image *autograd.Variable, locations []*autograd.Variable, threshold float64) (hitRate, meanIntensity float64) {
	if len(locations) <= 1 {
		return 0, 0
	}

	imgData := image.Data()
	shape := imgData.Shape()
	B := shape[0]

	hits := 0
	totalIntensity := float64(0)
	n := 0

	for _, loc := range locations[1:] {
		grid := loc.Data().Reshape([]int64{B, 1, 1, 2})
		sampled := imgData.GridSample(grid, 0, 0, true) // [B, 1, 1, 1]

		// Read sampled values.
		vals, _ := sampled.Float32Data()
		for _, v := range vals {
			totalIntensity += float64(v)
			if float64(v) > threshold {
				hits++
			}
			n++
		}
	}

	if n == 0 {
		return 0, 0
	}
	return float64(hits) / float64(n), totalIntensity / float64(n)
}

// RecodeLoss computes MSE between a reconstructed image and a target.
// This is a convenience wrapper — functionally identical to nn.MSELoss
// but named for clarity in training code.
func RecodeLoss(recon, target *autograd.Variable) *autograd.Variable {
	diff := recon.Sub(target)
	sq := diff.Mul(diff)

	numel := int64(1)
	for _, d := range sq.Data().Shape() {
		numel *= d
	}
	return sq.Sum().MulScalar(1.0 / float64(numel))
}

// zeroVar returns a non-gradient zero scalar variable.
func zeroVar() *autograd.Variable {
	z, _ := tensor.Zeros([]int64{1})
	return autograd.NewVariable(z, false)
}
