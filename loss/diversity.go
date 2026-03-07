// Fixation diversity loss — pairwise repulsion between fixation points.
package loss

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// FixationDiversityLoss computes pairwise Gaussian RBF repulsion between
// fixation points. Fixations closer than ~sigma (in [-1,1] normalized coords)
// repel each other, encouraging spatial spread.
//
// locations: fixation trajectory from g.Traces(tag) — [0] is the initial
// position (skipped), [1:] are learned fixations.
// sigma: repulsion radius in [-1,1] coords. 0.1 means ~10% of image width.
// vy: vertical scale factor. vy > 1.0 penalizes vertical clustering more.
func FixationDiversityLoss(locations []*autograd.Variable, sigma, vy float64) *autograd.Variable {
	locs := locations[1:]
	T := len(locs)
	if T < 2 {
		z, _ := tensor.Zeros([]int64{1})
		return autograd.NewVariable(z, false)
	}

	invTwoSigmaSq := -1.0 / (2 * sigma * sigma)
	vySquared := vy * vy

	// Explicit pairwise loop. T is small (8-12 for letter models).
	var total *autograd.Variable
	B := locs[0].Data().Shape()[0]
	for i := 0; i < T; i++ {
		for j := i + 1; j < T; j++ {
			dx := locs[i].Select(1, 0).Sub(locs[j].Select(1, 0)) // [B]
			dy := locs[i].Select(1, 1).Sub(locs[j].Select(1, 1)) // [B]

			distSq := dx.Mul(dx)
			if vySquared != 1.0 {
				distSq = distSq.Add(dy.Mul(dy).MulScalar(vySquared))
			} else {
				distSq = distSq.Add(dy.Mul(dy))
			}

			repulsion := distSq.MulScalar(invTwoSigmaSq).Exp() // [B]
			batchMean := repulsion.Sum().MulScalar(1.0 / float64(B))

			if total == nil {
				total = batchMean
			} else {
				total = total.Add(batchMean)
			}
		}
	}

	// Scale to match mean over all T*T entries (including zeroed diagonal),
	// consistent with Python: (repulsion * mask).mean() over [B, T, T].
	return total.MulScalar(2.0 / float64(T*T))
}
