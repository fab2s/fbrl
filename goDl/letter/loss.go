// Loss functions for the letter model — attention guidance, fixation
// diversity, reconstruction, and non-differentiable diagnostics.
package letter

import (
	"math"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// AttentionGuideLoss guides fixation locations toward image content.
//
// Blurs the image to create a soft "scent field" around strokes, then samples
// this field at each fixation point. Minimizing the loss pushes fixations
// onto high-content regions.
//
// image: [B, 1, H, W] clean image (unnoised).
// locations: fixation trajectory from g.Traces(tag) — [0] is the initial
// position (skipped), [1:] are learned fixations.
// blurSigmaRatio: Gaussian sigma as fraction of min(H,W). 0.16 is the default.
func AttentionGuideLoss(image *autograd.Variable, locations []*autograd.Variable, blurSigmaRatio float64) *autograd.Variable {
	if len(locations) <= 1 {
		z, _ := tensor.Zeros([]int64{1})
		return autograd.NewVariable(z, false)
	}

	imgData := image.Data()
	shape := imgData.Shape() // [B, C, H, W]
	H, W := shape[2], shape[3]

	blurSigma := blurSigmaRatio * float64(min(H, W))

	// Build 1D Gaussian kernel.
	k := int(4*blurSigma) | 1
	halfK := k / 2
	kernelData := make([]float32, k)
	kSum := float64(0)
	for i := range k {
		x := float64(i - halfK)
		v := math.Exp(-x * x / (2 * blurSigma * blurSigma))
		kernelData[i] = float32(v)
		kSum += v
	}
	for i := range k {
		kernelData[i] /= float32(kSum)
	}

	// Separable blur: convolve rows then columns.
	gaussT, _ := tensor.FromFloat32(kernelData, []int64{int64(k)}, tensor.WithDevice(imgData.Device()))
	kernelH := gaussT.Reshape([]int64{1, 1, int64(k), 1})
	kernelW := gaussT.Reshape([]int64{1, 1, 1, int64(k)})

	guide := imgData.Conv2d(kernelH, nil, []int64{1, 1}, []int64{int64(halfK), 0}, []int64{1, 1}, 1)
	guide = guide.Conv2d(kernelW, nil, []int64{1, 1}, []int64{0, int64(halfK)}, []int64{1, 1}, 1)
	guideVar := autograd.NewVariable(guide, false)

	// Sample guide at each learned fixation and accumulate.
	nLocs := len(locations) - 1
	var total *autograd.Variable
	for _, loc := range locations[1:] {
		grid := loc.Unsqueeze(1).Unsqueeze(2)
		sampled := guideVar.GridSample(grid, 0, 0, true) // bilinear, zeros, align_corners
		batchMean := sampled.Mean()
		if total == nil {
			total = batchMean
		} else {
			total = total.Add(batchMean)
		}
	}

	// Negate: maximize guide values → minimize negative.
	return total.MulScalar(-1.0 / float64(nLocs))
}

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
	for i := 0; i < T; i++ {
		for j := i + 1; j < T; j++ {
			dx := locs[i].Select(1, 0).Sub(locs[j].Select(1, 0)) // [B]
			dy := locs[i].Select(1, 1).Sub(locs[j].Select(1, 1)) // [B]

			distSq := dx.Pow(2)
			if vySquared != 1.0 {
				distSq = distSq.Add(dy.Pow(2).MulScalar(vySquared))
			} else {
				distSq = distSq.Add(dy.Pow(2))
			}

			repulsion := distSq.MulScalar(invTwoSigmaSq).Exp() // [B]
			batchMean := repulsion.Mean()

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

	hits := 0
	totalIntensity := float64(0)
	n := 0

	for _, loc := range locations[1:] {
		grid := loc.Data().Unsqueeze(1).Unsqueeze(2)
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
func RecodeLoss(recon, target *autograd.Variable) *autograd.Variable {
	return nn.MSELoss(recon, target)
}
