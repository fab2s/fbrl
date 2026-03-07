// Attention guide loss — pulls fixations toward image content.
package loss

import (
	"math"

	"github.com/fab2s/goDl/autograd"
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
	B, H, W := shape[0], shape[2], shape[3]

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
	gaussT, _ := tensor.FromFloat32(kernelData, []int64{int64(k)})
	kernelH := gaussT.Reshape([]int64{1, 1, int64(k), 1})
	kernelW := gaussT.Reshape([]int64{1, 1, 1, int64(k)})

	guide := imgData.Conv2d(kernelH, nil, []int64{1, 1}, []int64{int64(halfK), 0}, []int64{1, 1}, 1)
	guide = guide.Conv2d(kernelW, nil, []int64{1, 1}, []int64{0, int64(halfK)}, []int64{1, 1}, 1)
	guideVar := autograd.NewVariable(guide, false)

	// Sample guide at each learned fixation and accumulate.
	nLocs := len(locations) - 1
	var total *autograd.Variable
	for _, loc := range locations[1:] {
		grid := loc.Reshape([]int64{B, 1, 1, 2})
		sampled := guideVar.GridSample(grid, 0, 0, true) // bilinear, zeros, align_corners
		batchMean := sampled.Sum().MulScalar(1.0 / float64(B))
		if total == nil {
			total = batchMean
		} else {
			total = total.Add(batchMean)
		}
	}

	// Negate: maximize guide values → minimize negative.
	return total.MulScalar(-1.0 / float64(nLocs))
}
