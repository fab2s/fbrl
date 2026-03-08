// Foveal attention sensor — extracts multi-resolution patches via grid_sample.
package letter

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// GlimpseSensor extracts a small patch from the image at a given (x,y) location,
// encodes it through a CNN, fuses with the location embedding, and returns
// a single latent vector representing "what + where".
//
// This is the foveal attention core: the model only sees what it looks at.
type GlimpseSensor struct {
	patchH, patchW int64
	scales         []int // e.g. [1] for foveal-only, [1,2] for foveal+peripheral

	// CNN: n_scales channels -> 32 -> 64 -> 128 -> AdaptiveAvgPool(1,1) -> flatten
	conv1, conv2, conv3 *nn.Conv2d
	glimpseFC           *nn.Linear
	locationFC          *nn.Linear
	combineFC           *nn.Linear
}

// NewGlimpseSensor creates a sensor with the given patch size and scale count.
func NewGlimpseSensor(patchH, patchW int64, nScales, latentDim int) *GlimpseSensor {
	scales := make([]int, nScales)
	for i := range nScales {
		scales[i] = 1 << i
	}

	conv1 := nn.MustConv2d(int64(nScales), 32, 3)
	conv1.Padding = [2]int64{1, 1}

	conv2 := nn.MustConv2d(32, 64, 3)
	conv2.Stride = [2]int64{2, 2}
	conv2.Padding = [2]int64{1, 1}

	conv3 := nn.MustConv2d(64, 128, 3)
	conv3.Stride = [2]int64{2, 2}
	conv3.Padding = [2]int64{1, 1}

	return &GlimpseSensor{
		patchH:     patchH,
		patchW:     patchW,
		scales:     scales,
		conv1:      conv1,
		conv2:      conv2,
		conv3:      conv3,
		glimpseFC:  nn.MustLinear(128, int64(latentDim)),
		locationFC: nn.MustLinear(2, 128),
		combineFC:  nn.MustLinear(int64(latentDim)+128, int64(latentDim)),
	}
}

// Parameters returns all learnable parameters.
func (g *GlimpseSensor) Parameters() []*nn.Parameter {
	var params []*nn.Parameter
	params = append(params, g.conv1.Parameters()...)
	params = append(params, g.conv2.Parameters()...)
	params = append(params, g.conv3.Parameters()...)
	params = append(params, g.glimpseFC.Parameters()...)
	params = append(params, g.locationFC.Parameters()...)
	params = append(params, g.combineFC.Parameters()...)
	return params
}

// Forward extracts a patch at the given location, encodes it, and fuses with location.
// image: [B, C, H, W], location: [B, 2] in [-1, 1].
// Returns: [B, latentDim].
func (g *GlimpseSensor) Forward(image, location *autograd.Variable) *autograd.Variable {
	imgShape := image.Data().Shape() // [B, C, H, W]
	H, W := imgShape[2], imgShape[3]

	// Extract one patch per scale via differentiable grid_sample.
	dev := image.Data().Device()
	var patches []*autograd.Variable
	for _, scale := range g.scales {
		grid := makeGrid(location, scale, g.patchH, g.patchW, H, W, dev)
		patch := image.GridSample(grid, 0, 0, true) // bilinear, zeros, align_corners
		patches = append(patches, patch)
	}

	// Cat scales as channels: [B, nScales, patchH, patchW]
	combined := patches[0]
	for i := 1; i < len(patches); i++ {
		combined = combined.Cat(patches[i], 1)
	}

	// CNN -> pool -> flatten
	feat := g.conv1.Forward(combined).ReLU()
	feat = g.conv2.Forward(feat).ReLU()
	feat = g.conv3.Forward(feat).ReLU()
	feat = feat.AdaptiveAvgPool2d([]int64{1, 1})
	feat = feat.Flatten(1)

	// Fuse "what I see" + "where I am"
	glimpseFeat := g.glimpseFC.Forward(feat).ReLU()
	locFeat := g.locationFC.Forward(location).ReLU()
	fused := glimpseFeat.Cat(locFeat, 1)
	return g.combineFC.Forward(fused).ReLU()
}

// makeGrid builds a sampling grid for grid_sample, centered on location.
// Returns a Variable of shape [B, patchH, patchW, 2].
func makeGrid(location *autograd.Variable, scale int, patchH, patchW, H, W int64, device tensor.Device) *autograd.Variable {
	locData := location.Data()
	B := locData.Shape()[0]

	// Normalized extent in [-1, 1] coords.
	deltaH := float64(scale) * float64(patchH) / float64(H)
	deltaW := float64(scale) * float64(patchW) / float64(W)

	// Build centered grid (no gradient needed for the base grid).
	gridY, _ := tensor.Linspace(-deltaH, deltaH, patchH)
	gridX, _ := tensor.Linspace(-deltaW, deltaW, patchW)

	// Meshgrid: expand gridX to [patchH, patchW], gridY to [patchH, patchW].
	gx := gridX.Reshape([]int64{1, patchW}).Expand([]int64{patchH, patchW})
	gy := gridY.Reshape([]int64{patchH, 1}).Expand([]int64{patchH, patchW})

	// Stack [gx, gy] -> [patchH, patchW, 2], unsqueeze to [1, patchH, patchW, 2]
	gxFlat := gx.Reshape([]int64{patchH * patchW})
	gyFlat := gy.Reshape([]int64{patchH * patchW})
	stacked := tensor.Stack([]*tensor.Tensor{gxFlat, gyFlat}, 1) // [patchH*patchW, 2]
	baseGrid := stacked.Reshape([]int64{1, patchH, patchW, 2})
	baseGrid = baseGrid.Expand([]int64{B, patchH, patchW, 2}).ToDevice(device)
	gridVar := autograd.NewVariable(baseGrid, false)

	// Shift by location: location is [B, 2], reshape to [B, 1, 1, 2] for broadcast.
	locReshaped := location.Unsqueeze(1).Unsqueeze(2)
	return gridVar.Add(locReshaped)
}

// SetTraining sets training mode (currently no-op, but matches Module interface).
func (g *GlimpseSensor) SetTraining(training bool) {}
