// VisualDecoder — reconstructs images from latent vectors.
package model

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// VisualDecoder generates images from latent vectors using transposed convolutions.
//
// FC projects to spatial feature map, then two stride-2 deconvs to output_shape.
// For the letter model: input_dim=latentDim+1 (latent + case label), output 128x128.
//
// Implements nn.NamedInputModule: receives "latent" and "case" via Using refs.
// The stream input (from the graph chain) is ignored — the decoder consumes
// only the refs.
type VisualDecoder struct {
	spatialH, spatialW int64

	fc      *nn.Linear
	deconv1 *nn.ConvTranspose2d // 128 -> 64, stride 2
	bn1     *nn.BatchNorm
	deconv2 *nn.ConvTranspose2d // 64 -> 32, stride 2
	bn2     *nn.BatchNorm
	conv    *nn.Conv2d // 32 -> 1 (output channel)
}

// NewVisualDecoder creates a decoder for the given input dimension and output shape.
func NewVisualDecoder(inputDim int64, outputH, outputW int64) *VisualDecoder {
	spatialH := outputH / 4
	spatialW := outputW / 4

	deconv1 := nn.MustConvTranspose2d(128, 64, 3)
	deconv1.Stride = [2]int64{2, 2}
	deconv1.Padding = [2]int64{1, 1}
	deconv1.OutputPadding = [2]int64{1, 1}

	deconv2 := nn.MustConvTranspose2d(64, 32, 5)
	deconv2.Stride = [2]int64{2, 2}
	deconv2.Padding = [2]int64{2, 2}
	deconv2.OutputPadding = [2]int64{1, 1}

	convOut := nn.MustConv2d(32, 1, 3)
	convOut.Padding = [2]int64{1, 1}

	bn1, err := nn.NewBatchNorm(64)
	if err != nil {
		panic(err)
	}
	bn2, err := nn.NewBatchNorm(32)
	if err != nil {
		panic(err)
	}

	return &VisualDecoder{
		spatialH: spatialH,
		spatialW: spatialW,
		fc:       nn.MustLinear(inputDim, 128*spatialH*spatialW),
		deconv1:  deconv1,
		bn1:      bn1,
		deconv2:  deconv2,
		bn2:      bn2,
		conv:     convOut,
	}
}

// Forward decodes a latent vector (optionally concatenated with condition) to an image.
// z: [B, inputDim]. Returns: [B, 1, outputH, outputW].
func (d *VisualDecoder) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return d.decode(inputs[0])
}

// ForwardNamed receives "latent" and "case" via refs, ignoring the stream.
func (d *VisualDecoder) ForwardNamed(stream *autograd.Variable, refs map[string]*autograd.Variable) *autograd.Variable {
	latent := refs["latent"]
	caseLabel := refs["case"]
	return d.decode(latent.Cat(caseLabel, 1))
}

// RefNames declares expected Using refs for build-time validation.
func (d *VisualDecoder) RefNames() []string { return []string{"latent", "case"} }

// decode runs the actual deconvolution stack.
func (d *VisualDecoder) decode(z *autograd.Variable) *autograd.Variable {
	B := z.Data().Shape()[0]
	x := d.fc.Forward(z).ReLU()
	x = x.Reshape([]int64{B, 128, d.spatialH, d.spatialW})

	x = d.deconv1.Forward(x)
	x = batchNorm2d(x, d.bn1)
	x = x.ReLU()

	x = d.deconv2.Forward(x)
	x = batchNorm2d(x, d.bn2)
	x = x.ReLU()

	x = d.conv.Forward(x)
	return x.Sigmoid()
}

// batchNorm2d applies 1D batch norm to a 4D tensor by reshaping.
// Input: [B, C, H, W] -> [B*H*W, C] -> BN -> [B, C, H, W]
func batchNorm2d(x *autograd.Variable, bn *nn.BatchNorm) *autograd.Variable {
	shape := x.Data().Shape() // [B, C, H, W]
	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	// Transpose to [B, H, W, C] then reshape to [B*H*W, C]
	xt := x.Transpose(1, 2).Transpose(2, 3) // [B, H, W, C]
	flat := xt.Reshape([]int64{B * H * W, C})
	normed := bn.Forward(flat)
	// Reshape back to [B, H, W, C] then transpose to [B, C, H, W]
	back := normed.Reshape([]int64{B, H, W, C})
	return back.Transpose(2, 3).Transpose(1, 2) // [B, C, H, W]
}

// Parameters returns all learnable parameters.
func (d *VisualDecoder) Parameters() []*nn.Parameter {
	var params []*nn.Parameter
	params = append(params, d.fc.Parameters()...)
	params = append(params, d.deconv1.Parameters()...)
	params = append(params, d.bn1.Parameters()...)
	params = append(params, d.deconv2.Parameters()...)
	params = append(params, d.bn2.Parameters()...)
	params = append(params, d.conv.Parameters()...)
	return params
}

// SetTraining propagates training mode to batch norm layers.
func (d *VisualDecoder) SetTraining(training bool) {
	d.bn1.SetTraining(training)
	d.bn2.SetTraining(training)
}
