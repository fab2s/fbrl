// Graph-compatible modules for the attention pipeline.
package letter

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// Sensor extracts a feature vector from an image at a given fixation location.
// Both GlimpseSensor (foveal, sharp) and future PeripheralSensor (wide, blurred)
// implement this interface — the controller doesn't care which one it's using.
type Sensor interface {
	Forward(image, location *autograd.Variable) *autograd.Variable
	Parameters() []*nn.Parameter
}

// Identity passes the input through unchanged. Used as the graph entry
// point to tag the image before routing it to downstream modules.
type Identity struct{}

func (i *Identity) Forward(inputs ...*autograd.Variable) *autograd.Variable { return inputs[0] }
func (i *Identity) Parameters() []*nn.Parameter                            { return nil }

// H0Init ignores its input (image) and returns the learned initial hidden
// state expanded to match the batch dimension.
type H0Init struct {
	H0        *nn.Parameter
	hiddenDim int64
}

// NewH0Init creates a learned initial hidden state of the given dimension.
func NewH0Init(hiddenDim int64) *H0Init {
	h0Data, err := tensor.Zeros([]int64{1, hiddenDim})
	if err != nil {
		panic(err)
	}
	return &H0Init{
		H0:        nn.NewParameter(h0Data, "h0"),
		hiddenDim: hiddenDim,
	}
}

func (h *H0Init) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	B := inputs[0].Data().Shape()[0]
	return h.H0.Reshape([]int64{1, h.hiddenDim}).Expand([]int64{B, h.hiddenDim})
}

func (h *H0Init) Parameters() []*nn.Parameter { return []*nn.Parameter{h.H0} }

// AttentionStep is the loop body: receives h as stream, image via ref,
// manages location as internal recurrent state.
//
// Implements nn.NamedInputModule so the graph Loop forwards the "image"
// ref via ForwardNamed. Implements nn.Resettable and nn.Traced so the
// graph auto-resets before each forward and the loop collects the
// fixation trajectory automatically.
type AttentionStep struct {
	sensor  Sensor
	gru     *nn.GRUCell
	locHead *nn.Linear

	// Recurrent state (auto-reset by graph before each forward pass).
	location *autograd.Variable
}

// NewAttentionStep creates an attention step with the given sensor and dimensions.
func NewAttentionStep(sensor Sensor, hiddenDim int64) *AttentionStep {
	gru, err := nn.NewGRUCell(hiddenDim, hiddenDim)
	if err != nil {
		panic(err)
	}
	return &AttentionStep{
		sensor:  sensor,
		gru:     gru,
		locHead: nn.MustLinear(hiddenDim, 2),
	}
}

// Reset initializes location to zeros on the same device as the model parameters.
// Called automatically by the graph before each forward pass (implements nn.Resettable).
func (s *AttentionStep) Reset(batchSize int64) {
	locT, err := tensor.Zeros([]int64{batchSize, 2})
	if err != nil {
		panic(err)
	}
	// Match device of model parameters (set via Graph.SetDevice).
	dev := s.locHead.Weight.Data().Device()
	locT = locT.ToDevice(dev)
	s.location = autograd.NewVariable(locT, false)
}

// Trace returns the current fixation location. Called by the loop
// executor to collect the trajectory (implements nn.Traced).
func (s *AttentionStep) Trace() *autograd.Variable {
	return s.location
}

// Forward is the plain module interface (called when no Using refs).
func (s *AttentionStep) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return s.ForwardNamed(inputs[0], nil)
}

// ForwardNamed receives h as stream and image via refs.
func (s *AttentionStep) ForwardNamed(h *autograd.Variable, refs map[string]*autograd.Variable) *autograd.Variable {
	image := refs["image"]
	glimpse := s.sensor.Forward(image, s.location)
	newH := s.gru.Forward(glimpse, h)
	s.location = s.locHead.Forward(newH).Tanh()
	return newH
}

// RefNames declares the expected Using refs for build-time validation.
func (s *AttentionStep) RefNames() []string { return []string{"image"} }

func (s *AttentionStep) Parameters() []*nn.Parameter {
	var params []*nn.Parameter
	params = append(params, s.sensor.Parameters()...)
	params = append(params, s.gru.Parameters()...)
	params = append(params, s.locHead.Parameters()...)
	return params
}

func (s *AttentionStep) SetTraining(training bool) {}

// Detach breaks the gradient chain on the carried location state.
// Called by Graph.DetachState via the nn.Detachable interface.
func (s *AttentionStep) Detach() {
	if s.location != nil {
		s.location = autograd.NewVariable(s.location.Data(), false)
	}
}

// LatentHead projects the final hidden state to the latent space.
type LatentHead struct {
	fc *nn.Linear
}

func NewLatentHead(hiddenDim, latentDim int64) *LatentHead {
	return &LatentHead{fc: nn.MustLinear(hiddenDim, latentDim)}
}

func (l *LatentHead) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return l.fc.Forward(inputs[0])
}

func (l *LatentHead) Parameters() []*nn.Parameter { return l.fc.Parameters() }

// SelectFirst is a merge module that returns the first input.
// Used after Split+TagGroup when the graph output is irrelevant
// (we access individual heads via Tagged).
type SelectFirst struct{}

func (s *SelectFirst) Forward(inputs ...*autograd.Variable) *autograd.Variable { return inputs[0] }
func (s *SelectFirst) Parameters() []*nn.Parameter                            { return nil }
