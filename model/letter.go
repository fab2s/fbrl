// LetterModel — single-letter recognition via foveal attention.
//
// Built as a graph with two inputs (image, case label):
//
//	image → Tag("image") → H0Init → Loop(AttentionStep).Using("image").Tag("attention")
//	  → LatentHead → Tag("latent") → Split(letterHead, caseHead)
//	  → TagGroup("heads") → Merge → Decoder.Using("latent", "case") → Tag("recon")
//
// Fixation locations are collected automatically via graph Traces.
// All outputs are accessible via Tagged after Forward.
package model

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/graph"
	"github.com/fab2s/goDl/nn"
)

// LetterResult holds everything from a forward pass.
type LetterResult struct {
	Recon        *autograd.Variable   // [B, 1, 128, 128] reconstructed image
	LetterLogits *autograd.Variable   // [B, nClasses] letter identity
	CaseLogits   *autograd.Variable   // [B, 2] upper/lower
	Locations    []*autograd.Variable // fixation trajectory
	Latent       *autograd.Variable   // [B, latentDim]
}

// LetterModel is the single-letter model built as a computation graph.
type LetterModel struct {
	Graph *graph.Graph
}

// NewLetterModel creates the single-letter model.
func NewLetterModel(nClasses, nGlimpses, patchSize, nScales, latentDim int) *LetterModel {
	ld := int64(latentDim)

	sensor := NewGlimpseSensor(int64(patchSize), int64(patchSize), nScales, latentDim)
	step := NewAttentionStep(sensor, ld)
	letterHead := nn.MustLinear(ld, int64(nClasses))
	caseHead := nn.MustLinear(ld, 2)
	decoder := NewVisualDecoder(ld+1, 128, 128)

	g, err := graph.From(&Identity{}).Tag("image").
		Input("case").
		Through(NewH0Init(ld)).
		Loop(step).For(nGlimpses).Using("image").Tag("attention").
		Through(NewLatentHead(ld, ld)).Tag("latent").
		Split(letterHead, caseHead).TagGroup("heads").
		Merge(&SelectFirst{}).
		Through(decoder).Using("latent", "case").Tag("recon").
		Build()
	if err != nil {
		panic(err)
	}

	return &LetterModel{Graph: g}
}

// Forward runs the full pipeline: encode → classify → decode.
//
// img: [B, 1, 128, 128] input image.
// caseLabel: [B, 1] float — 0.0=upper, 1.0=lower (conditions the decoder).
func (m *LetterModel) Forward(img, caseLabel *autograd.Variable) *LetterResult {
	m.Graph.Forward(img, caseLabel)

	return &LetterResult{
		LetterLogits: m.Graph.Tagged("heads_0"),
		CaseLogits:   m.Graph.Tagged("heads_1"),
		Latent:       m.Graph.Tagged("latent"),
		Locations:    m.Graph.Traces("attention"),
		Recon:        m.Graph.Tagged("recon"),
	}
}

// Recode encodes an image and decodes with a different case label.
func (m *LetterModel) Recode(img, targetCase *autograd.Variable) (*autograd.Variable, []*autograd.Variable) {
	m.Graph.Forward(img, targetCase)

	return m.Graph.Tagged("recon"), m.Graph.Traces("attention")
}

// Parameters returns all learnable parameters.
func (m *LetterModel) Parameters() []*nn.Parameter {
	return m.Graph.Parameters()
}

// SetTraining propagates training mode.
func (m *LetterModel) SetTraining(training bool) {
	m.Graph.SetTraining(training)
}
