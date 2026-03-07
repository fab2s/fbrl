// Sensor interface — any component that extracts features from image + location.
package model

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// Sensor extracts a feature vector from an image at a given fixation location.
// Both GlimpseSensor (foveal, sharp) and future PeripheralSensor (wide, blurred)
// implement this interface — the controller doesn't care which one it's using.
type Sensor interface {
	Forward(image, location *autograd.Variable) *autograd.Variable
	Parameters() []*nn.Parameter
}
