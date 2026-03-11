// Synthetic data generation for testing (no font rendering dependency).
package letter

import (
	"github.com/fab2s/goDl/tensor"
)

// NewSyntheticDataset creates a dataset of random images for testing.
// Each sample gets a random 128x128 image with a random letter/case label.
// Not useful for real training — only for testing the pipeline.
func NewSyntheticDataset(n int) (*LetterDataset, error) {
	samples := make([]LetterSample, n)
	for i := range n {
		// Uniform [0, 1] random image.
		img, err := tensor.Rand([]int64{1, 128, 128})
		if err != nil {
			return nil, err
		}

		letterIdx := int64(i % 26)
		var caseLabel float32
		if i%2 == 1 {
			caseLabel = 1.0
		}

		samples[i] = LetterSample{
			Image:     img,
			Clean:     img, // clean == noisy for synthetic
			LetterIdx: letterIdx,
			CaseLabel: caseLabel,
		}
	}
	return &LetterDataset{Samples: samples}, nil
}
