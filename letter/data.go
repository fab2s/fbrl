// Letter dataset and batched loader for training.
package letter

import (
	"math/rand/v2"

	"github.com/fab2s/goDl/tensor"
)

// LetterSample holds one rendered letter with metadata.
type LetterSample struct {
	Image     *tensor.Tensor // [1, H, W] noisy image
	Clean     *tensor.Tensor // [1, H, W] clean image (for attention guide)
	LetterIdx int64          // 0-25 (A=0, B=1, ...)
	CaseLabel float32        // 0.0=upper, 1.0=lower
}

// LetterBatch holds a stacked mini-batch ready for training.
type LetterBatch struct {
	Image     *tensor.Tensor // [B, 1, H, W]
	Clean     *tensor.Tensor // [B, 1, H, W]
	LetterIdx *tensor.Tensor // [B] int64
	CaseLabel *tensor.Tensor // [B, 1] float32
}

// LetterDataset holds all samples in memory.
type LetterDataset struct {
	Samples []LetterSample
}

// Len returns the number of samples.
func (d *LetterDataset) Len() int { return len(d.Samples) }

// LetterLoader iterates over a LetterDataset in shuffled batches.
//
//	loader := NewLetterLoader(ds, 32, true)
//	for loader.Next() {
//	    batch := loader.Batch()
//	    // ... training step ...
//	}
//	loader.Reset() // new epoch
type LetterLoader struct {
	ds        *LetterDataset
	batchSize int
	shuffle   bool
	dropLast  bool
	device    *tensor.Device
	perm      []int
	pos       int
	cur       *LetterBatch
	err       error
}

// NewLetterLoader creates a loader for the given dataset.
func NewLetterLoader(ds *LetterDataset, batchSize int, shuffle bool) *LetterLoader {
	l := &LetterLoader{
		ds:        ds,
		batchSize: batchSize,
		shuffle:   shuffle,
		dropLast:  true,
	}
	l.initPerm()
	return l
}

func (l *LetterLoader) initPerm() {
	n := l.ds.Len()
	l.perm = make([]int, n)
	for i := range l.perm {
		l.perm[i] = i
	}
	if l.shuffle {
		rand.Shuffle(n, func(i, j int) {
			l.perm[i], l.perm[j] = l.perm[j], l.perm[i]
		})
	}
	l.pos = 0
}

// Next advances to the next batch. Returns false when the epoch is exhausted.
func (l *LetterLoader) Next() bool {
	if l.err != nil {
		return false
	}
	n := l.ds.Len()
	if l.pos >= n {
		return false
	}
	end := l.pos + l.batchSize
	if end > n {
		if l.dropLast {
			return false
		}
		end = n
	}

	indices := l.perm[l.pos:end]
	l.pos = end
	B := len(indices)

	images := make([]*tensor.Tensor, B)
	cleans := make([]*tensor.Tensor, B)
	letterData := make([]int64, B)
	caseData := make([]float32, B)

	for i, idx := range indices {
		s := l.ds.Samples[idx]
		images[i] = s.Image
		cleans[i] = s.Clean
		letterData[i] = s.LetterIdx
		caseData[i] = s.CaseLabel
	}

	imgBatch := tensor.Stack(images, 0)
	if err := imgBatch.Err(); err != nil {
		l.err = err
		return false
	}
	cleanBatch := tensor.Stack(cleans, 0)
	if err := cleanBatch.Err(); err != nil {
		l.err = err
		return false
	}
	letterIdx, err := tensor.FromInt64(letterData, []int64{int64(B)})
	if err != nil {
		l.err = err
		return false
	}
	caseLabel, err := tensor.FromFloat32(caseData, []int64{int64(B), 1})
	if err != nil {
		l.err = err
		return false
	}

	// Move to target device if set.
	if l.device != nil {
		imgBatch = imgBatch.ToDevice(*l.device)
		cleanBatch = cleanBatch.ToDevice(*l.device)
		letterIdx = letterIdx.ToDevice(*l.device)
		caseLabel = caseLabel.ToDevice(*l.device)
	}

	l.cur = &LetterBatch{
		Image:     imgBatch,
		Clean:     cleanBatch,
		LetterIdx: letterIdx,
		CaseLabel: caseLabel,
	}
	return true
}

// Batch returns the current batch. Valid only after Next returns true.
func (l *LetterLoader) Batch() *LetterBatch { return l.cur }

// Err returns the first error encountered during iteration.
func (l *LetterLoader) Err() error { return l.err }

// Reset prepares the loader for a new epoch.
func (l *LetterLoader) Reset() {
	l.err = nil
	l.cur = nil
	l.initPerm()
}
