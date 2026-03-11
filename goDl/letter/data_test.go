package letter

import (
	"testing"
)

func TestSyntheticDataset(t *testing.T) {
	ds, err := NewSyntheticDataset(100)
	if err != nil {
		t.Fatal(err)
	}

	if ds.Len() != 100 {
		t.Errorf("expected 100 samples, got %d", ds.Len())
	}

	s := ds.Samples[0]
	shape := s.Image.Shape()
	if shape[0] != 1 || shape[1] != 128 || shape[2] != 128 {
		t.Errorf("expected [1, 128, 128], got %v", shape)
	}
}

func TestLetterLoaderBasic(t *testing.T) {
	ds, err := NewSyntheticDataset(50)
	if err != nil {
		t.Fatal(err)
	}

	loader := NewLetterLoader(ds, 16, false)

	batches := 0
	totalSamples := 0
	for loader.Next() {
		batch := loader.Batch()
		batches++

		imgShape := batch.Image.Shape()
		if imgShape[0] != 16 {
			t.Errorf("batch %d: expected batch size 16, got %d", batches, imgShape[0])
		}
		if imgShape[1] != 1 || imgShape[2] != 128 || imgShape[3] != 128 {
			t.Errorf("batch %d: expected [16, 1, 128, 128], got %v", batches, imgShape)
		}

		cleanShape := batch.Clean.Shape()
		if cleanShape[0] != imgShape[0] {
			t.Errorf("clean batch size mismatch")
		}

		letterShape := batch.LetterIdx.Shape()
		if letterShape[0] != 16 {
			t.Errorf("letter idx shape: expected [16], got %v", letterShape)
		}

		caseShape := batch.CaseLabel.Shape()
		if caseShape[0] != 16 || caseShape[1] != 1 {
			t.Errorf("case label shape: expected [16, 1], got %v", caseShape)
		}

		totalSamples += int(imgShape[0])
	}

	if err := loader.Err(); err != nil {
		t.Fatal(err)
	}

	// 50 samples, batch 16, dropLast=true → 3 batches (48 samples)
	if batches != 3 {
		t.Errorf("expected 3 batches, got %d", batches)
	}

	t.Logf("batches: %d, total samples: %d", batches, totalSamples)
}

func TestLetterLoaderShuffle(t *testing.T) {
	ds, err := NewSyntheticDataset(32)
	if err != nil {
		t.Fatal(err)
	}

	loader := NewLetterLoader(ds, 32, true)

	// First epoch.
	if !loader.Next() {
		t.Fatal("expected first batch")
	}
	b1 := loader.Batch()
	if loader.Next() {
		t.Error("expected no more batches")
	}

	// Second epoch — should still work after Reset.
	loader.Reset()
	if !loader.Next() {
		t.Fatal("expected batch after reset")
	}
	b2 := loader.Batch()

	// Both batches should have same shape.
	if b1.Image.Shape()[0] != b2.Image.Shape()[0] {
		t.Error("batch sizes differ after reset")
	}

	t.Logf("epoch 1 letter[0]: %v, epoch 2 letter[0]: %v",
		b1.LetterIdx.Shape(), b2.LetterIdx.Shape())
}

func TestLetterLoaderEmpty(t *testing.T) {
	ds := &LetterDataset{}
	loader := NewLetterLoader(ds, 16, false)
	if loader.Next() {
		t.Error("expected no batches from empty dataset")
	}
}
