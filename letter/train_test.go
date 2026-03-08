package letter

import (
	"testing"
)

func TestTrainLetterSmoke(t *testing.T) {
	ds, err := NewSyntheticDataset(64)
	if err != nil {
		t.Fatal(err)
	}

	cfg := DefaultLetterConfig()
	cfg.NGlimpses = 2 // minimal for speed
	cfg.PatchSize = 8
	cfg.LatentDim = 32
	cfg.BatchSize = 16
	cfg.Epochs = 2
	cfg.LR = 0.001

	var stats []EpochStats
	err = TrainLetter(cfg, ds, func(s EpochStats) {
		stats = append(stats, s)
		t.Logf("epoch %d: letter=%.4f case=%.4f recon=%.4f guide=%.4f div=%.4f total=%.4f hit=%.0f%% lr=%.6f [%s ETA %s]",
			s.Epoch, s.LetterLoss, s.CaseLoss, s.ReconLoss, s.GuideLoss, s.DivLoss,
			s.TotalLoss, s.HitRate*100, s.LR, s.Duration, s.ETA)
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(stats) != 2 {
		t.Errorf("expected 2 epoch stats, got %d", len(stats))
	}

	// Sanity: losses should be finite positive numbers.
	for _, s := range stats {
		if s.LetterLoss <= 0 {
			t.Errorf("epoch %d: letter loss should be positive, got %f", s.Epoch, s.LetterLoss)
		}
		if s.ReconLoss <= 0 {
			t.Errorf("epoch %d: recon loss should be positive, got %f", s.Epoch, s.ReconLoss)
		}
	}

	// LR should decrease with cosine schedule.
	if len(stats) == 2 && stats[1].LR >= stats[0].LR {
		t.Errorf("LR should decrease: epoch 0=%.6f, epoch 1=%.6f", stats[0].LR, stats[1].LR)
	}
}
