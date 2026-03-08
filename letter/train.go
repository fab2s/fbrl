// Training loop and configuration for the letter model.
package letter

import (
	"fmt"
	"os"
"time"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/graph"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// LetterConfig holds hyperparameters for letter model training.
type LetterConfig struct {
	// Model architecture.
	NClasses  int // number of letter classes (26)
	NGlimpses int // attention steps per forward pass
	PatchSize int // glimpse sensor patch size
	NScales   int // multi-resolution scales
	LatentDim int // hidden/latent dimension

	// Training.
	BatchSize   int
	Epochs      int
	LR          float64 // initial learning rate
	MinLR       float64 // minimum learning rate (cosine floor)
	MaxGradNorm float64 // gradient clipping norm

	// Loss weights.
	GuideWeight     float64 // attention guide loss weight
	DiversityWeight float64 // fixation diversity loss weight
	ReconWeight     float64 // reconstruction loss weight
	RecodeWeight    float64 // recode (case-swap reconstruction) weight
	BlurSigmaRatio  float64 // Gaussian blur sigma as fraction of min(H,W)
	DiversitySigma  float64 // repulsion radius in [-1,1] coords
	DiversityVy     float64 // vertical scale for diversity loss

	// Checkpointing.
	SaveDir            string
	CheckpointInterval int // save every N epochs (0 = only final)

	// Profiling.
	Profile bool // enable per-node timing
}

// DefaultLetterConfig returns sensible defaults for single-letter training.
func DefaultLetterConfig() LetterConfig {
	return LetterConfig{
		NClasses:  26,
		NGlimpses: 8,
		PatchSize: 12,
		NScales:   1,
		LatentDim: 128,

		BatchSize:   32,
		Epochs:      100,
		LR:          0.001,
		MinLR:       0,
		MaxGradNorm: 5.0,

		GuideWeight:     8.0,
		DiversityWeight: 1.0,
		ReconWeight:     1.0,
		RecodeWeight:    0.0,
		BlurSigmaRatio:  0.16,
		DiversitySigma:  0.1,
		DiversityVy:     1.0,

		SaveDir:            "letter/training",
		CheckpointInterval: 50,
	}
}

// EpochStats holds per-epoch averaged metrics.
type EpochStats struct {
	Epoch      int
	LetterLoss float64
	CaseLoss   float64
	LetterAcc  float64
	CaseAcc    float64
	ReconLoss  float64
	GuideLoss  float64
	DivLoss    float64
	TotalLoss  float64
	HitRate    float64
	LR         float64
	Duration   time.Duration
	ETA        string
}

// metricTags lists all metric tags recorded per batch.
var metricTags = []string{
	"letter_ce", "case_ce", "letter_acc", "case_acc", "recon_mse", "guide", "diversity", "total", "hit_rate", "lr",
}

// TrainLetter runs the full training loop for the letter model.
//
// All metrics are recorded into the graph's observation layer, enabling
// PlotHTML, ExportTrends, WriteLog, and trend analysis after training.
func TrainLetter(cfg LetterConfig, ds *LetterDataset, onEpoch func(EpochStats)) error {
	m := NewLetterModel(cfg.NClasses, cfg.NGlimpses, cfg.PatchSize, cfg.NScales, cfg.LatentDim)

	// Move model to CUDA if available.
	if tensor.CUDAAvailable() {
		m.Graph.SetDevice(tensor.CUDA)
		fmt.Println("Using CUDA")
	} else {
		fmt.Println("Using CPU")
	}

	optimizer := nn.NewAdam(m.Parameters(), cfg.LR)
	scheduler := nn.NewCosineScheduler(optimizer, cfg.LR, cfg.MinLR, cfg.Epochs)
	m.SetTraining(true)

	if cfg.Profile {
		m.Graph.EnableProfiling()
	}

	loader := NewLetterLoader(ds, cfg.BatchSize, true)
	loader.device = m.Graph.Device()

	// Ensure save directory exists.
	if cfg.SaveDir != "" {
		if err := os.MkdirAll(cfg.SaveDir, 0755); err != nil {
			return fmt.Errorf("create save dir: %w", err)
		}
	}

	// Open streaming log for tail -f during training.
	var logFile *os.File
	if cfg.SaveDir != "" {
		var err error
		logFile, err = os.Create(cfg.SaveDir + "/training.log")
		if err != nil {
			return fmt.Errorf("create log file: %w", err)
		}
		defer logFile.Close()
		fmt.Fprintf(logFile, "# fbrl letter training — %s\n", time.Now().Format(time.RFC3339))
		fmt.Fprintf(logFile, "# epochs=%d  batch=%d  lr=%.4f→%.4f  glimpses=%d\n",
			cfg.Epochs, cfg.BatchSize, cfg.LR, cfg.MinLR, cfg.NGlimpses)
	}

	for epoch := range cfg.Epochs {
		loader.Reset()
		epochStart := time.Now()
		nBatches := 0

		for loader.Next() {
			batch := loader.Batch()

			imgVar := autograd.NewVariable(batch.Image, false)
			caseVar := autograd.NewVariable(batch.CaseLabel, false)
			cleanVar := autograd.NewVariable(batch.Clean, false)

			result := m.Forward(imgVar, caseVar)

			// Debug: print fixation locations and guide diagnostics on first batch.
			if epoch == 0 && nBatches == 0 {
				fmt.Printf("DEBUG: %d traces (locations)\n", len(result.Locations))
				for i, loc := range result.Locations {
					d := loc.Data()
					fmt.Printf("  loc[%d]: shape=%v device=%v", i, d.Shape(), d.Device())
					vals, err := d.ToCPU().Float32Data()
					fmt.Printf(" len=%d err=%v", len(vals), err)
					if len(vals) >= 2 {
						fmt.Printf(" sample[0]=(%.4f, %.4f)", vals[0], vals[1])
					}
					fmt.Println()
				}
				// Check clean image stats.
				cleanData, _ := batch.Clean.ToCPU().Float32Data()
				sum := float32(0)
				for _, v := range cleanData {
					sum += v
				}
				fmt.Printf("DEBUG: clean image mean=%.6f (nPixels=%d)\n", float64(sum)/float64(len(cleanData)), len(cleanData))
			}

			// Classification losses.
			letterTarget := autograd.NewVariable(batch.LetterIdx, false)
			caseTarget := autograd.NewVariable(caseIdxFromFloat(batch.CaseLabel), false)
			letterLoss := nn.CrossEntropyLoss(result.LetterLogits, letterTarget)
			caseLoss := nn.CrossEntropyLoss(result.CaseLogits, caseTarget)

			// Reconstruction loss.
			reconLoss := nn.MSELoss(result.Recon, imgVar)

			// Attention losses.
			guideLoss := AttentionGuideLoss(cleanVar, result.Locations, cfg.BlurSigmaRatio)
			divLoss := FixationDiversityLoss(result.Locations, cfg.DiversitySigma, cfg.DiversityVy)

			// Total loss.
			total := letterLoss.Add(caseLoss)
			total = total.Add(reconLoss.MulScalar(cfg.ReconWeight))
			total = total.Add(guideLoss.MulScalar(cfg.GuideWeight))
			total = total.Add(divLoss.MulScalar(cfg.DiversityWeight))

			if err := total.Err(); err != nil {
				return fmt.Errorf("epoch %d: forward error: %w", epoch, err)
			}

			optimizer.ZeroGrad()
			total.Backward()
			nn.ClipGradNorm(m.Parameters(), cfg.MaxGradNorm)
			optimizer.Step()

			// Break gradient chain to free computation graph memory.
			m.Graph.DetachState()

			// Record metrics into graph observation layer.
			hr, _ := FixationHitRate(cleanVar, result.Locations, 0.3)
			letterAcc := accuracy(result.LetterLogits.Data(), batch.LetterIdx)
			caseAcc := accuracy(result.CaseLogits.Data(), caseIdxFromFloat(batch.CaseLabel))
			m.Graph.Record("letter_ce", letterLoss.Item())
			m.Graph.Record("case_ce", caseLoss.Item())
			m.Graph.Record("letter_acc", letterAcc)
			m.Graph.Record("case_acc", caseAcc)
			m.Graph.Record("recon_mse", reconLoss.Item())
			m.Graph.Record("guide", guideLoss.Item()*cfg.GuideWeight)
			m.Graph.Record("diversity", divLoss.Item())
			m.Graph.Record("total", total.Item())
			m.Graph.Record("hit_rate", hr)
			m.Graph.Record("lr", scheduler.LR())

			if cfg.Profile {
				m.Graph.CollectTimings()
			}

			nBatches++
		}

		if err := loader.Err(); err != nil {
			return fmt.Errorf("epoch %d: loader error: %w", epoch, err)
		}

		if nBatches == 0 {
			return fmt.Errorf("epoch %d: no batches produced", epoch)
		}

		// Flush batch means → epoch history.
		m.Graph.Flush()
		if cfg.Profile {
			m.Graph.FlushTimings()
		}

		scheduler.Step()

		epochDur := time.Since(epochStart)
		eta := m.Graph.ETA(cfg.Epochs)

		// Build epoch stats from flushed trends.
		stats := EpochStats{
			Epoch:      epoch,
			LetterLoss: m.Graph.Trend("letter_ce").Latest(),
			CaseLoss:   m.Graph.Trend("case_ce").Latest(),
			LetterAcc:  m.Graph.Trend("letter_acc").Latest(),
			CaseAcc:    m.Graph.Trend("case_acc").Latest(),
			ReconLoss:  m.Graph.Trend("recon_mse").Latest(),
			GuideLoss:  m.Graph.Trend("guide").Latest(),
			DivLoss:    m.Graph.Trend("diversity").Latest(),
			TotalLoss:  m.Graph.Trend("total").Latest(),
			HitRate:    m.Graph.Trend("hit_rate").Latest(),
			LR:         scheduler.LR(),
			Duration:   epochDur,
			ETA:        graph.FormatDuration(eta),
		}

		if onEpoch != nil {
			onEpoch(stats)
		}

		// Append to streaming log.
		if logFile != nil {
			fmt.Fprintf(logFile, "epoch %3d  ltr=%.4f(%.0f%%)  case=%.4f(%.0f%%)  recon=%.4f  guide=%.4f  div=%.4f  hit=%.0f%%  lr=%.6f  [%s  ETA %s]\n",
				epoch+1, stats.LetterLoss, stats.LetterAcc*100, stats.CaseLoss, stats.CaseAcc*100,
				stats.ReconLoss, stats.GuideLoss, stats.DivLoss, stats.HitRate*100, stats.LR,
				stats.Duration, stats.ETA)
			logFile.Sync()
		}

		// Checkpoint.
		if cfg.SaveDir != "" && cfg.CheckpointInterval > 0 && (epoch+1)%cfg.CheckpointInterval == 0 {
			path := fmt.Sprintf("%s/checkpoint_epoch_%d.bin", cfg.SaveDir, epoch+1)
			if err := nn.SaveParametersFile(path, m.Parameters()); err != nil {
				return fmt.Errorf("save checkpoint: %w", err)
			}
		}
	}

	// Save final outputs.
	if cfg.SaveDir != "" {
		if err := nn.SaveParametersFile(cfg.SaveDir+"/model_final.bin", m.Parameters()); err != nil {
			return fmt.Errorf("save final model: %w", err)
		}
		if err := m.Graph.PlotHTML(cfg.SaveDir + "/training.html"); err != nil {
			fmt.Fprintf(os.Stderr, "warning: plot HTML: %v\n", err)
		}
		if err := m.Graph.ExportTrends(cfg.SaveDir+"/training.csv", metricTags...); err != nil {
			fmt.Fprintf(os.Stderr, "warning: export CSV: %v\n", err)
		}
		if cfg.Profile {
			if err := m.Graph.PlotTimingsHTML(cfg.SaveDir + "/timings.html"); err != nil {
				fmt.Fprintf(os.Stderr, "warning: plot timings: %v\n", err)
			}
		}
	}

	return nil
}

// caseIdxFromFloat converts [B, 1] float case labels to [B] int64 indices.
// Stays on-device — no CPU round-trip.
func caseIdxFromFloat(caseLabel *tensor.Tensor) *tensor.Tensor {
	return caseLabel.Squeeze(1).GTScalar(0.5).ToInt64()
}

// accuracy computes classification accuracy from logits and target indices.
// logits: [B, C], targets: [B] int64. Returns fraction correct in [0, 1].
func accuracy(logits, targets *tensor.Tensor) float64 {
	preds, _ := logits.ToCPU().ArgMax(1, false).Int64Data()
	truth, _ := targets.ToCPU().Int64Data()
	if len(preds) == 0 {
		return 0
	}
	correct := 0
	for i, p := range preds {
		if p == truth[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(preds))
}
