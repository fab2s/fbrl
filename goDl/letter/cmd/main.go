// CLI entry point for letter model training.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/fab2s/fbrl/letter"
)

func main() {
	cfg := letter.DefaultLetterConfig()

	// Data paths.
	dataDir := flag.String("data", "", "path to training data directory")
	synthetic := flag.Int("synthetic", 0, "generate N synthetic samples instead of loading data")
	saveDir := flag.String("save", cfg.SaveDir, "directory to save model, plots, and logs")

	// Architecture overrides.
	flag.IntVar(&cfg.NGlimpses, "glimpses", cfg.NGlimpses, "attention steps per forward pass")
	flag.IntVar(&cfg.PatchSize, "patch-size", cfg.PatchSize, "glimpse sensor patch size")
	flag.IntVar(&cfg.NScales, "scales", cfg.NScales, "multi-resolution scales")
	flag.IntVar(&cfg.LatentDim, "latent-dim", cfg.LatentDim, "hidden/latent dimension")

	// Training overrides.
	flag.IntVar(&cfg.Epochs, "epochs", cfg.Epochs, "number of training epochs")
	flag.IntVar(&cfg.BatchSize, "batch-size", cfg.BatchSize, "batch size")
	flag.Float64Var(&cfg.LR, "lr", cfg.LR, "initial learning rate")
	flag.Float64Var(&cfg.MinLR, "min-lr", cfg.MinLR, "minimum learning rate (cosine floor)")
	flag.Float64Var(&cfg.MaxGradNorm, "max-grad-norm", cfg.MaxGradNorm, "gradient clipping norm")

	// Loss weight overrides.
	flag.Float64Var(&cfg.GuideWeight, "guide-weight", cfg.GuideWeight, "attention guide loss weight")
	flag.Float64Var(&cfg.DiversityWeight, "diversity-weight", cfg.DiversityWeight, "fixation diversity loss weight")
	flag.Float64Var(&cfg.ReconWeight, "recon-weight", cfg.ReconWeight, "reconstruction loss weight")

	// Checkpointing.
	flag.IntVar(&cfg.CheckpointInterval, "checkpoint-interval", cfg.CheckpointInterval, "save every N epochs (0 = only final)")

	// Profiling.
	flag.BoolVar(&cfg.Profile, "profile", cfg.Profile, "enable per-node timing profiling")

	flag.Parse()

	cfg.SaveDir = *saveDir

	// Load or generate dataset.
	var ds *letter.LetterDataset
	var err error

	switch {
	case *synthetic > 0:
		ds, err = letter.NewSyntheticDataset(*synthetic)
		if err != nil {
			log.Fatalf("synthetic dataset: %v", err)
		}
		fmt.Printf("Generated %d synthetic samples\n", ds.Len())

	case *dataDir != "":
		ds, err = letter.LoadLetterDataset(*dataDir)
		if err != nil {
			log.Fatalf("load dataset: %v", err)
		}
		fmt.Printf("Loaded %d samples from %s\n", ds.Len(), *dataDir)

	default:
		fmt.Fprintln(os.Stderr, "Usage: letter --data <dir> or --synthetic <N>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Train.
	fmt.Printf("Training: %d epochs, batch %d, lr %.4f→%.4f (cosine), %d glimpses\n",
		cfg.Epochs, cfg.BatchSize, cfg.LR, cfg.MinLR, cfg.NGlimpses)
	if cfg.SaveDir != "" {
		fmt.Printf("Output: %s\n", cfg.SaveDir)
	}
	fmt.Println()

	err = letter.TrainLetter(cfg, ds, func(s letter.EpochStats) {
		fmt.Printf("Epoch %3d  Ltr %.4f(%.0f%%)  Case %.4f(%.0f%%)  Recon %.4f  Guide %.4f  Div %.4f  Hit %.0f%%  lr %.6f  [%s  ETA %s]\n",
			s.Epoch+1, s.LetterLoss, s.LetterAcc*100, s.CaseLoss, s.CaseAcc*100,
			s.ReconLoss, s.GuideLoss, s.DivLoss, s.HitRate*100, s.LR, s.Duration, s.ETA)
	})
	if err != nil {
		log.Fatalf("training: %v", err)
	}

	fmt.Println("\nTraining complete.")
	if cfg.SaveDir != "" {
		fmt.Printf("Model:   %s/model_final.bin\n", cfg.SaveDir)
		fmt.Printf("Plot:    %s/training.html\n", cfg.SaveDir)
		fmt.Printf("CSV:     %s/training.csv\n", cfg.SaveDir)
		fmt.Printf("Log:     %s/training.log\n", cfg.SaveDir)
		if cfg.Profile {
			fmt.Printf("Timings: %s/timings.html\n", cfg.SaveDir)
		}
	}
}
