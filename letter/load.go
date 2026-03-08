// Load letter datasets from Python-generated data directories.
package letter

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/fab2s/goDl/tensor"
)

// metadata entry from Python's metadata.json.
type metaEntry struct {
	Image  string `json:"image"`
	Clean  string `json:"clean"`
	Letter string `json:"letter"`
	Case   string `json:"case"`
	Font   string `json:"font"`
}

// LoadLetterDataset loads a dataset from a Python-generated data directory.
// The directory must contain metadata.json and referenced PNG files.
func LoadLetterDataset(dir string) (*LetterDataset, error) {
	metaPath := filepath.Join(dir, "metadata.json")
	f, err := os.Open(metaPath)
	if err != nil {
		return nil, fmt.Errorf("open metadata: %w", err)
	}
	defer f.Close()

	var meta map[string]metaEntry
	if err := json.NewDecoder(f).Decode(&meta); err != nil {
		return nil, fmt.Errorf("parse metadata: %w", err)
	}

	// Cache clean images (shared across noisy variants).
	cleanCache := make(map[string]*tensor.Tensor)

	samples := make([]LetterSample, 0, len(meta))
	for _, entry := range meta {
		// Load noisy image.
		imgPath := entry.Image
		if !filepath.IsAbs(imgPath) {
			imgPath = filepath.Join(dir, imgPath)
		}
		imgT, err := loadGrayPNG(imgPath)
		if err != nil {
			return nil, fmt.Errorf("load image %s: %w", imgPath, err)
		}

		// Load or cache clean image.
		var cleanT *tensor.Tensor
		if entry.Clean != "" {
			cleanPath := entry.Clean
			if !filepath.IsAbs(cleanPath) {
				cleanPath = filepath.Join(dir, cleanPath)
			}
			if cached, ok := cleanCache[cleanPath]; ok {
				cleanT = cached
			} else {
				cleanT, err = loadGrayPNG(cleanPath)
				if err != nil {
					return nil, fmt.Errorf("load clean %s: %w", cleanPath, err)
				}
				cleanCache[cleanPath] = cleanT
			}
		} else {
			cleanT = imgT
		}

		// Letter index: 'A'-'Z' → 0-25.
		letter := strings.ToUpper(entry.Letter)
		if len(letter) == 0 {
			continue
		}
		letterIdx := int64(letter[0] - 'A')
		if letterIdx < 0 || letterIdx > 25 {
			continue
		}

		// Case label.
		var caseLabel float32
		if entry.Case == "lower" {
			caseLabel = 1.0
		}

		samples = append(samples, LetterSample{
			Image:     imgT,
			Clean:     cleanT,
			LetterIdx: letterIdx,
			CaseLabel: caseLabel,
		})
	}

	if len(samples) == 0 {
		return nil, fmt.Errorf("no valid samples found in %s", dir)
	}

	return &LetterDataset{Samples: samples}, nil
}

// loadGrayPNG loads a grayscale PNG as a [1, H, W] float32 tensor in [0, 1].
func loadGrayPNG(path string) (*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	h, w := bounds.Dy(), bounds.Dx()
	data := make([]float32, h*w)

	switch g := img.(type) {
	case *image.Gray:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				data[y*w+x] = float32(g.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y) / 255.0
			}
		}
	default:
		// Fallback: convert any image to grayscale via luminance.
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
				// Standard luminance weights (same as Pillow 'L' mode).
				lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
				data[y*w+x] = float32(lum / 65535.0)
			}
		}
	}

	return tensor.FromFloat32(data, []int64{1, int64(h), int64(w)})
}
