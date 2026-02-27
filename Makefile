SERVICE = fbrl
RUN     = docker compose exec $(SERVICE) python vision_training.py

# Overridable defaults (e.g. make train-gpu EPOCHS=500 DEVICE=cuda)
EPOCHS   ?= 100
CKPT     ?= 50
DEVICE   ?= auto
LETTERS  ?= Aa-Zz
VARIANTS ?= 20
NOISE    ?= 0.1
FONTS    ?= all
BATCH    ?= 52
GUIDE    ?= 8.0
SCAFFOLD_RATIO ?= 0.67
SCAFFOLD_FLOOR ?= 0.0
SCAFFOLD_EPOCHS ?=
TRANSFER ?=
SCAN_GLIMPSES ?= 5
READ_GLIMPSES ?= 6
SCAN_PATCH    ?= 12,18
READ_PATCH    ?= 12
SCAN_GUIDE    ?=
MASK     ?= 0.5
VY       ?= 1.0
SCAN_VY  ?= 0.3
READ_VY  ?= 1.5
EDGE     ?= 0.0
CONTENT  ?= 0.5
ISOLATION ?= 0.5
WORD_SCAN_GLIMPSES ?= 8
WORD_READ_GLIMPSES ?= 12
ISOLATION_DATA ?=
MULTI_HEAD ?=
ISO_RANDOM ?= 0.0
RESUME_FROM ?= model_final.pth

# Lifecycle
build:
	BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f $(SERVICE)

shell:
	docker compose exec $(SERVICE) bash

# Cleanup helpers — remove all files except .gitignore and archived logs
CLEAN_MODELS = find data/models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_RESULTS = find data/results -type f ! -name .gitignore -delete 2>/dev/null; true
CLEAN_BIGRAM_MODELS = find data/bigram_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_BIGRAM_RESULTS = find data/bigram_results -type f ! -name .gitignore -delete 2>/dev/null; true
CLEAN_WORD_MODELS = find data/word_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_WORD_RESULTS = find data/word_results -type f ! -name .gitignore -delete 2>/dev/null; true

# Pipeline
generate:
	$(RUN) generate --letters $(LETTERS) --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/letters --fonts $(FONTS)

generate-test:
	$(RUN) generate_test --letters $(LETTERS) --output_dir data/test --fonts $(FONTS)

train:
	@$(CLEAN_MODELS)
	$(RUN) train --data_dir data/letters --epochs $(EPOCHS) --save_dir data/models --checkpoint_interval $(CKPT) --n_glimpses 10 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) --diversity_vy $(VY)

resume:
	$(RUN) train --data_dir data/letters --epochs $(EPOCHS) --save_dir data/models --checkpoint_interval $(CKPT) --n_glimpses 10 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) --diversity_vy $(VY) --resume data/models/$(RESUME_FROM)

test:
	@$(CLEAN_RESULTS)
	$(RUN) test --model_dir data/models --test_data_dir data/test --output_dir data/results --device $(DEVICE)

visualize:
	$(RUN) visualize --model_dir data/models --data_dir data/letters --output_dir data/visualizations

atlas:
	@rm -f data/atlas.html
	$(RUN) atlas --model_dir data/models --test_data_dir data/test --output data/atlas.html --device $(DEVICE)

check-attention:
	$(RUN) check_attention --data_dir data/letters --device $(DEVICE)

# Archive a trained model: make archive NAME=v1-single-font
archive:
ifndef NAME
	$(error Usage: make archive NAME=v1-single-font)
endif
	@mkdir -p runs/$(NAME)
	$(RUN) compress_model --input data/models/model_final.pth --output data/models/_archive.pth.gz
	mv data/models/_archive.pth.gz runs/$(NAME)/model_final.pth.gz
	cp data/models/training_metrics.png runs/$(NAME)/ 2>/dev/null || true
	cp data/models/training.log runs/$(NAME)/ 2>/dev/null || true
	cp data/atlas.html runs/$(NAME)/ 2>/dev/null || true
	@echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')" > runs/$(NAME)/info.txt
	@echo "date: $$(date -Iseconds)" >> runs/$(NAME)/info.txt
	@echo "fonts: $(FONTS)" >> runs/$(NAME)/info.txt
	@echo "letters: $(LETTERS)" >> runs/$(NAME)/info.txt
	@echo "epochs: $(EPOCHS)" >> runs/$(NAME)/info.txt
	@echo "batch: $(BATCH)" >> runs/$(NAME)/info.txt
	@echo "guide: $(GUIDE)" >> runs/$(NAME)/info.txt
	@echo "make: make train EPOCHS=$(EPOCHS) DEVICE=$(DEVICE) BATCH=$(BATCH) GUIDE=$(GUIDE) VY=$(VY) FONTS=$(FONTS)" >> runs/$(NAME)/info.txt
	@echo "cli: python vision_training.py train --data_dir data/letters --epochs $(EPOCHS) --save_dir data/models --checkpoint_interval $(CKPT) --n_glimpses 10 --patch_size 12 --n_scales 1 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) --diversity_weight 1.0 --diversity_sigma 0.1 --recode_weight 1.0 --blur_sigma_ratio 0.16 --diversity_vy $(VY)" >> runs/$(NAME)/info.txt
	@echo "Archived to runs/$(NAME)/"

# Bigram pipeline (separate data dirs from single-letter)
generate-bigrams:
	$(RUN) generate_bigrams --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/bigrams --fonts $(FONTS)

generate-bigrams-test:
	$(RUN) generate_bigrams_test --output_dir data/bigram_test --fonts $(FONTS)

train-bigrams:
	@$(CLEAN_BIGRAM_MODELS)
	$(RUN) train_bigrams --data_dir data/bigrams --epochs $(EPOCHS) --save_dir data/bigram_models --checkpoint_interval $(CKPT) --n_scan_glimpses $(SCAN_GLIMPSES) --n_read_glimpses $(READ_GLIMPSES) --scan_patch_size $(SCAN_PATCH) --read_patch_size $(READ_PATCH) --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) $(if $(SCAN_GUIDE),--scan_guide_weight $(SCAN_GUIDE)) --scaffold_ratio $(SCAFFOLD_RATIO) --scaffold_floor $(SCAFFOLD_FLOOR) --mask_weight $(MASK) --scan_vy $(SCAN_VY) --read_vy $(READ_VY) --edge_weight $(EDGE) $(if $(TRANSFER),--transfer $(TRANSFER))

resume-bigrams:
	$(RUN) train_bigrams --data_dir data/bigrams --epochs $(EPOCHS) --save_dir data/bigram_models --checkpoint_interval $(CKPT) --n_scan_glimpses $(SCAN_GLIMPSES) --n_read_glimpses $(READ_GLIMPSES) --scan_patch_size $(SCAN_PATCH) --read_patch_size $(READ_PATCH) --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) $(if $(SCAN_GUIDE),--scan_guide_weight $(SCAN_GUIDE)) --scaffold_ratio $(SCAFFOLD_RATIO) --scaffold_floor $(SCAFFOLD_FLOOR) --mask_weight $(MASK) --scan_vy $(SCAN_VY) --read_vy $(READ_VY) --edge_weight $(EDGE) --resume data/bigram_models/$(RESUME_FROM)

test-bigrams:
	@$(CLEAN_BIGRAM_RESULTS)
	$(RUN) test_bigrams --model_dir data/bigram_models --test_data_dir data/bigram_test --output_dir data/bigram_results --device $(DEVICE)

check-bigram-attention:
	$(RUN) check_bigram_attention --data_dir data/bigrams --device $(DEVICE)

bigram-atlas:
	@rm -f data/bigram_atlas.html
	$(RUN) bigram_atlas --model_dir data/bigram_models --test_data_dir data/bigram_test --output data/bigram_atlas.html --device $(DEVICE)

# Archive a trained bigram model: make archive-bigrams NAME=v4-bigram-transfer
archive-bigrams:
ifndef NAME
	$(error Usage: make archive-bigrams NAME=v4-bigram-transfer)
endif
	@mkdir -p runs/$(NAME)
	$(RUN) compress_model --input data/bigram_models/model_final.pth --output data/bigram_models/_archive.pth.gz
	mv data/bigram_models/_archive.pth.gz runs/$(NAME)/model_final.pth.gz
	cp data/bigram_models/training_metrics.png runs/$(NAME)/ 2>/dev/null || true
	cp data/bigram_models/training.log runs/$(NAME)/ 2>/dev/null || true
	cp data/bigram_atlas.html runs/$(NAME)/atlas.html 2>/dev/null || true
	@echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')" > runs/$(NAME)/info.txt
	@echo "date: $$(date -Iseconds)" >> runs/$(NAME)/info.txt
	@echo "model: bigram (two-phase)" >> runs/$(NAME)/info.txt
	@echo "fonts: $(FONTS)" >> runs/$(NAME)/info.txt
	@echo "epochs: $(EPOCHS)" >> runs/$(NAME)/info.txt
	@echo "batch: $(BATCH)" >> runs/$(NAME)/info.txt
	@echo "guide: $(GUIDE)" >> runs/$(NAME)/info.txt
	@echo "scan: $(SCAN_GLIMPSES) glimpses, patch $(SCAN_PATCH)" >> runs/$(NAME)/info.txt
	@echo "read: $(READ_GLIMPSES) glimpses, patch $(READ_PATCH)" >> runs/$(NAME)/info.txt
	@echo "scaffold_ratio: $(SCAFFOLD_RATIO)" >> runs/$(NAME)/info.txt
	@echo "make: make train-bigrams EPOCHS=$(EPOCHS) DEVICE=$(DEVICE) BATCH=$(BATCH) GUIDE=$(GUIDE) SCAFFOLD_RATIO=$(SCAFFOLD_RATIO) SCAFFOLD_FLOOR=$(SCAFFOLD_FLOOR) SCAN_VY=$(SCAN_VY) READ_VY=$(READ_VY) MASK=$(MASK) EDGE=$(EDGE) FONTS=$(FONTS)$(if $(SCAN_GUIDE), SCAN_GUIDE=$(SCAN_GUIDE))$(if $(TRANSFER), TRANSFER=$(TRANSFER))" >> runs/$(NAME)/info.txt
	@echo "cli: python vision_training.py train_bigrams --data_dir data/bigrams --epochs $(EPOCHS) --save_dir data/bigram_models --checkpoint_interval $(CKPT) --n_scan_glimpses $(SCAN_GLIMPSES) --n_read_glimpses $(READ_GLIMPSES) --scan_patch_size $(SCAN_PATCH) --read_patch_size $(READ_PATCH) --n_scales 1 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) --scaffold_ratio $(SCAFFOLD_RATIO) --scaffold_floor $(SCAFFOLD_FLOOR) --mask_weight $(MASK) --scan_vy $(SCAN_VY) --read_vy $(READ_VY) --edge_weight $(EDGE) --blur_sigma_ratio 0.16 --diversity_weight 1.0 --diversity_sigma 0.1$(if $(SCAN_GUIDE), --scan_guide_weight $(SCAN_GUIDE))$(if $(TRANSFER), --transfer $(TRANSFER))" >> runs/$(NAME)/info.txt
	@echo "Archived to runs/$(NAME)/"

# Word pipeline (4-letter words, 256x128 canvas, prescribed x-scan)
generate-words:
	$(RUN) generate_words --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/words --fonts $(FONTS)

generate-words-test:
	$(RUN) generate_words_test --output_dir data/word_test --fonts $(FONTS)

train-words:
	@$(CLEAN_WORD_MODELS)
	$(RUN) train_words --data_dir data/words --epochs $(EPOCHS) --save_dir data/word_models --checkpoint_interval $(CKPT) --n_scan_glimpses $(WORD_SCAN_GLIMPSES) --n_read_glimpses $(WORD_READ_GLIMPSES) --scan_patch_size $(SCAN_PATCH) --read_patch_size $(READ_PATCH) --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) $(if $(SCAN_GUIDE),--scan_guide_weight $(SCAN_GUIDE)) --scaffold_ratio $(SCAFFOLD_RATIO) --scaffold_floor $(SCAFFOLD_FLOOR) $(if $(SCAFFOLD_EPOCHS),--scaffold_epochs $(SCAFFOLD_EPOCHS)) --content_weight $(CONTENT) --isolation_weight $(ISOLATION) --scan_vy $(SCAN_VY) --read_vy $(READ_VY) --edge_weight $(EDGE) $(if $(TRANSFER),--transfer $(TRANSFER)) $(if $(ISOLATION_DATA),--isolation_data_dir $(ISOLATION_DATA)) --isolation_random_prob $(ISO_RANDOM) $(if $(MULTI_HEAD),--multi_head)

resume-words:
	$(RUN) train_words --data_dir data/words --epochs $(EPOCHS) --save_dir data/word_models --checkpoint_interval $(CKPT) --n_scan_glimpses $(WORD_SCAN_GLIMPSES) --n_read_glimpses $(WORD_READ_GLIMPSES) --scan_patch_size $(SCAN_PATCH) --read_patch_size $(READ_PATCH) --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) $(if $(SCAN_GUIDE),--scan_guide_weight $(SCAN_GUIDE)) --scaffold_ratio $(SCAFFOLD_RATIO) --scaffold_floor $(SCAFFOLD_FLOOR) $(if $(SCAFFOLD_EPOCHS),--scaffold_epochs $(SCAFFOLD_EPOCHS)) --content_weight $(CONTENT) --isolation_weight $(ISOLATION) --scan_vy $(SCAN_VY) --read_vy $(READ_VY) --edge_weight $(EDGE) --resume data/word_models/$(RESUME_FROM) $(if $(ISOLATION_DATA),--isolation_data_dir $(ISOLATION_DATA)) --isolation_random_prob $(ISO_RANDOM) $(if $(MULTI_HEAD),--multi_head)

test-words:
	@$(CLEAN_WORD_RESULTS)
	$(RUN) test_words --model_dir data/word_models --test_data_dir data/word_test --output_dir data/word_results --device $(DEVICE)

word-atlas:
	@rm -f data/word_atlas.html
	$(RUN) word_atlas --model_dir data/word_models --test_data_dir data/word_test --output data/word_atlas.html --device $(DEVICE)

# Archive a trained word model: make archive-words NAME=v1-word-prescribed
archive-words:
ifndef NAME
	$(error Usage: make archive-words NAME=v1-word-prescribed)
endif
	@mkdir -p runs/$(NAME)
	$(RUN) compress_model --input data/word_models/model_final.pth --output data/word_models/_archive.pth.gz
	mv data/word_models/_archive.pth.gz runs/$(NAME)/model_final.pth.gz
	cp data/word_models/training_metrics.png runs/$(NAME)/ 2>/dev/null || true
	cp data/word_models/training.log runs/$(NAME)/ 2>/dev/null || true
	cp data/word_atlas.html runs/$(NAME)/atlas.html 2>/dev/null || true
	@echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')" > runs/$(NAME)/info.txt
	@echo "date: $$(date -Iseconds)" >> runs/$(NAME)/info.txt
	@echo "model: word (prescribed x-scan)" >> runs/$(NAME)/info.txt
	@echo "fonts: $(FONTS)" >> runs/$(NAME)/info.txt
	@echo "epochs: $(EPOCHS)" >> runs/$(NAME)/info.txt
	@echo "batch: $(BATCH)" >> runs/$(NAME)/info.txt
	@echo "guide: $(GUIDE)" >> runs/$(NAME)/info.txt
	@echo "scan: $(WORD_SCAN_GLIMPSES) glimpses (prescribed x), patch $(SCAN_PATCH)" >> runs/$(NAME)/info.txt
	@echo "read: $(WORD_READ_GLIMPSES) glimpses (free), patch $(READ_PATCH)" >> runs/$(NAME)/info.txt
	@echo "scaffold_ratio: $(SCAFFOLD_RATIO)" >> runs/$(NAME)/info.txt
	@echo "content_weight: $(CONTENT)" >> runs/$(NAME)/info.txt
	@echo "make: make train-words EPOCHS=$(EPOCHS) DEVICE=$(DEVICE) BATCH=$(BATCH) GUIDE=$(GUIDE) SCAFFOLD_RATIO=$(SCAFFOLD_RATIO) SCAFFOLD_FLOOR=$(SCAFFOLD_FLOOR) CONTENT=$(CONTENT) ISOLATION=$(ISOLATION) SCAN_VY=$(SCAN_VY) READ_VY=$(READ_VY) EDGE=$(EDGE) FONTS=$(FONTS) ISO_RANDOM=$(ISO_RANDOM)$(if $(SCAN_GUIDE), SCAN_GUIDE=$(SCAN_GUIDE))$(if $(TRANSFER), TRANSFER=$(TRANSFER))$(if $(ISOLATION_DATA), ISOLATION_DATA=$(ISOLATION_DATA))$(if $(MULTI_HEAD), MULTI_HEAD=$(MULTI_HEAD))" >> runs/$(NAME)/info.txt
	@echo "cli: python vision_training.py train_words --data_dir data/words --epochs $(EPOCHS) --save_dir data/word_models --checkpoint_interval $(CKPT) --n_scan_glimpses $(WORD_SCAN_GLIMPSES) --n_read_glimpses $(WORD_READ_GLIMPSES) --scan_patch_size $(SCAN_PATCH) --read_patch_size $(READ_PATCH) --n_scales 1 --n_positions 4 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) --scaffold_ratio $(SCAFFOLD_RATIO) --scaffold_floor $(SCAFFOLD_FLOOR) --content_weight $(CONTENT) --isolation_weight $(ISOLATION) --scan_vy $(SCAN_VY) --read_vy $(READ_VY) --edge_weight $(EDGE) --blur_sigma_ratio 0.16 --diversity_weight 1.0 --diversity_sigma 0.1 --isolation_random_prob $(ISO_RANDOM)$(if $(SCAN_GUIDE), --scan_guide_weight $(SCAN_GUIDE))$(if $(TRANSFER), --transfer $(TRANSFER))$(if $(ISOLATION_DATA), --isolation_data_dir $(ISOLATION_DATA))$(if $(MULTI_HEAD), --multi_head)" >> runs/$(NAME)/info.txt
	@echo "Archived to runs/$(NAME)/"

.PHONY: build up down restart logs shell generate generate-test train resume test visualize atlas check-attention archive \
       generate-bigrams generate-bigrams-test train-bigrams resume-bigrams test-bigrams check-bigram-attention \
       bigram-atlas archive-bigrams \
       generate-words generate-words-test train-words resume-words test-words word-atlas archive-words
