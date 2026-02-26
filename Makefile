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
SCAFFOLD ?= 200
TRANSFER ?=

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

# Pipeline
generate:
	$(RUN) generate --letters $(LETTERS) --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/letters --fonts $(FONTS)

generate-test:
	$(RUN) generate_test --letters $(LETTERS) --output_dir data/test --fonts $(FONTS)

train:
	$(RUN) train --data_dir data/letters --epochs $(EPOCHS) --save_dir data/models --checkpoint_interval $(CKPT) --n_glimpses 10 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE)

resume:
	$(RUN) train --data_dir data/letters --epochs $(EPOCHS) --save_dir data/models --checkpoint_interval $(CKPT) --n_glimpses 10 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) --resume data/models/model_final.pth

test:
	$(RUN) test --model_dir data/models --test_data_dir data/test --output_dir data/results --device $(DEVICE)

visualize:
	$(RUN) visualize --model_dir data/models --data_dir data/letters --output_dir data/visualizations

atlas:
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
	@echo "Archived to runs/$(NAME)/"

# Bigram pipeline (separate data dirs from single-letter)
generate-bigrams:
	$(RUN) generate_bigrams --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/bigrams --fonts $(FONTS)

generate-bigrams-test:
	$(RUN) generate_bigrams_test --output_dir data/bigram_test --fonts $(FONTS)

train-bigrams:
	$(RUN) train_bigrams --data_dir data/bigrams --epochs $(EPOCHS) --save_dir data/bigram_models --checkpoint_interval $(CKPT) --n_glimpses 15 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) --scaffold_epochs $(SCAFFOLD) $(if $(TRANSFER),--transfer $(TRANSFER))

resume-bigrams:
	$(RUN) train_bigrams --data_dir data/bigrams --epochs $(EPOCHS) --save_dir data/bigram_models --checkpoint_interval $(CKPT) --n_glimpses 15 --device $(DEVICE) --batch_size $(BATCH) --guide_weight $(GUIDE) --scaffold_epochs $(SCAFFOLD) --resume data/bigram_models/model_final.pth

test-bigrams:
	$(RUN) test_bigrams --model_dir data/bigram_models --test_data_dir data/bigram_test --output_dir data/bigram_results --device $(DEVICE)

check-bigram-attention:
	$(RUN) check_bigram_attention --data_dir data/bigrams --device $(DEVICE)

.PHONY: build up down restart logs shell generate generate-test train resume test visualize atlas check-attention archive \
       generate-bigrams generate-bigrams-test train-bigrams resume-bigrams test-bigrams check-bigram-attention
