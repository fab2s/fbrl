SERVICE = fbrl
RUN     = docker compose exec $(SERVICE) python vision_training.py

# Runtime overrides (e.g. make train-words EPOCHS=500 DEVICE=cuda)
DEVICE   ?= auto
EPOCHS   ?=
BATCH    ?=
TRANSFER ?=
RESUME_FROM ?= model_final.pth

# Generation defaults
LETTERS  ?= Aa-Zz
VARIANTS ?= 20
NOISE    ?= 0.1
FONTS    ?= all
CKPT     ?=

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

# Pipeline — Generation (unchanged)
generate:
	$(RUN) generate --letters $(LETTERS) --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/letters --fonts $(FONTS)

generate-test:
	$(RUN) generate_test --letters $(LETTERS) --output_dir data/test --fonts $(FONTS)

# Config-based training commands
# Override any config value via CLI: make train EPOCHS=500 BATCH=64
CONFIG ?= configs/letter.yaml

train:
	@$(CLEAN_MODELS)
	$(RUN) train --config $(CONFIG) --device $(DEVICE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH)) $(if $(TRANSFER),--transfer $(TRANSFER)) $(if $(CKPT),--checkpoint_interval $(CKPT))

resume:
	$(RUN) train --config $(CONFIG) --device $(DEVICE) --resume data/models/$(RESUME_FROM) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH))

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
	cp $(CONFIG) runs/$(NAME)/config.yaml 2>/dev/null || true
	@echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')" > runs/$(NAME)/info.txt
	@echo "date: $$(date -Iseconds)" >> runs/$(NAME)/info.txt
	@echo "config: $(CONFIG)" >> runs/$(NAME)/info.txt
	@echo "Archived to runs/$(NAME)/"

# Bigram pipeline
generate-bigrams:
	$(RUN) generate_bigrams --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/bigrams --fonts $(FONTS)

generate-bigrams-test:
	$(RUN) generate_bigrams_test --output_dir data/bigram_test --fonts $(FONTS)

BIGRAM_CONFIG ?= configs/bigram.yaml

train-bigrams:
	@$(CLEAN_BIGRAM_MODELS)
	$(RUN) train_bigrams --config $(BIGRAM_CONFIG) --device $(DEVICE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH)) $(if $(TRANSFER),--transfer $(TRANSFER)) $(if $(CKPT),--checkpoint_interval $(CKPT))

resume-bigrams:
	$(RUN) train_bigrams --config $(BIGRAM_CONFIG) --device $(DEVICE) --resume data/bigram_models/$(RESUME_FROM) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH))

test-bigrams:
	@$(CLEAN_BIGRAM_RESULTS)
	$(RUN) test_bigrams --model_dir data/bigram_models --test_data_dir data/bigram_test --output_dir data/bigram_results --device $(DEVICE)

check-bigram-attention:
	$(RUN) check_bigram_attention --data_dir data/bigrams --device $(DEVICE)

bigram-atlas:
	@rm -f data/bigram_atlas.html
	$(RUN) bigram_atlas --model_dir data/bigram_models --test_data_dir data/bigram_test --output data/bigram_atlas.html --device $(DEVICE)

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
	cp $(BIGRAM_CONFIG) runs/$(NAME)/config.yaml 2>/dev/null || true
	@echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')" > runs/$(NAME)/info.txt
	@echo "date: $$(date -Iseconds)" >> runs/$(NAME)/info.txt
	@echo "config: $(BIGRAM_CONFIG)" >> runs/$(NAME)/info.txt
	@echo "Archived to runs/$(NAME)/"

# Word pipeline
generate-words:
	$(RUN) generate_words --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/words --fonts $(FONTS)

generate-words-test:
	$(RUN) generate_words_test --output_dir data/word_test --fonts $(FONTS)

WORD_CONFIG ?= configs/word.yaml

train-words:
	@$(CLEAN_WORD_MODELS)
	$(RUN) train_words --config $(WORD_CONFIG) --device $(DEVICE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH)) $(if $(TRANSFER),--transfer $(TRANSFER)) $(if $(CKPT),--checkpoint_interval $(CKPT))

resume-words:
	$(RUN) train_words --config $(WORD_CONFIG) --device $(DEVICE) --resume data/word_models/$(RESUME_FROM) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH))

test-words:
	@$(CLEAN_WORD_RESULTS)
	$(RUN) test_words --model_dir data/word_models --test_data_dir data/word_test --output_dir data/word_results --device $(DEVICE)

test-word-isolation:
	$(RUN) test_word_isolation --model_dir data/word_models --test_data_dir data/test --output_dir data/word_results --device $(DEVICE)

word-atlas:
	@rm -f data/word_atlas.html
	$(RUN) word_atlas --model_dir data/word_models --test_data_dir data/word_test --output data/word_atlas.html --device $(DEVICE)

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
	cp $(WORD_CONFIG) runs/$(NAME)/config.yaml 2>/dev/null || true
	@echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')" > runs/$(NAME)/info.txt
	@echo "date: $$(date -Iseconds)" >> runs/$(NAME)/info.txt
	@echo "config: $(WORD_CONFIG)" >> runs/$(NAME)/info.txt
	@echo "Archived to runs/$(NAME)/"

# Motor pipeline
MOTOR_CONFIG ?= configs/letter_motor.yaml
CLEAN_MOTOR_MODELS = find data/motor_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_MOTOR_RESULTS = find data/motor_results -type f ! -name .gitignore -delete 2>/dev/null; true

generate-trajectories:
	$(RUN) generate_trajectories --output_dir data/trajectories --font dejavu-sans --letters $(LETTERS) --n_points 32

train-motor:
	@$(CLEAN_MOTOR_MODELS)
	$(RUN) train_motor --config $(MOTOR_CONFIG) --device $(DEVICE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH)) $(if $(CKPT),--checkpoint_interval $(CKPT))

resume-motor:
	$(RUN) train_motor --config $(MOTOR_CONFIG) --device $(DEVICE) --resume data/motor_models/$(RESUME_FROM) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH))

test-motor:
	@$(CLEAN_MOTOR_RESULTS)
	$(RUN) test_motor --model_dir data/motor_models --test_data_dir data/test --output_dir data/motor_results --device $(DEVICE)

motor-atlas:
	@rm -f data/motor_atlas.html
	$(RUN) motor_atlas --model_dir data/motor_models --test_data_dir data/test --output data/motor_atlas.html --device $(DEVICE)

archive-motor:
ifndef NAME
	$(error Usage: make archive-motor NAME=v1-motor-baseline)
endif
	@mkdir -p runs/$(NAME)
	$(RUN) compress_model --input data/motor_models/model_final.pth --output data/motor_models/_archive.pth.gz
	mv data/motor_models/_archive.pth.gz runs/$(NAME)/model_final.pth.gz
	cp data/motor_models/training_metrics.png runs/$(NAME)/ 2>/dev/null || true
	cp data/motor_models/training.log runs/$(NAME)/ 2>/dev/null || true
	cp data/motor_atlas.html runs/$(NAME)/atlas.html 2>/dev/null || true
	cp $(MOTOR_CONFIG) runs/$(NAME)/config.yaml 2>/dev/null || true
	@echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')" > runs/$(NAME)/info.txt
	@echo "date: $$(date -Iseconds)" >> runs/$(NAME)/info.txt
	@echo "config: $(MOTOR_CONFIG)" >> runs/$(NAME)/info.txt
	@echo "Archived to runs/$(NAME)/"

.PHONY: build up down restart logs shell generate generate-test train resume test visualize atlas check-attention archive \
       generate-bigrams generate-bigrams-test train-bigrams resume-bigrams test-bigrams check-bigram-attention \
       bigram-atlas archive-bigrams \
       generate-words generate-words-test train-words resume-words test-words test-word-isolation word-atlas archive-words \
       generate-trajectories train-motor resume-motor test-motor motor-atlas archive-motor
