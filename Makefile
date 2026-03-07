SERVICE = python
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
CASE_FILTER ?=

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
CLEAN_MODELS = find python/data/letter_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_RESULTS = find python/data/letter_results -type f ! -name .gitignore -delete 2>/dev/null; true
CLEAN_BIGRAM_MODELS = find python/data/bigram_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_BIGRAM_RESULTS = find python/data/bigram_results -type f ! -name .gitignore -delete 2>/dev/null; true
CLEAN_WORD_MODELS = find python/data/word_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_WORD_RESULTS = find python/data/word_results -type f ! -name .gitignore -delete 2>/dev/null; true

# Pipeline — Generation (unchanged)
generate:
	$(RUN) generate --letters $(LETTERS) --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/letters --fonts $(FONTS)

generate-test:
	$(RUN) generate_test --letters $(LETTERS) --output_dir data/letter_test --fonts $(FONTS)

# Config-based training commands
# Override any config value via CLI: make train EPOCHS=500 BATCH=64
CONFIG ?= configs/letter.yaml

train:
	@$(CLEAN_MODELS)
	$(RUN) train --config $(CONFIG) --device $(DEVICE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH)) $(if $(TRANSFER),--transfer $(TRANSFER)) $(if $(CKPT),--checkpoint_interval $(CKPT))

resume:
	$(RUN) train --config $(CONFIG) --device $(DEVICE) --resume data/letter_models/$(RESUME_FROM) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH))

test:
	@$(CLEAN_RESULTS)
	$(RUN) test --model_dir data/letter_models --test_data_dir data/letter_test --output_dir data/letter_results --device $(DEVICE)

visualize:
	$(RUN) visualize --model_dir data/letter_models --data_dir data/letters --output_dir data/letter_visualizations

atlas:
	@rm -f python/data/letter_atlas.html
	$(RUN) atlas --model_dir data/letter_models --test_data_dir data/letter_test --output data/letter_atlas.html --device $(DEVICE)

check-attention:
	$(RUN) check_attention --data_dir data/letters --device $(DEVICE)

# Archive a trained model: make archive NAME=v1-single-font
archive:
ifndef NAME
	$(error Usage: make archive NAME=v1-single-font)
endif
	@mkdir -p python/runs/$(NAME)
	$(RUN) compress_model --input data/letter_models/model_final.pth --output data/letter_models/_archive.pth.gz
	mv python/data/letter_models/_archive.pth.gz python/runs/$(NAME)/model_final.pth.gz
	cp python/data/letter_models/training_metrics.png python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/letter_models/training.log python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/letter_atlas.html python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/letter_models/config.yaml python/runs/$(NAME)/ 2>/dev/null || cp python/$(CONFIG) python/runs/$(NAME)/config.yaml 2>/dev/null || true
	cp python/data/letter_models/info.txt python/runs/$(NAME)/ 2>/dev/null || { echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')\ndate: $$(date -Iseconds)\nconfig: $(CONFIG)" > python/runs/$(NAME)/info.txt; }
	@echo "Archived to python/runs/$(NAME)/"

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
	@rm -f python/data/bigram_atlas.html
	$(RUN) bigram_atlas --model_dir data/bigram_models --test_data_dir data/bigram_test --output data/bigram_atlas.html --device $(DEVICE)

archive-bigrams:
ifndef NAME
	$(error Usage: make archive-bigrams NAME=v4-bigram-transfer)
endif
	@mkdir -p python/runs/$(NAME)
	$(RUN) compress_model --input data/bigram_models/model_final.pth --output data/bigram_models/_archive.pth.gz
	mv python/data/bigram_models/_archive.pth.gz python/runs/$(NAME)/model_final.pth.gz
	cp python/data/bigram_models/training_metrics.png python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/bigram_models/training.log python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/bigram_atlas.html python/runs/$(NAME)/atlas.html 2>/dev/null || true
	cp python/data/bigram_models/config.yaml python/runs/$(NAME)/ 2>/dev/null || cp python/$(BIGRAM_CONFIG) python/runs/$(NAME)/config.yaml 2>/dev/null || true
	cp python/data/bigram_models/info.txt python/runs/$(NAME)/ 2>/dev/null || { echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')\ndate: $$(date -Iseconds)\nconfig: $(BIGRAM_CONFIG)" > python/runs/$(NAME)/info.txt; }
	@echo "Archived to python/runs/$(NAME)/"

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
	$(RUN) test_word_isolation --model_dir data/word_models --test_data_dir data/letter_test --output_dir data/word_results --device $(DEVICE)

word-atlas:
	@rm -f python/data/word_atlas.html
	$(RUN) word_atlas --model_dir data/word_models --test_data_dir data/word_test --output data/word_atlas.html --device $(DEVICE)

isolation-atlas:
	@rm -f python/data/isolation_atlas.html
	$(RUN) isolation_atlas --model_dir data/word_models --test_data_dir data/letter_test --output data/isolation_atlas.html --device $(DEVICE)

archive-words:
ifndef NAME
	$(error Usage: make archive-words NAME=v1-word-prescribed)
endif
	@mkdir -p python/runs/$(NAME)
	$(RUN) compress_model --input data/word_models/model_final.pth --output data/word_models/_archive.pth.gz
	mv python/data/word_models/_archive.pth.gz python/runs/$(NAME)/model_final.pth.gz
	cp python/data/word_models/training_metrics.png python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/word_models/training.log python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/word_atlas.html python/runs/$(NAME)/atlas.html 2>/dev/null || true
	cp python/data/isolation_atlas.html python/runs/$(NAME)/isolation_atlas.html 2>/dev/null || true
	cp python/data/word_models/config.yaml python/runs/$(NAME)/ 2>/dev/null || cp python/$(WORD_CONFIG) python/runs/$(NAME)/config.yaml 2>/dev/null || true
	cp python/data/word_models/info.txt python/runs/$(NAME)/ 2>/dev/null || { echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')\ndate: $$(date -Iseconds)\nconfig: $(WORD_CONFIG)" > python/runs/$(NAME)/info.txt; }
	@echo "Archived to python/runs/$(NAME)/"

# Motor pipeline
MOTOR_CONFIG ?= configs/letter_motor.yaml
CLEAN_MOTOR_MODELS = find python/data/motor_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_MOTOR_RESULTS = find python/data/motor_results -type f ! -name .gitignore -delete 2>/dev/null; true

generate-trajectories:
	$(RUN) generate_trajectories --output_dir data/trajectories --font dejavu-sans --letters $(LETTERS) --n_points 48

train-motor:
	@$(CLEAN_MOTOR_MODELS)
	$(RUN) train_motor --config $(MOTOR_CONFIG) --device $(DEVICE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH)) $(if $(TRANSFER),--transfer $(TRANSFER)) $(if $(CKPT),--checkpoint_interval $(CKPT)) $(if $(CASE_FILTER),--case_filter $(CASE_FILTER))

resume-motor:
	$(RUN) train_motor --config $(MOTOR_CONFIG) --device $(DEVICE) --resume data/motor_models/$(RESUME_FROM) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH))

test-motor:
	@$(CLEAN_MOTOR_RESULTS)
	$(RUN) test_motor --model_dir data/motor_models --test_data_dir data/letter_test --output_dir data/motor_results --device $(DEVICE)

motor-atlas:
	@rm -f python/data/motor_atlas.html
	$(RUN) motor_atlas --model_dir data/motor_models --test_data_dir data/letter_test --output data/motor_atlas.html --device $(DEVICE)

archive-motor:
ifndef NAME
	$(error Usage: make archive-motor NAME=v1-motor-baseline)
endif
	@mkdir -p python/runs/$(NAME)
	$(RUN) compress_model --input data/motor_models/model_final.pth --output data/motor_models/_archive.pth.gz
	mv python/data/motor_models/_archive.pth.gz python/runs/$(NAME)/model_final.pth.gz
	cp python/data/motor_models/training_metrics.png python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/motor_models/training.log python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/motor_atlas.html python/runs/$(NAME)/atlas.html 2>/dev/null || true
	cp python/data/motor_models/config.yaml python/runs/$(NAME)/ 2>/dev/null || cp python/$(MOTOR_CONFIG) python/runs/$(NAME)/config.yaml 2>/dev/null || true
	cp python/data/motor_models/info.txt python/runs/$(NAME)/ 2>/dev/null || { echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')\ndate: $$(date -Iseconds)\nconfig: $(MOTOR_CONFIG)" > python/runs/$(NAME)/info.txt; }
	@echo "Archived to python/runs/$(NAME)/"

# Counting pipeline
COUNTING_CONFIG ?= configs/counting.yaml
CLEAN_COUNTING_MODELS = find python/data/counting_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_COUNTING_RESULTS = find python/data/counting_results -type f ! -name .gitignore -delete 2>/dev/null; true

generate-counting:
	$(RUN) generate_counting --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/counting --fonts $(FONTS)

generate-counting-test:
	$(RUN) generate_counting_test --output_dir data/counting_test --fonts $(FONTS)

train-counting:
	@$(CLEAN_COUNTING_MODELS)
	$(RUN) train_counting --config $(COUNTING_CONFIG) --device $(DEVICE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH)) $(if $(CKPT),--checkpoint_interval $(CKPT))

resume-counting:
	$(RUN) train_counting --config $(COUNTING_CONFIG) --device $(DEVICE) --resume data/counting_models/$(RESUME_FROM) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH))

test-counting:
	@$(CLEAN_COUNTING_RESULTS)
	$(RUN) test_counting --model_dir data/counting_models --test_data_dir data/counting_test --output_dir data/counting_results --device $(DEVICE)

counting-atlas:
	@rm -f python/data/counting_atlas.html
	$(RUN) counting_atlas --model_dir data/counting_models --test_data_dir data/counting_test --output data/counting_atlas.html --device $(DEVICE)

archive-counting:
ifndef NAME
	$(error Usage: make archive-counting NAME=v1-counting-baseline)
endif
	@mkdir -p python/runs/$(NAME)
	$(RUN) compress_model --input data/counting_models/model_final.pth --output data/counting_models/_archive.pth.gz
	mv python/data/counting_models/_archive.pth.gz python/runs/$(NAME)/model_final.pth.gz
	cp python/data/counting_models/training_metrics.png python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/counting_models/training.log python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/counting_atlas.html python/runs/$(NAME)/atlas.html 2>/dev/null || true
	cp python/data/counting_models/config.yaml python/runs/$(NAME)/ 2>/dev/null || cp python/$(COUNTING_CONFIG) python/runs/$(NAME)/config.yaml 2>/dev/null || true
	cp python/data/counting_models/info.txt python/runs/$(NAME)/ 2>/dev/null || { echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')\ndate: $$(date -Iseconds)\nconfig: $(COUNTING_CONFIG)" > python/runs/$(NAME)/info.txt; }
	@echo "Archived to python/runs/$(NAME)/"

# Reading pipeline (three-phase foveal)
READING_CONFIG ?= configs/reading.yaml
CLEAN_READING_MODELS = find python/data/reading_models -type f ! -name .gitignore ! -name 'training_*.log' -delete 2>/dev/null; true
CLEAN_READING_RESULTS = find python/data/reading_results -type f ! -name .gitignore -delete 2>/dev/null; true

train-reading:
	@$(CLEAN_READING_MODELS)
	$(RUN) train_reading --config $(READING_CONFIG) --device $(DEVICE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH)) $(if $(CKPT),--checkpoint_interval $(CKPT))

resume-reading:
	$(RUN) train_reading --config $(READING_CONFIG) --device $(DEVICE) --resume data/reading_models/$(RESUME_FROM) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch_size $(BATCH))

test-reading:
	@$(CLEAN_READING_RESULTS)
	$(RUN) test_reading --model_dir data/reading_models --test_data_dir data/counting_test --output_dir data/reading_results --device $(DEVICE)

reading-atlas:
	@rm -f python/data/reading_atlas.html
	$(RUN) reading_atlas --model_dir data/reading_models --test_data_dir data/counting_test --output data/reading_atlas.html --device $(DEVICE)

archive-reading:
ifndef NAME
	$(error Usage: make archive-reading NAME=v1-reading-baseline)
endif
	@mkdir -p python/runs/$(NAME)
	$(RUN) compress_model --input data/reading_models/model_final.pth --output data/reading_models/_archive.pth.gz
	mv python/data/reading_models/_archive.pth.gz python/runs/$(NAME)/model_final.pth.gz
	cp python/data/reading_models/training_metrics.png python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/reading_models/training.log python/runs/$(NAME)/ 2>/dev/null || true
	cp python/data/reading_atlas.html python/runs/$(NAME)/atlas.html 2>/dev/null || true
	cp python/data/reading_models/config.yaml python/runs/$(NAME)/ 2>/dev/null || cp python/$(READING_CONFIG) python/runs/$(NAME)/config.yaml 2>/dev/null || true
	cp python/data/reading_models/info.txt python/runs/$(NAME)/ 2>/dev/null || { echo "git: $$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')\ndate: $$(date -Iseconds)\nconfig: $(READING_CONFIG)" > python/runs/$(NAME)/info.txt; }
	@echo "Archived to python/runs/$(NAME)/"

test-unit:
	docker compose exec $(SERVICE) pytest tests/ -v

.PHONY: build up down restart logs shell generate generate-test train resume test visualize atlas check-attention archive \
       generate-bigrams generate-bigrams-test train-bigrams resume-bigrams test-bigrams check-bigram-attention \
       bigram-atlas archive-bigrams \
       generate-words generate-words-test train-words resume-words test-words test-word-isolation word-atlas isolation-atlas archive-words \
       generate-trajectories train-motor resume-motor test-motor motor-atlas archive-motor \
       generate-counting generate-counting-test train-counting resume-counting test-counting counting-atlas archive-counting \
       train-reading resume-reading test-reading reading-atlas archive-reading \
       test-unit
