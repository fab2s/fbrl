SERVICE = fbrl
RUN     = docker compose exec $(SERVICE) python vision_training.py

# Overridable defaults (e.g. make train-gpu EPOCHS=500 DEVICE=cuda)
EPOCHS   ?= 200
CKPT     ?= 100
DEVICE   ?= cpu
LETTERS  ?= Aa-Zz
VARIANTS ?= 20
NOISE    ?= 0.1

# Lifecycle
build:
	docker compose build

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
	$(RUN) generate --letters $(LETTERS) --num_variants $(VARIANTS) --noise_level $(NOISE) --output_dir data/letters

generate-test:
	$(RUN) generate_test --letters $(LETTERS) --output_dir data/test

train:
	$(RUN) train --data_dir data/letters --epochs $(EPOCHS) --save_dir data/models --checkpoint_interval $(CKPT) --n_glimpses 10 --device $(DEVICE)

resume:
	$(RUN) train --data_dir data/letters --epochs $(EPOCHS) --save_dir data/models --checkpoint_interval $(CKPT) --n_glimpses 10 --device $(DEVICE) --resume data/models/model_final.pth

test:
	$(RUN) test --model_dir data/models --test_data_dir data/test --output_dir data/results --device $(DEVICE)

visualize:
	$(RUN) visualize --model_dir data/models --data_dir data/letters --output_dir data/visualizations

check-attention:
	$(RUN) check_attention --data_dir data/letters --device $(DEVICE)

.PHONY: build up down restart logs shell generate generate-test train resume test visualize check-attention
