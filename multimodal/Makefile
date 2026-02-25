SERVICE = fbrl
RUN     = docker compose exec $(SERVICE) python multimodal_training.py

# Lifecycle
cache:
	docker buildx build --output type=cacheonly .

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

# Pipeline commands
generate-clean:
	$(RUN) generate_clean --letters A-Z --output_dir data/clean_video --use_real_tts

generate:
	$(RUN) generate --letters A-Z --noise_level 0.01 --num_variants 20 --output_dir data/letters --clean_video_dir data/clean_video

train:
	$(RUN) train --data_dir data/letters --epochs 500 --save_dir data/models --checkpoint_interval 100 --n_glimpses 10 --device cpu

train-fast:
	$(RUN) train --data_dir data/letters --epochs 100 --save_dir data/models --checkpoint_interval 50 --n_glimpses 10 --device cpu

train-gpu:
	$(RUN) train --data_dir data/letters --epochs 500 --save_dir data/models --checkpoint_interval 100 --n_glimpses 10 --device cuda

train-gpu-fast:
	$(RUN) train --data_dir data/letters --epochs 100 --save_dir data/models --checkpoint_interval 50 --n_glimpses 10 --device cuda

resume-gpu:
	$(RUN) train --data_dir data/letters --epochs 400 --save_dir data/models --checkpoint_interval 100 --n_glimpses 10 --device cuda --resume data/models/model_final.pth

generate-test:
	$(RUN) generate_test --clean_video_dir data/clean_video --output_dir data/test_videos

test:
	$(RUN) test --model_dir data/models --test_data_dir data/test_videos --output_dir data/results --n_glimpses 10 --device cpu

test-gpu:
	$(RUN) test --model_dir data/models --test_data_dir data/test_videos --output_dir data/results --n_glimpses 10 --device cuda

visualize:
	$(RUN) visualize --model_dir data/models --data_dir data/letters --output_dir data/visualizations --n_glimpses 10

.PHONY: cache build up down restart logs shell generate-clean generate generate-test train train-fast train-gpu train-gpu-fast resume-gpu test test-gpu visualize
