# fbrl development commands
#
# All commands run inside the Docker container via docker compose.
# Working directory is letter/ (set in docker-compose.yml).
# The override mounts the parent so flodl path dependency resolves.

COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm dev
FEATURES ?= cuda

.PHONY: image build test test-release check clippy doc shell train-letter clean kill

# Build the Docker image
image:
	$(COMPOSE) build

# Build the project (debug)
build: image
	$(RUN) cargo build --features $(FEATURES)

# Build the project (release)
build-release: image
	$(RUN) cargo build --release --features $(FEATURES)

# Run all tests
test: image
	$(RUN) cargo test --features $(FEATURES) -- --nocapture

# Run tests in release mode
test-release: image
	$(RUN) cargo test --release --features $(FEATURES) -- --nocapture

# Type check without building
check: image
	$(RUN) cargo check --features $(FEATURES)

# Lint
clippy: image
	$(RUN) cargo clippy --features $(FEATURES) -- -W clippy::all

# Generate API docs
doc: image
	$(RUN) cargo doc --features $(FEATURES) --no-deps --document-private-items

# Train letter model
# Usage: make train-letter SYNTHETIC=64 EPOCHS=2
#        make train-letter DATA=path/to/data EPOCHS=100 SAVE=runs/v1
SAVE ?= training
train-letter: image
	$(RUN) cargo run --release --features $(FEATURES) -- $(if $(DATA),--data $(DATA)) $(if $(SYNTHETIC),--synthetic $(SYNTHETIC)) $(if $(SAVE),--save $(SAVE)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch-size $(BATCH)) $(if $(LR),--lr $(LR)) $(if $(PROFILE),--profile)

# Interactive shell
shell: image
	$(COMPOSE) run --rm dev bash

# Kill running containers
kill:
	@docker ps -q --filter "name=fbrl-dev-run" | xargs -r docker stop

# Clean up containers and volumes
clean:
	$(COMPOSE) down -v --rmi local
