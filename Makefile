# fbrl development commands
#
# All commands run inside the Docker container via docker compose.
# Reuses goDl's Docker image (Go + libtorch + CUDA).
# Both fbrl and goDl are mounted at /workspace/{fbrl,goDl}.

COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm dev

.PHONY: image test test-cpu test-race shell vet clean kill

# Build the Docker image (reuses goDl's Dockerfile)
image:
	$(COMPOSE) build

# Run all tests (CPU + CUDA if available)
test: image
	$(RUN) go test -tags cuda -v ./...

# Run tests without CUDA
test-cpu: image
	$(RUN) go test -v ./...

# Run tests with race detector
test-race: image
	$(RUN) go test -tags cuda -v -race ./...

# Interactive shell
shell: image
	$(COMPOSE) run --rm dev bash

# Go vet
vet: image
	$(RUN) go vet -tags cuda ./...

# Train letter model
# Usage: make train-letter DATA=path/to/data [EPOCHS=100] [BATCH=32] [LR=0.001] [SAVE=letter/training] [PROFILE=1]
SAVE ?= letter/training
train-letter: image
	$(RUN) bash -c "go build -tags cuda -o /tmp/letter-train ./letter/cmd && /tmp/letter-train $(if $(DATA),--data $(DATA)) $(if $(SYNTHETIC),--synthetic $(SYNTHETIC)) --save $(SAVE) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch-size $(BATCH)) $(if $(LR),--lr $(LR)) $(if $(PROFILE),--profile)"

# Kill running training containers
kill:
	@docker ps -q --filter "name=fbrl-dev-run" | xargs -r docker stop

# Clean up containers and volumes
clean:
	$(COMPOSE) down -v --rmi local
