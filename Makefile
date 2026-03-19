# fbrl development commands
#
# All commands run inside the Docker container via docker compose.
# Working directory is letter/ (set in docker-compose.yml).
# The override mounts the parent so flodl path dependency resolves.

COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm --service-ports dev
RUN_WORD = $(COMPOSE) run --rm --service-ports -w /workspace/fbrl/word dev
FEATURES ?= cuda

.PHONY: image build test test-release check clippy doc shell train-letter clean kill \
        build-word check-word test-word smoke-word rebuild-word train-word

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

# Force flodl recompilation (touch shim to bust cargo cache, then build)
rebuild: image
	@touch ../rdl/flodl-sys/shim.cpp 2>/dev/null || true
	$(RUN) cargo build --release --features $(FEATURES)

# Train letter model
# Usage: make train-letter SYNTHETIC=64 EPOCHS=2
#        make train-letter DATA=path/to/data EPOCHS=100 SAVE=runs/v1 MONITOR=3000
SAVE ?= training
train-letter: rebuild
	$(RUN) cargo run --release --features $(FEATURES) -- $(if $(DATA),--data $(DATA)) $(if $(SYNTHETIC),--synthetic $(SYNTHETIC)) $(if $(SAVE),--save $(SAVE)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch-size $(BATCH)) $(if $(LR),--lr $(LR)) $(if $(MONITOR),--monitor $(MONITOR))

# Evaluate trained model
# Usage: make eval-letter RUN=runs/v1 TEST=../python/data/letter_test
eval-letter: rebuild
	$(RUN) cargo run --release --features $(FEATURES) -- --eval $(RUN_DIR) --test-data $(TEST) $(if $(EVAL_SAVE),--save $(EVAL_SAVE))
RUN_DIR ?= runs/v1
TEST ?= ../python/data/letter_test

# Profile with nsys (1 epoch, captures CUDA kernels + CPU activity)
# Output: runs/profile/rust_profile.nsys-rep + stats on stderr
profile-letter: rebuild
	$(RUN) bash -c 'mkdir -p runs/profile && nsys profile \
		--stats=true \
		--force-overwrite=true \
		--trace=cuda,nvtx \
		--sample=none \
		-o runs/profile/rust_profile \
		cargo run --release --features $(FEATURES) -- \
		$(if $(DATA),--data $(DATA)) --save runs/profile --epochs 1 \
		$(if $(BATCH),--batch-size $(BATCH)) --monitor 0'

# Profile Python for comparison (1 epoch, same model)
profile-python: image
	$(RUN) bash -c 'which python3 && nsys profile \
		--stats=true \
		--force-overwrite=true \
		--cuda-memory-usage=true \
		-o runs/profile/python_profile \
		python3 -c "$$PYTHON_PROFILE_SCRIPT"'

# Interactive shell
shell: image
	$(COMPOSE) run --rm dev bash

# Kill running containers
kill:
	@docker ps -q --filter "name=fbrl-dev-run" | xargs -r docker stop

# Clean up containers and volumes
clean:
	$(COMPOSE) down -v --rmi local

# --- Word model ---

# Build word crate (debug)
build-word: image
	$(RUN_WORD) cargo build --features $(FEATURES)

# Check word crate
check-word: image
	$(RUN_WORD) cargo check --features $(FEATURES)

# Test word crate
test-word: image
	$(RUN_WORD) cargo test --features $(FEATURES) -- --nocapture

# Run word smoke test
smoke-word: image
	$(RUN_WORD) cargo run --features $(FEATURES)

# Force recompile + build word (release)
rebuild-word: image
	@touch ../rdl/flodl-sys/shim.cpp 2>/dev/null || true
	$(RUN_WORD) cargo build --release --features $(FEATURES)

# Train word model
# Usage: make train-word DATA=../python/data/words SAVE=runs/v1 MONITOR=3000
#        make train-word DATA=../python/data/words TRANSFER=../letter/runs/v2/model_final.fdl.gz
WORD_SAVE ?= training
train-word: rebuild-word
	$(RUN_WORD) cargo run --release --features $(FEATURES) -- $(if $(DATA),--data $(DATA)) $(if $(SYNTHETIC),--synthetic $(SYNTHETIC)) $(if $(WORD_SAVE),--save $(WORD_SAVE)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch-size $(BATCH)) $(if $(TRANSFER),--transfer $(TRANSFER)) $(if $(ISOLATION),--isolation-data $(ISOLATION)) $(if $(MONITOR),--monitor $(MONITOR))
