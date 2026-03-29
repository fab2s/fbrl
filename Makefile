# fbrl development commands
#
# All commands run inside the Docker container via docker compose.
# Working directory is letter/ (set in docker-compose.yml).
# The override mounts the parent so flodl path dependency resolves.

COMPOSE  = docker compose
RUN      = $(COMPOSE) run --rm $(if $(MONITOR),--service-ports) dev
RUN_WORD = $(COMPOSE) run --rm $(if $(MONITOR),--service-ports) -w /workspace/fbrl/word dev
FEATURES ?= cuda
FLODL_DIR ?= ../rdl

.PHONY: image build test test-release check clippy doc shell train-letter clean kill \
        build-word check-word test-word smoke-word rebuild-word train-subscan train-word \
        eval-letter-direct

# Build the Docker image (skips if already exists)
image:
	@if ! docker image inspect fbrl-dev:latest >/dev/null 2>&1; then \
		$(COMPOSE) build; \
	fi

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

# Smart flodl rebuild: only touch shim when flodl source actually changed.
# Uses git describe --always --dirty to detect commits + uncommitted edits.
FLODL_HASH_FILE := .flodl-hash
define check_flodl
	@NEW_HASH=$$(find $(FLODL_DIR) -type f \( -name '*.rs' -o -name '*.cpp' -o -name '*.h' -o -name 'Cargo.toml' \) | sort | xargs sha256sum 2>/dev/null | sha256sum | cut -c1-16); \
	OLD_HASH=$$(cat $(FLODL_HASH_FILE) 2>/dev/null || echo "none"); \
	if [ "$$NEW_HASH" != "$$OLD_HASH" ]; then \
		echo "flodl changed ($$OLD_HASH → $$NEW_HASH), busting cache"; \
		touch $(FLODL_DIR)/flodl-sys/shim.cpp 2>/dev/null || true; \
		echo "$$NEW_HASH" > $(FLODL_HASH_FILE); \
	else \
		echo "flodl unchanged ($$OLD_HASH), skipping recompile"; \
	fi
endef

rebuild: image
	$(check_flodl)
	$(RUN) cargo build --release --features $(FEATURES)

# Train letter model
# Usage: make train-letter SYNTHETIC=64 EPOCHS=2
#        make train-letter DATA=path/to/data EPOCHS=100 SAVE=runs/v1 MONITOR=3000
SAVE ?= training
train-letter: rebuild
	$(RUN) cargo run --release --features $(FEATURES) -- $(if $(GEN),--generate $(GEN)) $(if $(GEN_SAVE),--gen-save $(GEN_SAVE)) $(if $(DATA),--data $(DATA)) $(if $(SYNTHETIC),--synthetic $(SYNTHETIC)) $(if $(SAVE),--save $(SAVE)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch-size $(BATCH)) $(if $(LR),--lr $(LR)) $(if $(MONITOR),--monitor $(MONITOR)) $(if $(LEASH),--leash-weight $(LEASH)) $(if $(LEASH_R),--leash-radius $(LEASH_R))

# Generate test data (no training)
# Usage: make gen-test GEN_CFG=gen_test_config.json GEN_OUT=runs/v2_gen/test_data
gen-test: rebuild
	$(RUN) cargo run --release --features $(FEATURES) -- --generate $(GEN_CFG) --gen-save $(GEN_OUT) --epochs 0
GEN_CFG ?= gen_test_config.json
GEN_OUT ?= test_data

# Evaluate trained model
# Usage: make eval-letter RUN_DIR=runs/v2_gen TEST=runs/v2_gen/test_data
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

# Smart recompile + build word (release)
rebuild-word: image
	$(check_flodl)
	$(RUN_WORD) cargo build --release --features $(FEATURES)

# Train SubScan (Step 2: triangle MSE — independent, no letter model)
# Usage: make train-subscan WORD_DATA=../python/data/words EPOCHS=100 MONITOR=3000
SUBSCAN_SAVE ?= training
train-subscan: rebuild-word
	$(RUN_WORD) cargo run --release --features $(FEATURES) -- train-subscan --word-data $(WORD_DATA) $(if $(SUBSCAN_SAVE),--save-dir $(SUBSCAN_SAVE)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch-size $(BATCH)) $(if $(LR),--subscan-lr $(LR)) $(if $(MONITOR),--monitor $(MONITOR))

# Eval SubScan + LetterModel composition
# Usage: make eval-subscan WORD_DATA=../python/data/words SUBSCAN_CKPT=training/subscan_final.fdl.gz LETTER_CKPT=../letter/runs/v1_relative/model_final.fdl.gz
eval-subscan: rebuild-word
	$(RUN_WORD) cargo run --release --features $(FEATURES) -- eval-subscan --word-data $(WORD_DATA) --subscan $(SUBSCAN_CKPT) --letter $(LETTER_CKPT) $(if $(NOISE_X),--noise-x $(NOISE_X)) $(if $(NOISE_Y),--noise-y $(NOISE_Y)) $(if $(EVAL_SAVE),--save-dir $(EVAL_SAVE))

# Eval letter model with GT origins (no SubScan — clean baseline)
# Usage: make eval-letter-direct WORD_DATA=test_words LETTER_CKPT=../letter/runs/v2_gen/model_final.fdl.gz
eval-letter-direct: rebuild-word
	$(RUN_WORD) cargo run --release --features $(FEATURES) -- eval-letter-direct --word-data $(WORD_DATA) --letter $(LETTER_CKPT) $(if $(EVAL_SAVE),--save-dir $(EVAL_SAVE)) $(if $(BATCH),--batch-size $(BATCH))

# Train word model (Step 3, future)
# Usage: make train-word DATA=../python/data/words SAVE=runs/v1 MONITOR=3000
WORD_SAVE ?= training
train-word: rebuild-word
	$(RUN_WORD) cargo run --release --features $(FEATURES) -- train-word $(if $(DATA),--data $(DATA)) $(if $(SYNTHETIC),--synthetic $(SYNTHETIC)) $(if $(WORD_SAVE),--save $(WORD_SAVE)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(BATCH),--batch-size $(BATCH)) $(if $(TRANSFER),--transfer $(TRANSFER)) $(if $(ISOLATION),--isolation-data $(ISOLATION)) $(if $(MONITOR),--monitor $(MONITOR))
