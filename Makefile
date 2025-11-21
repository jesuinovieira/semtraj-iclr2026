BACKENDS  = gemini openai qwen
FILENAMES = UFABC_PLT_combined swear-fluency CPN120 italian german parkinson

OUTDIR    = notebooks/executed
VENV      = .venv

# Notebook paths
EMBED_NOTEBOOK    = notebooks/01-embed.ipynb
METRICS_NOTEBOOK  = notebooks/02-metrics.ipynb
ANALYSIS_NOTEBOOK = notebooks/03-analysis.ipynb

# Targets per notebook
EMBED_TARGETS    := $(foreach B,$(BACKENDS),$(foreach F,$(FILENAMES),$(OUTDIR)/01-embed_$(B)_$(F).ipynb))
METRICS_TARGETS  := $(foreach B,$(BACKENDS),$(foreach F,$(FILENAMES),$(OUTDIR)/02-metrics_$(B)_$(F).ipynb))
ANALYSIS_TARGETS := $(foreach B,$(BACKENDS),$(foreach F,$(FILENAMES),$(OUTDIR)/03-analysis_$(B)_$(F).ipynb))

# Master targets
run: embed metrics analysis
embed:   $(EMBED_TARGETS)
metrics: $(METRICS_TARGETS)
analysis:$(ANALYSIS_TARGETS)

# Ensure output directory exists
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Rule for embed
$(OUTDIR)/01-embed_%.ipynb: $(EMBED_NOTEBOOK) | $(OUTDIR)
	@stem="$*"; \
	BACKEND="$${stem%%_*}"; \
	FILENAME="$${stem#*_}"; \
	echo ">>> Running $< with BACKEND=$${BACKEND} FILENAME=$${FILENAME} -> $@"; \
	. $(VENV)/bin/activate && \
	BACKEND="$${BACKEND}" FILENAME="$${FILENAME}" \
	$(VENV)/bin/jupyter nbconvert --to notebook --execute "$<" \
		--ExecutePreprocessor.timeout=-1 \
		--output "01-embed_$${stem}.ipynb" \
		--output-dir "$(OUTDIR)"

# Rule for metrics
$(OUTDIR)/02-metrics_%.ipynb: $(METRICS_NOTEBOOK) | $(OUTDIR)
	@stem="$*"; \
	BACKEND="$${stem%%_*}"; \
	FILENAME="$${stem#*_}"; \
	echo ">>> Running $< with BACKEND=$${BACKEND} FILENAME=$${FILENAME} -> $@"; \
	. $(VENV)/bin/activate && \
	BACKEND="$${BACKEND}" FILENAME="$${FILENAME}" \
	$(VENV)/bin/jupyter nbconvert --to notebook --execute "$<" \
		--ExecutePreprocessor.timeout=-1 \
		--output "02-metrics_$${stem}.ipynb" \
		--output-dir "$(OUTDIR)"

# Rule for analysis
$(OUTDIR)/03-analysis_%.ipynb: $(ANALYSIS_NOTEBOOK) | $(OUTDIR)
	@stem="$*"; \
	BACKEND="$${stem%%_*}"; \
	FILENAME="$${stem#*_}"; \
	echo ">>> Running $< with BACKEND=$${BACKEND} FILENAME=$${FILENAME} -> $@"; \
	. $(VENV)/bin/activate && \
	BACKEND="$${BACKEND}" FILENAME="$${FILENAME}" \
	$(VENV)/bin/jupyter nbconvert --to notebook --execute "$<" \
		--ExecutePreprocessor.timeout=-1 \
		--output "03-analysis_$${stem}.ipynb" \
		--output-dir "$(OUTDIR)"

.PHONY: run embed metrics analysis clean
clean:
	rm -rf $(OUTDIR)
