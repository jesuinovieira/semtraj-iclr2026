BACKENDS  = openai gemini qwen fasttext
FILENAMES = parkinson swear-fluency italian german
CUMULATIVES = true false

OUTDIR    = notebooks/executed
VENV      = .venv

# Notebook paths
EMBED_NOTEBOOK    = notebooks/02-embed.ipynb
METRICS_NOTEBOOK  = notebooks/03-metrics.ipynb
ANALYSIS_NOTEBOOK = notebooks/04-analysis-boxplots.ipynb

# Targets per notebook
EMBED_TARGETS 	 := $(foreach B,$(BACKENDS),\
						$(foreach F,$(FILENAMES),\
						$(OUTDIR)/02-embed_$(B)_$(F).ipynb))
METRICS_TARGETS  := $(foreach C,$(CUMULATIVES),\
						$(foreach B,$(BACKENDS),\
						$(foreach F,$(FILENAMES),\
						$(OUTDIR)/03-metrics_$(C)_$(B)_$(F).ipynb)))
ANALYSIS_TARGETS := $(foreach C,$(CUMULATIVES),\
						$(foreach B,$(BACKENDS),\
						$(foreach F,$(FILENAMES),\
						$(OUTDIR)/04-analysis-boxplots_$(C)_$(B)_$(F).ipynb)))

# Master targets
run: embed metrics analysis
embed:   $(EMBED_TARGETS)
metrics: $(METRICS_TARGETS)
analysis:$(ANALYSIS_TARGETS)

# Ensure output directory exists
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Rule for embed
$(OUTDIR)/02-embed_%.ipynb: $(EMBED_NOTEBOOK) | $(OUTDIR)
	@stem="$*"; \
	BACKEND="$${stem%%_*}"; \
	FILENAME="$${stem#*_}"; \
	echo ">>> Running $< with BACKEND=$${BACKEND} FILENAME=$${FILENAME}"; \
	. $(VENV)/bin/activate && \
	BACKEND="$${BACKEND}" FILENAME="$${FILENAME}" \
	$(VENV)/bin/jupyter nbconvert --to notebook --execute "$<" \
		--ExecutePreprocessor.timeout=-1 \
		--output "02-embed_$${stem}.ipynb" \
		--output-dir "$(OUTDIR)"

# Rule for metrics
$(OUTDIR)/03-metrics_%.ipynb: $(METRICS_NOTEBOOK) | $(OUTDIR)
	@stem="$*"; \
	CUMULATIVE="$${stem%%_*}"; \
	tmp="$${stem#*_}"; \
	BACKEND="$${tmp%%_*}"; \
	FILENAME="$${tmp#*_}"; \
	echo ">>> Running $< with CUMULATIVE=$${CUMULATIVE} BACKEND=$${BACKEND} FILENAME=$${FILENAME}"; \
	. $(VENV)/bin/activate && \
	CUMULATIVE="$${CUMULATIVE}" BACKEND="$${BACKEND}" FILENAME="$${FILENAME}" \
	$(VENV)/bin/jupyter nbconvert --to notebook --execute "$<" \
		--ExecutePreprocessor.timeout=-1 \
		--output "03-metrics_$${stem}.ipynb" \
		--output-dir "$(OUTDIR)"

# Rule for analysis
$(OUTDIR)/04-analysis-boxplots_%.ipynb: $(ANALYSIS_NOTEBOOK) | $(OUTDIR)
	@stem="$*"; \
	CUMULATIVE="$${stem%%_*}"; \
	tmp="$${stem#*_}"; \
	BACKEND="$${tmp%%_*}"; \
	FILENAME="$${tmp#*_}"; \
	echo ">>> Running $< with CUMULATIVE=$${CUMULATIVE} BACKEND=$${BACKEND} FILENAME=$${FILENAME}"; \
	. $(VENV)/bin/activate && \
	CUMULATIVE="$${CUMULATIVE}" BACKEND="$${BACKEND}" FILENAME="$${FILENAME}" \
	$(VENV)/bin/jupyter nbconvert --to notebook --execute "$<" \
		--ExecutePreprocessor.timeout=-1 \
		--output "04-analysis-boxplots_$${stem}.ipynb" \
		--output-dir "$(OUTDIR)"

.PHONY: run embed metrics analysis clean
clean:
	rm -rf $(OUTDIR)
