# Runs every experiment in sequence and writes one log per script to logs/.
# Usage (from src/extension/):
#   make -k all     # -k keeps going if one experiment fails
#   make cifar100   # or any single group / target below

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

LOGS := $(CURDIR)/logs
REPL := $(CURDIR)/../replication
PY := python -u

.PHONY: all synthetic cifar10 cifar100 \
	synthetic_laplace synthetic_gmm \
	cifar10_contrastive cifar10_supervised \
	cifar100_contrastive cifar100_supervised

all: synthetic cifar10 cifar100

synthetic: synthetic_laplace synthetic_gmm
cifar10: cifar10_contrastive cifar10_supervised
cifar100: cifar100_contrastive cifar100_supervised

$(LOGS):
	mkdir -p $@

synthetic_laplace: | $(LOGS)
	cd $(REPL) && $(PY) run_single_laplace.py 2>&1 | tee $(LOGS)/$@.log

synthetic_gmm: | $(LOGS)
	cd $(REPL) && $(PY) run_single_gmm.py 2>&1 | tee $(LOGS)/$@.log

cifar10_contrastive: | $(LOGS)
	cd $(REPL)/cifar10_resnet18 && $(PY) train_contrastive.py 2>&1 | tee $(LOGS)/$@.log

cifar10_supervised: | $(LOGS)
	cd $(REPL)/cifar10_resnet18 && $(PY) train_supervised.py 2>&1 | tee $(LOGS)/$@.log

cifar100_contrastive: | $(LOGS)
	cd cifar100_resnet18 && $(PY) train_contrastive.py 2>&1 | tee $(LOGS)/$@.log

cifar100_supervised: | $(LOGS)
	cd cifar100_resnet18 && $(PY) train_supervised.py 2>&1 | tee $(LOGS)/$@.log
