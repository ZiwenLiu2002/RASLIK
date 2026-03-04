# RASLIK: Randomized Antipodal Search on Linearized Influence Kernel for Data-Centric LLM Unlearning


The implementation for the paper **“Randomized Antipodal Search Done Right for Data Pareto Improvement of LLM Unlearning”** (ICLR 2026).

RASLIK is a retrieval framework for data-centric LLM unlearning. Instead of assuming the forget and retain sets are already available, RASLIK retrieves them from a large training corpus using randomized antipodal search over a linearized influence kernel. It is designed to improve the forgetting–retention trade-off while remaining scalable to large language models.

**Keywords:** LLM Unlearning, Training Data Retrieval, Influence Estimation, Data Attribution, Randomized Search

## Overview

Existing unlearning methods typically assume that the forget and retain sets are known in advance. In practice, however, unlearning is often triggered only after an undesirable generation is observed at inference time. This makes **data retrieval** the central challenge.

RASLIK addresses this problem by:

* retrieving both **forget** and **retain** candidates from the training corpus,
* using **randomized antipodal search** for efficient large-scale retrieval,
* improving both **retrieval quality** and **retrieval efficiency** compared with deterministic baselines.

This repository includes:

* caching scripts for training gradients / sketches,
* retrieval scripts for test queries,
* post-processing utilities to directly export `forget.jsonl` and `retain.jsonl`.

## Quick Start

Clone this repository to your local machine.

```bash
git clone https://github.com/ZiwenLiu2002/RASLIK.git
cd RASLIK
```

Create a new environment and install dependencies.

```bash
conda create -n RASLIK python=3.10
conda activate RASLIK
pip install -r requirements.txt
```

Once you have a config file, you can run:

```bash
python MP_main.py --config_path path/to/config.json
```

## Running Example

We provide example config files in `./examples`.

A typical workflow has two stages:

1. **Caching Stage**
   Precompute and store training gradients or RapidGrad sketches.

2. **Retrieval Stage**
   Load cached gradients / sketches and retrieve influential training data for the test queries.

If post-processing is enabled, the retrieval stage will additionally export:

* `forget.jsonl`
* `retain.jsonl`

to the configured export directory.

## Caching Stage

Run the caching stage first to generate the gradient cache or RapidGrad sketches for the training corpus.

Example:

```bash
cd ./examples
python ../MP_main.py --config_path ./config_caching.json
```

This stage typically writes:

* standard logs and retrieval outputs to `influence.outdir`,
* cached gradients / sketches to `influence.grads_path`.

Please make sure you have sufficient disk space for cached gradients or sketches.

## Retrieval Stage

After caching the training gradients, run retrieval on the test dataset.

```bash
python ../MP_main.py --config_path ./config_retrieval.json
```

This stage produces a retrieval result JSON in the configured output directory. If post-processing is enabled, it also exports the final retrieved subsets:

* `forget.jsonl`
* `retain.jsonl`

These files can be directly used by downstream unlearning methods.


## Configure File

A typical config file contains the following sections.

### `data`

* `train_data_path` *(required, str)*: path to the training dataset.
* `test_data_path` *(optional for caching, required for retrieval, str)*: path to the test queries / generations.

### `influence`

* `outdir` *(required, str)*: path to standard outputs and retrieval result JSON files.
* `seed` *(optional, int, default: 42)*: random seed.
* `cal_words_infl` *(optional, bool, default: false)*: whether to compute token-wise influence.
* `grads_path` *(optional, str)*: path to cached full gradients or RapidGrad sketches.
* `load_from_grads_path` *(optional, bool, default: false)*: whether to load cached gradients from `grads_path`.
* `save_to_grads_path` *(optional, bool, default: false)*: whether to save gradients to `grads_path`.
* `n_threads` *(optional, int, default: 1)*: number of threads per GPU.
* `skip_test` *(optional, bool)*: skip test processing, typically used in caching.
* `skip_influence` *(optional, bool)*: skip influence computation, typically used in caching-only runs.
* `top_k` *(optional, int, default: 1000)*: number of top retrieved candidates to keep.
* `offload_test_grad` *(optional, bool)*: whether to offload test gradients to CPU.
* `offload_train_grad` *(optional, bool)*: whether to offload training gradients to CPU.
* `calculate_infl_in_gpu` *(optional, bool)*: whether to compute influence scores on GPU.
* `delete_model` *(optional, bool)*: whether to delete the model after gradient extraction to save memory.

#### `influence.RapidGrad`

* `enable` *(optional, bool, default: false)*: whether to use RapidGrad sketches.
* `RapidGrad_K` *(optional, int, default: 65536)*: projected gradient dimensionality.
* `shuffle_lambda` *(optional, int, default: 20)*: number of shuffles used in randomized projection.

### `model`

* `model_path` *(required, str)*: base model path or Hugging Face model name.
* `lora_path` *(optional, str)*: LoRA / QLoRA adapter path.
* `max_length` *(optional, int, default: 512)*: maximum sequence length.
* `load_in_4bit` *(optional, bool, default: false)*: whether to quantize the model in 4-bit.

### `postprocess`

* `enable` *(optional, bool, default: false)*: whether to export final retrieved sets after retrieval.
* `top_n` *(optional, int)*: number of samples to export for each of `forget.jsonl` and `retain.jsonl`.
* `list_key` *(optional, str, e.g. `helpful` or `harmful`)*: ranking field used for aggregation.
* `infl_key` *(optional, str, e.g. `helpful_infl` or `harmful_infl`)*: score field paired with `list_key`.
* `export_dir` *(optional, str)*: directory to save `forget.jsonl` and `retain.jsonl`.

## Example Configs

### Caching example

Use a config that saves gradient caches and skips retrieval:

* `save_to_grads_path: true`
* `skip_test: true`
* `skip_influence: true`

### Retrieval example

Use a config that loads cached gradients and performs retrieval:

* `load_from_grads_path: true`
* `save_to_grads_path: false`
* `test_data_path: ...`
* `postprocess.enable: true`

## Notes

* The retrieval stage assumes that the training dataset indexing is consistent with the IDs stored in the retrieval result JSON.
* If `postprocess.enable` is turned on, the code automatically aggregates ranked IDs and maps them back to the training JSONL file.
* For large models and datasets, make sure both GPU memory and disk space are sufficient.

## Citation

```bibtex
@inproceedings{
liu2026randomized,
title={Randomized Antipodal Search Done Right for Data Pareto Improvement of {LLM} Unlearning},
author={Ziwen Liu and Huawei Lin and Yide Ran and Denghui Zhang and Jianwen Xie and Chuan Li and Weijie Zhao and Zhaozhuo Xu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Xn6EnJZghu}
}
```


