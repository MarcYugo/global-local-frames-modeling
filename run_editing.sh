#! /usr/bin/env bash

export GIT_PYTHON_REFRESH=quiet
export HUGGINGFACE_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxx

# ====================== inversion ======================
python glfm_preprocess.py --config_path ./configs/config_pnp_woman_running2_glfm.yaml

# ======================= edit ==========================
python run_tokenflow_pnp.py --config_path ./configs/config_pnp_woman_running2_glfm.yaml

# ====================== inversion ======================
# python stem_preprocess.py --config_path ./configs/config_pnp_woman_running2.yaml

# ======================= edit ==========================
# python run_tokenflow_pnp.py --config_path ./configs/config_pnp_woman_running2.yaml
