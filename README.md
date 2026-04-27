# Fed-FuseZO: Information-Guided Hybrid Optimization for Memory-Constrained Federated LLM Fine-Tuning

## 1) Overview
This repo contains the source code and reproducing guide of Fed-FuseZO. This research introduces an information-guided hybrid optimization framework designed to tackle the "memory wall" in federated fine-tuning of Large Language Models (LLMs). Our study formalizes the memory-variance dilemma by introducing a layer-wise ZO-cost-adjusted information score, which quantifies the trade-off between gradient information gain (Fisher Information) and geometric sensitivity (Curvature). Fed-FuseZO automatically determines the optimal FO/ZO boundary using a Kneedle-based detection mechanism, enabling efficient, memory-constrained federated training without heuristic ratio tuning.

### This project covers the following scopes:

- Three LLM families: OPT-1.3B, LLAMA-2-3B, and GPT-J-6B.

- Three task complexities: SST-2 (Classification), COPA (Reasoning), and PiQA (Commonsense Reasoning).

- Three parameter-efficient fine-tuning (PEFT) paradigms: LoRA, Prefix tuning, and Prompt tuning.

### Core Innovation: An information-guided hybrid optimization mechanism that adaptively partitions layers into First-Order (FO) and Zeroth-Order (ZO) update streams based on layer-wise sensitivity.

## 2) Project Structure
This project is structured around the adaptive hybrid optimization for various tasks & models & PEFT schemes. Core logic is implemented in fed-fuse/optimizer.py and fed-fuse/trainer.py. Task configurations are defined in fed-fuse/tasks.py. The main entry point is fed-fuse/run.py.

'''Plaintext
.
├── fed-fuse
│   ├── modeling_opt
│   ├── modeling_llama
│   ├── modeling_gptj
│   ├── trainer.py           # Hybrid FO/ZO execution logic
│   ├── optimizer.py         # Information-guided score computation & Kneedle logic
│   ├── run.py               # Main entry point for federated training
│   ├── tasks.py             # Data loading and task formatting
│   ├── utils.py
│   ├── sweep                # Hyperparameter sweeps
│   │   ├── SST2_opt-1.3b
│   │   ├── Copa_llama-2-3b
│   │   ├── PiQA_gptj-6b
│   │   └── ...
├── environment.yml
'''

3) Getting Started
Ensure you have a compatible environment:

Bash
conda create -n fedfuse python=3.10
conda activate fedfuse
pip install -r requirements.txt
4) Reproducing Results
We provide detailed hyperparameter settings in sweeps/. The configuration for tuning a MODEL on TASK under SCHEME with Fed-FuseZO is organized as fed-fuse/sweeps/TASK_MODEL/SCHEME.yml.

Example: Running the federated hybrid optimization for LLaMA-2-3B on the COPA task with LoRA:

Bash
## Start the sweep
wandb sweep fed-fuse/sweeps/Copa_llama-2-3b/lora_fedfuse.yml

## Run the agent
wandb agent <your-sweep-id>
Extended Usage:

Custom Memory Budget: You can adjust the memory constraint via the --memory_budget_gb flag. Fed-FuseZO will automatically re-calculate the FO/ZO boundary.

Information-Guided Logic: To adjust the sensitivity (Fisher Information vs. Curvature), modify the hyperparameters in optimizer.py or pass them via command line:

Bash
python run.py --model_name=meta-llama/Llama-2-3b-hf --task_name=COPA \
--trainer=fed_fuse --alpha=0.5 --rho_min=0.5 --rho_max=1.0 \
--use_kneedle=True --per_device_train_batch_size=8
Gradient Pruning & Sparsity: If you wish to extend the method with gradient sparsity, utilize the --gradient_sparsity flag to enable additional variance reduction in the ZO update phase.
