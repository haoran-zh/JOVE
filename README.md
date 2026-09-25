# JOVE

Joint executor and verification allocation for LLM task graphs.

A planner decomposes each query into an executable DAG. JOVE then solves a per-query mixed-integer program that assigns an executor API to every node and decides whether to call a fixed verifier. Executor quality is tracked with LinUCB; verifier value is D-optimal information gain `I(u) = 0.5 * log(1 + u^2)`. A virtual queue `q` applies soft cost pressure against a long-term budget `gamma`, and sink latency is constrained by a deadline `mu`.

Verifier feedback is used only to update the executor quality model. Task outputs are never repaired or replanned from verifier labels.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp API_keys.txt.example API_keys.txt
```

Put `OPENROUTER_API_KEY=...` (or a NVIDIA NIM key) in `API_keys.txt`. Do not commit that file.

Offline optimizer check:

```bash
python jove.py --smoke-test
```

## Run

OpenRouter example (paper operating point: `gamma=200`, `mu=15`):

```bash
python jove.py \
  --base-url https://openrouter.ai/api/v1/chat/completions \
  --api-key-name OPENROUTER_API_KEY \
  --max-parallel-tasks 8 --request-interval 0 \
  --dataset bamboogle \
  --gamma 200 --mu 15 \
  --k-c 1e-9 --k-v 1.0 --beta 1.0 --lambda-reg 1.0 \
  --graph-node-counts 2,3,4,5
```

Datasets: `bamboogle`, `mmlu_pro`, `aime24`, `livebench_reasoning`, `gpqa` (Hugging Face gated `Idavidrein/gpqa`). `--sample-size 0` uses the full shuffled stream. Positive `--sample-size` keeps that many examples after the seed shuffle. `--graph-node-counts 2,3,4,5` expands each query into four planner DAGs.

Logs and `metrics.pkl` / `summary.json` are written under `logs/`.
