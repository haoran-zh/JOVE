# JOVE

Joint executor and verification allocation for LLM task graphs.

A planner decomposes each query into an executable DAG. JOVE then solves a per-query mixed-integer program that assigns an executor API to every node and decides whether to call a fixed verifier. Executor quality is tracked with LinUCB; verifier value is D-optimal information gain `I(u) = 0.5 * log(1 + u^2)`. The objective averages node quality and information gain over the graph size. The budget price is `lambda = k_c * q`, where `q` accumulates realized cost minus the long-term budget `gamma`; the full billed verifier cost enters both the objective and the queue. The sink deadline reserves time for planning and allocation.

Verifier feedback is used only to update the executor quality model. The response is produced from the sink before verifier calls, so verification time is excluded from response latency. Task outputs are never repaired or replanned from verifier labels.

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

OpenRouter example (paper operating point: `gamma=200` micro-USD, `mu=16` seconds):

```bash
python jove.py \
  --base-url https://openrouter.ai/api/v1/chat/completions \
  --api-key-name OPENROUTER_API_KEY \
  --max-parallel-tasks 8 --request-interval 0 \
  --dataset bamboogle \
  --gamma 200 --mu 16 \
  --k-c 1e-6 --k-v 5.0 --beta 0.25 --lambda-reg 1.0 \
  --graph-node-counts 2,3,4,5
```

Datasets: `bamboogle`, `mmlu_pro`, `aime24`, `livebench_reasoning`, `gpqa` (Hugging Face gated `Idavidrein/gpqa`). `--sample-size 0` uses the full shuffled stream. Positive `--sample-size` keeps that many examples after the seed shuffle. `--graph-node-counts 2,3,4,5` expands each query into four planner DAGs.

Logs and `metrics.pkl` / `summary.json` are written under `logs/`.

The OpenRouter defaults use the paper's eight executor models, Gemini 2.5 Flash Lite planner, Qwen3-235B-A22B-2507 verifier, `gamma = 200` micro-USD, `mu = 16` seconds, `k_v = 5`, `beta = 0.25`, and `delta = 0.1`. A task may optionally supply `candidate_executors` in a parsed plan; otherwise every configured executor is eligible. No baseline policy is included.

The catalog cost and latency priors are provisional. For a faithful empirical reproduction, calibrate them with provider statistics or preliminary calls, and confirm that the provider returns billed USD in `usage.cost`. The offline smoke test checks the optimizer and accounting without making API calls.
