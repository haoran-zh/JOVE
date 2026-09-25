# JOVE

Joint executor and verification allocation for LLM task graphs.

Verifier feedback is used only to update the executor quality model. The response is produced from the sink before verifier calls, so verification time is excluded from response latency. 

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp API_keys.txt.example API_keys.txt
```

Put `OPENROUTER_API_KEY=...`  in `API_keys.txt`. 

Offline optimizer check:

```bash
python jove.py --smoke-test
```

