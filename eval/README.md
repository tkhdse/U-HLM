# U-HLM Evaluation Guide

This directory contains scripts for evaluating U-HLM performance against a baseline LLM.

## Architecture

**U-HLM (2-tier):**
- SLM (Llama-3.2-1B-Instruct) → LLM (Llama-3.1-8B-Instruct)
- Decision logic: SKIP (trust SLM) or VERIFY_LLM (offload to LLM)
- Threshold: `u_th = 0.6316`

## Setup

1. **Install additional dependencies:**
   ```bash
   pip install pandas bert-score
   ```

2. **Model Configuration:**
   - SLM: `meta-llama/Llama-3.2-1B-Instruct` (draft model)
   - LLM: `meta-llama/Llama-3.1-8B-Instruct` (verifier model)
   - Threshold: `0.6316` (hardcoded)

## Running Evaluations

### Step 1: Start LLM Service

**Terminal 1 - LLM Service:**
```bash
cd llm-service
python -m server.main
```
Wait for: `gRPC server started on [::]:8081`

### Step 2: Run U-HLM Evaluation

**Terminal 2 - Evaluation:**
```bash
cd eval
python run_uhlm_eval.py --prompts test_prompts.txt --output uhlm-results --runs 3 --max-tokens 50
```

This will:
- Run each prompt 3 times (for averaging, especially for BERTScore)
- Collect metrics (TPS, TTNT, acceptance rate, RPC calls, etc.)
- Log GPU utilization automatically at 10 Hz
- Save results to `uhlm-results/` directory

### Step 3: Run Baseline Evaluation

```bash
python run_baseline_eval.py --prompts test_prompts.txt --output results_baseline --runs 3 --max-tokens 50
```

### Step 4: Analyze Results

After both evaluations complete, compare the results:

```bash
python analyze_results.py \
  --uhlm-results uhlm-results/all_results_YYYYMMDD_HHMMSS.json \
  --uhlm-gpu-log uhlm-results/gpu_log_YYYYMMDD_HHMMSS.csv \
  --baseline-results results_baseline/baseline_all_results_YYYYMMDD_HHMMSS.json \
  --baseline-gpu-log results_baseline/gpu_log_baseline_YYYYMMDD_HHMMSS.csv \
  --output comparison.json
```

### Step 5: Compute BERTScore

```bash
python compute_bertscore.py \
  --uhlm-results uhlm-results/all_results_YYYYMMDD_HHMMSS.json \
  --baseline-results results_baseline/baseline_all_results_YYYYMMDD_HHMMSS.json \
  --output bertscore_results.json
```

## Metrics Collected

### Performance Metrics (per instructions)
| Metric | Description | Formula |
|--------|-------------|---------|
| **TPS (measured)** | Tokens per second | `T_out / (T_end - T_start)` |
| **TPS (normalized)** | Theoretical TPS at 100% GPU | `TPS_measured / avg_U_compute` |
| **TTNT** | Time to next token | Difference between consecutive token timestamps |
| **End-to-End Latency** | Total wall-clock time per query | `T_end - T_start` |
| **GPU Utilization** | Average GPU compute utilization | Sampled via nvidia-smi at 10 Hz |

### Accuracy Metrics
| Metric | Description |
|--------|-------------|
| **Acceptance Rate** | % of tokens accepted (SKIP + LLM accepted) / total |
| **BERTScore** | Semantic similarity between U-HLM and baseline outputs |

### Communication Overhead
| Metric | Description |
|--------|-------------|
| **RPC Calls** | Count of BeginSession, VerifyToken, Sync, EndSession |
| **Decision Counts** | Breakdown of SKIP vs VERIFY_LLM decisions |

## Output Files

- `uhlm-results/all_results_*.json`: Complete U-HLM evaluation results
- `results_baseline/baseline_all_results_*.json`: Complete baseline evaluation results
- `results_*/gpu_log_*.csv`: GPU utilization logs
- `comparison.json`: Comparison report between U-HLM and baseline
- `bertscore_results.json`: BERTScore analysis with per-prompt averages

## Manual GPU Monitoring

If you want to monitor GPU manually:

```bash
python gpu_monitor.py --output gpu_log.csv --rate 10.0
```

Press Ctrl+C to stop.

## Notes

1. **Multiple Runs**: Use `--runs 3` (or more) to average across runs, reducing variance (especially important for BERTScore as specified in instructions).

2. **GPU Monitoring**: Runs automatically at 10 Hz during evaluation. Logs timestamp, GPU utilization %, memory utilization %, memory used/total.

3. **Normalized TPS Interpretation**:
   - `TPS_measured` = real-world performance
   - `TPS_norm` = theoretical performance at 100% GPU utilization
   - Large gap indicates overhead from context switching, memory contention, IPC latency, etc.

4. **BERTScore**: Compares U-HLM output against baseline LLM output. Higher score = better semantic preservation.

## Troubleshooting

- **Services not starting**: Check port 8081 is available
- **GPU monitoring fails**: Ensure `nvidia-smi` is available
- **Import errors**: Run from repo root or set PYTHONPATH
- **Out of memory**: Reduce `--max-tokens` or run fewer prompts
