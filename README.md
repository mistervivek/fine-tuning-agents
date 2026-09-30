# AI Agent & Research
### SFT • Direct Preference Optimization (DPO) • Reward Modeling (PRM/ORM) • Trajectory Distillation • Trace Inspection & Failure Diagnostics

This repository has been redesigned from a naive prompt-and-call tutorial into a **modular, production-grade Post-Training & Agent Research Engineering platform**. It equips research engineers to fine-tune, align, verify, and diagnose multi-turn agent policies.

---

## 🏛️ Architecture Overview

```
                                 [ Frontier Model / Policy Teacher ]
                                                 │
                               Trajectory Distillation & Filtering
                               (Rejection Sampling / Verifiers)
                                                 │
                                                 ▼
                                [ 1. Agent SFT with Loss Masking ]
                                (Assistant-only CrossEntropy, -100 on prompts)
                                                 │
                                   Contrastive Failure Pairs
                                                 │
                                                 ▼
                              [ 2. Preference Optimization (DPO) ]
                              (Implicit Reward Margins & Policy Loss)
                                                 │
                                  Step-Level PRM & Outcome ORM
                                                 │
                                                 ▼
                              [ 3. Reward Modeling (PRM & ORM) ]
                              (Credit assignment & invariant scoring)
                                                 │
                                 Deterministic Benchmark Harness
                                                 │
                                                 ▼
                            [ 4. Automated Failure Mode Diagnostics ]
                            (Security breach, Hallucination, Inefficiencies)
```

---

## 🔬 Core Research Engineering Pillars

### 1. Agent SFT with Assistant-Only Loss Masking (`src/sft/`)
- **The Problem**: Standard causal language modeling trains on the entire token sequence. Computing cross-entropy loss over user prompts and external tool observation outputs forces models to memorize DB dumps and hallucinate responses rather than calling tools.
- **The Solution**: Token-level loss masking (`labels = -100` on User and Tool tokens, backpropagating gradients strictly on Assistant reasoning tokens and tool arguments).

### 2. Preference Optimization (DPO) for Agent Policies (`src/preference/`)
- Implements Direct Preference Optimization (Rafailov et al., 2023) directly on agent trajectories:
  $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
- Contrastive trajectory pairs $(y_w, y_l)$ penalize specific failure modes:
  - Tool Hallucination (claiming balances without executing queries)
  - Unauthorized Access (calling private endpoints without verifying identity)
  - AML Compliance Bypass (skipping regulatory fraud flags on transactions $\ge \$10,000$)

### 3. Reward Modeling: ORM vs PRM (`src/reward/`)
- **Outcome Reward Model (ORM)**: Evaluates terminal completion, factuality, and invariant conformance.
- **Process Reward Model (PRM)**: Solves the credit assignment problem by scoring each discrete step:
  - Action relevance
  - Argument extraction correctness
  - Step-count penalty (penalizing rambling / redundant tool calls)

### 4. Trajectory Distillation (`src/distillation/`)
- Collects high-capacity teacher trajectories.
- Runs **Verifier-based Rejection Sampling** to prune flawed or inefficient runs ($\ge 0.60$ threshold).
- Exports validated, clean multi-turn records for student model fine-tuning.

### 5. Automated Failure Mode Classifier & Trace Telemetry (`src/eval/`)
- Telemetry captures step type, duration, tool call signatures, and observations.
- Failure taxonomy automatically classifies agent runs:
  - `UNAUTHORIZED_ACCESS`
  - `TOOL_HALLUCINATION`
  - `POLICY_VIOLATION`
  - `ARGUMENT_INVALID`
  - `REDUNDANT_TOOL_CALL`
  - `PREMATURE_TERMINATION`

---

## 📁 Repository Structure

```
fine tuning agents/
├── cli.py                            # Research engineer CLI runner
├── fine_tuning_agents_guide.ipynb    # Comprehensive interactive research guide
├── requirements.txt                  # Environment dependencies
├── src/
│   ├── schemas/                      # Pydantic schemas (Trajectory, Preference, Evaluation)
│   ├── environment/                  # Deterministic BankEnvironment & LangChain tools
│   ├── sft/                          # Dataset builder & Assistant-only loss masking
│   ├── preference/                   # DPO contrastive builder & mathematical loss engine
│   ├── reward/                       # Outcome (ORM) and Process (PRM) reward models
│   ├── distillation/                 # Rejection sampling & trajectory distiller
│   ├── agent/                        # LangGraph agent & offline deterministic policy
│   └── eval/                         # Benchmark harness, taxonomy classifier, trace inspector
├── tests/                            # Comprehensive PyTest test suite (100% passing)
└── data/                             # Generated SFT and DPO datasets
```
