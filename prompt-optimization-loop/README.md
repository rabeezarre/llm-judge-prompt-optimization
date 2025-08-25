# Prompt Optimization Loop

This folder contains code and experiments for **automated prompt improvement** in LLM-as-a-Judge systems. The task studied here is **directive-strength classification**, where assistant messages are categorized as *Direct Command*, *Suggestion*, or *No Instruction*.  

The optimization loop builds on systematic error analysis to iteratively refine prompts, aiming to reduce **systematic biases** that persist across different prompting strategies.  

---

## Background

While ensemble methods aggregate multiple judges, they inherit shared systematic errors and come with high computational cost. A more effective alternative is **prompt optimization**, where prompts are automatically refined based on error patterns.  

This project implements an **automated loop**:  
1. Run baseline judges and collect predictions.  
2. Analyze systematic errors (e.g., *embedded instruction blindness*).  
3. Generate improved prompts with targeted bias-breaking instructions.  
4. Re-evaluate and measure performance gains.  

All experiments use 50 multi-turn conversations from Mistral’s production distribution with the `mistral-small` model at **temperature = 0**.  

---

## Key Findings

- **Prompt optimization consistently improved accuracy** across judge families (+6–14% absolute).  
- Achieved **breakthrough performance in Direct Command detection**, solving the *embedded instruction blindness* bias.  
- Required only **1.5× cost**, compared to 4–13× for ensemble methods.  
- Delivers the best trade-off between **performance and efficiency** for LLM-as-a-Judge reliability.  

---

## Results: Baseline vs Optimized Judges

| Judge Variant                  | Accuracy | Macro-F1 | Weighted-F1 | Δ Accuracy |
|--------------------------------|----------|----------|-------------|------------|
| Chain-of-Thought (Baseline)    | 0.680    | 0.444    | 0.683       | –          |
| Chain-of-Thought (Enhanced)    | 0.740    | 0.460    | 0.720       | **+0.060** |
| Closed Prompt (Baseline)       | 0.500    | 0.429    | 0.580       | –          |
| Closed Prompt (Enhanced)       | 0.640    | 0.488    | 0.644       | **+0.140** |
| Role Prompt (Baseline)         | 0.680    | 0.481    | 0.693       | –          |
| Role Prompt (Enhanced)         | 0.760    | 0.528    | 0.738       | **+0.080** |
| Few-shot (Baseline)            | 0.740    | 0.437    | 0.704       | –          |
| Few-shot (Enhanced)            | 0.820    | 0.562    | 0.798       | **+0.080** |
| CoT + Role + Few-shot (Baseline) | 0.760  | 0.461    | 0.722       | –          |
| CoT + Role + Few-shot (Enhanced) | 0.840  | 0.587    | 0.821       | **+0.080** |

**Highlights:**
- Accuracy improved by **+6 to +14 percentage points** across judge families.  
- Enhanced judges consistently improved **Macro-F1** and **Weighted-F1**.  
- The largest gain was for **Closed Prompt (+0.140 Accuracy)**.  
- Enhanced CoT+Role+Few-shot achieved the **best overall performance** (Accuracy = 0.840, Macro-F1 = 0.587).  

---

## Repository Structure

```
prompt-optimization-loop/
├── analyses/                   # Reports on optimization results
├── annotated/                  # Final annotated datasets
├── annotations/                # Raw human annotations
├── evaluations/                # Judge predictions before/after optimization
├── events/                     # Satasets
├── judges/                     # Judge definitions (baseline + optimized prompts)
├── concatenate_annotations.py  # Merge annotations from human and judge
├── human_annotator.py          # Tool for manual annotation
├── judge_analyzer.py           # Compute metrics and error analysis
├── judge_evaluator.py          # Run baseline and optimized judges
├── judge_improver.py           # Core optimization loop (generates improved prompts)
├── run.sh                      # Commands to reproduce experiments
├── requirements.txt            # Python dependencies
└── README.md                   # (this file)
```

---

## Setup

1. Create and activate virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install numpy matplotlib seaborn scikit-learn
```

3. Set up your API key in `.env` file:

```bash
MISTRAL_API_KEY=your_actual_api_key_here
```

---

## Running Experiments

The pipeline has five steps: **annotate → evaluate → analyze → improve → re-evaluate**.

### 1. Annotate Conversations
Manual labeling of assistant messages.  

```bash
python human_annotator.py \
  --judge directive_strength_cot_prompt_judge.json \
  --events directive_strength_dataset.json \
  --output directive_strength_annotations.json \
  --use-folders
```

### 2. Run Baseline Evaluation
Evaluate with an initial judge definition.  

```bash
python judge_evaluator.py \
  --judge judges/directive_strength_cot_prompt_judge.json \
  --events directive_strength_dataset.json \
  --output baseline_results.json \
  --use-folders
```

### 3. Analyze Errors
Compute metrics and identify systematic failure cases.  

```bash
python judge_analyzer.py \
  --annotated annotated/conversations.json \
  --judge judges/directive_strength_cot_prompt_judge.json \
  --output analyses/baseline_analysis.json \
  --use-folders
```

### 4. Improve Prompts
Automatically refine the prompt based on error analysis.  

```bash
python judge_improver.py \
  --input judges/directive_strength_cot_prompt_judge.json \
  --errors analyses/baseline_analysis.json \
  --output judges/directive_strength_cot_prompt_judge_optimized.json
```

### 5. Re-Evaluate with Optimized Prompt
Run the improved judge and measure gains.  

```bash
python judge_evaluator.py \
  --judge judges/directive_strength_cot_prompt_judge_optimized.json \
  --events directive_strength_dataset.json \
  --output optimized_results.json \
  --use-folders
```

---

## Generalization

The optimization loop is **task-agnostic**:  
- Works with any judge JSON in `judges/`.  
- Can be applied to any dataset of events.  
- The same loop (evaluate → analyze → improve → re-evaluate) applies to other evaluation domains (factuality, coherence, safety, etc.).  
