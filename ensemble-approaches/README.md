# Ensemble Approaches

This folder contains code and experiments for evaluating **ensemble methods in LLM-as-a-Judge systems**. The task studied here is **directive-strength classification**, where assistant messages are categorized as *Direct Command*, *Suggestion*, or *No Instruction*.  

Ensemble methods combine outputs from multiple judges or multiple runs of the same judge, with the aim of improving reliability.  

---

## Background

While individual prompting strategies can reduce some errors, they often share the same **systematic biases**. A common idea is to use **ensembles**: either aggregating results from several prompts (majority vote) or running the same prompt multiple times and aggregating (self-consistency).  

In practice, ensembles increase computational cost, and if all components share the same bias, the ensemble inherits the same failures.  

This project evaluates two ensemble methods on 50 multi-turn conversations from Mistral’s production distribution, using the `mistral-small` model with **temperature = 0** for reproducibility.  

---

## Ensemble Methods Tested

1. **Prompt-Ensemble Majority Vote**  
   - Runs multiple judges with different prompts.  
   - The final decision is the label chosen by the majority.  

2. **Self-Consistency Ensemble**  
   - Runs the same judge multiple times.  
   - The final decision is based on the most common prediction across runs.  

---

## Performance and Efficiency Comparison

| Method                           | Accuracy | Direct Command F1 | Cost Multiplier | Efficiency Ratio |
|----------------------------------|----------|-------------------|-----------------|------------------|
| **Single Judge (CoT + Role + Few-shot)** | 0.760    | 0.000             | 1×              | 1.00             |
| Prompt-Ensemble (4 judges)       | 0.760    | 0.000             | 4×              | 0.25             |
| Self-Consistency (5–13 paths)    | 0.760–0.780 | 0.000          | 5–13×           | 0.08–0.20        |
| **Prompt Optimization (ref)**    | +6–14%   | 0.571             | 1.5×            | 4.0–9.3          |

**Notes:**
- All ensemble methods failed on Direct Command detection (F1 = 0.000), showing **embedded instruction blindness** persisted.  
- Prompt-ensemble gave no gains while multiplying cost by 4×.  
- Self-consistency produced agreement but replicated the same systematic errors.  
- Prompt optimization (outside this folder) achieved actual improvements at much lower cost.  

---

## Repository Structure

```
ensemble-approaches/
├── analyses/                          # Performance reports (JSON, metrics, confusion matrices)
├── annotations/                       # Raw human annotations
├── ensembles/                         # Ensemble run results
├── events/                            # Dataset
├── judges/                            # Judge definitions (prompt templates in JSON)
├── ensemble_majority_vote.py          # Run prompt-ensemble majority voting
├── ensemble_majority_vote_analyzer.py # Analyze majority vote results
├── ensemble_self_consistency.py       # Run self-consistency ensemble
├── ensemble_self_consistency_analyzer.py # Analyze self-consistency results
├── human_annotator.py                 # Tool for manual annotation
├── judge_analyzer.py                  # Compute metrics and error analysis
├── judge_evaluator.py                 # Run judge prompts on dataset
├── run.sh                             # Commands to reproduce experiments
├── requirements.txt                   # Python dependencies
└── README.md                          # (this file)
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

The pipeline has four steps: **annotate → evaluate → ensemble → analyze**.

### 1. Annotate Conversations
Manual labeling of assistant messages.  

```bash
python human_annotator.py \
  --judge directive_strength_open_prompt_judge.json \
  --events directive_strength_dataset.json \
  --output directive_strength_annotations.json \
  --use-folders
```

### 2. Run Ensemble Methods
**Majority Vote**:  

```bash
python ensemble_majority_vote.py \
  --judges judges/ \
  --events directive_strength_dataset.json \
  --output ensemble_majority_results.json \
  --use-folders
```

**Self-Consistency**:  

```bash
python ensemble_self_consistency.py \
  --judge judges/directive_strength_cot_prompt_judge.json \
  --events directive_strength_dataset.json \
  --output ensemble_self_consistency_results.json \
  --use-folders
```

### 3. Analyze Results
Compute ensemble performance.  

**Majority Vote**:  
```bash
python ensemble_majority_vote_analyzer.py \
  --input ensemble_majority_results.json \
  --output analyses/ensemble_majority_analysis.json \
  --use-folders
```

**Self-Consistency**:  
```bash
python ensemble_self_consistency_analyzer.py \
  --input ensemble_self_consistency_results.json \
  --output analyses/ensemble_self_consistency_analysis.json \
  --use-folders
```

---

## Generalization

The ensemble framework is **not limited to directive-strength classification**.  
- Any judge JSON in `judges/` can be used.  
- Any dataset of events can be substituted.  
- Both majority vote and self-consistency can be applied to other evaluation tasks (factuality, coherence, safety, etc.).  
