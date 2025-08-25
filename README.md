# LLM Judge Prompt Optimization

This repository contains experiments and code for **evaluating and improving the reliability of LLM-as-a-Judge systems**.  
The project investigates:  

1. **Prompt Strategies** – benchmarking different prompting techniques.  
2. **Ensemble Approaches** – combining multiple judges or reasoning paths.  
3. **Prompt Optimization Loop** – automated refinement of prompts based on systematic error analysis.  

The work is part of the Master’s thesis *Prompt Optimization for LLM-as-a-Judge Evaluation* (École Polytechnique & Mistral AI).  

---

## Background

Large Language Models are increasingly used as **automatic judges** to evaluate model outputs. This approach scales human evaluation but introduces **systematic biases and reliability issues**.  

The key evaluation task in this project is **directive-strength classification**, where assistant messages are categorized as:  
- **Direct Command**  
- **Suggestion**  
- **No Instruction**  

Experiments were run on 50 multi-turn conversations sampled from Mistral’s production distribution using `mistral-small` at **temperature = 0** for reproducibility.  

---

## Repository Structure

```
llm-judge-prompt-optimization/
├── prompt-strategies/         # Seven prompting strategies
├── ensemble-approaches/       # Majority vote and self-consistency ensembles
├── prompt-optimization-loop/  # Automated prompt improvement framework
├── figures/                   # Plots and tables for report
├── data/                      # Datasets (events, annotations)
└── README.md                  # (this file)
```

---

## Modules Overview

### Prompt Strategies
- Tests **seven prompting approaches**: Open, Rubric, Closed, Chain-of-Thought (CoT), Role, Few-shot, CoT+Role+Few-shot.  
- Best single-judge result: **CoT+Role+Few-shot** (Accuracy 0.760).  
- **Key finding:** all strategies failed at detecting embedded instructions (*embedded instruction blindness*).  

➡ [Read more](./prompt-strategies/README.md)  

---

### Ensemble Approaches
- **Majority Vote**: aggregates results from multiple prompts.  
- **Self-Consistency**: aggregates multiple reasoning paths.  
- Cost ×4–13 higher than single judges, **no gains** due to shared systematic bias.  
- **Key finding:** ensembles do not overcome *embedded instruction blindness*.  

➡ [Read more](./ensemble-approaches/README.md)  

---

### Prompt Optimization Loop
- Automated loop for **evaluating → analyzing → improving → re-evaluating** judges.  
- Generated **enhanced judges** that fix embedded instruction blindness.  
- Accuracy gains of **+6 to +14 percentage points** across judge families.  
- Achieved **Direct Command F1 up to 0.571**, with only **1.5× cost**.  

➡ [Read more](./prompt-optimization-loop/README.md)  

---

## Key Insights

- **Prompt strategies** matter: structured methods outperform simple instructions.  
- **Ensembles** are expensive and do not fix systematic errors.  
- **Optimization loop** delivers the best trade-off: reliable improvements at low cost.  

---

## Running Experiments

Each subfolder (`prompt-strategies/`, `ensemble-approaches/`, `prompt-optimization-loop/`) contains its own `README.md` with detailed instructions.  
In general, the pipeline follows:  

1. **Annotate** conversations (`human_annotator.py`)  
2. **Evaluate** with judges (`judge_evaluator.py`)  
3. **Analyze** results (`judge_analyzer.py`)  
4. **Improve** prompts (only for optimization loop)  
