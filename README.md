# Auto_model

**Auto_model** is a collection of Python scripts for **automatic prompt generation**, **prompt combination**, and **model training**.  
It is designed to support rapid experimentation with different prompt strategies in NLP / LLM-related tasks (e.g., Text-to-SQL, QA, or other prompt-driven learning settings).

---

## 📁 Repository Structure

```text
Auto_model/
├── Auto_Prompt_No_pf.py              # Auto-prompt without prefix filtering
├── Auto_Prompt_least_prompt.py       # Minimal prompt strategy
├── Auto_Prompt_multiple.py           # Multiple prompt generation strategy
├── Auto_Prompt_no_schema_filter.py   # Auto-prompt without schema filtering
├── Auto_Prompt_no_schema_link.py     # Auto-prompt without schema linking
├── prompt.py                         # Prompt templates and helper functions
├── train_fusion.py                   # Prompt fusion training script
├── train_fusion_2.0.py               # Fusion training (v2.0)
├── train_pro                         # Additional training scripts or configs
└── README.md
