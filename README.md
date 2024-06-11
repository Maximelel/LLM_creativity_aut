# LLM_creativity_aut
**Comparing the creative outputs of LLMs on the Alternative Uses Task (AUT)**

## 🔁 Reproducibility

- **Pipeline to generate new ideas with GPT models**: `nb_script_LLM_main.ipynb`
  1. Call OpenAI API
  2. Call API OCSAI to evaluate
See [website](https://openscoring.du.edu/scoringllm) and their [API](https://openscoring.du.edu/docs)
  4. Save ideas in dataframe

- **Pipeline to generate new ideas with open source LLMs**: see code on local server...

**Note:** For now, generation of ideas is done **one object at a time** and **one LLM at a time**.

- **General notebook to see benchmark results**: `prompts_analysis.ipynb`
  - 📊 **Radar charts**
    - Overall radar chart
    - Radar charts per object
    - Radar charts per model
  - 📈 **Other plots**
    - Univariate analysis
    - Multi-variate analysis
  - 🔍 **Further analysis**
    - POS tagging
    - Topic modeling

Enjoy exploring the creative potential of LLMs! 🚀
