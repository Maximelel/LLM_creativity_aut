# LLM_creativity_aut
Comparing the creative outputs of LLMs on the Alternative Uses Task (AUT)

## Reproducibility

- Pipeline to generate new ideas with GPT models: `nb_script_LLM_main.ipynb`
  1. Call OpenAI API
  2. Call API OCSAI to evaluate
  3. Save ideas in dataframe

- Pipeline to generate new ideas with open source LLMs: see code on local server...

**Note:** For now generation of ideas is done one object at the time and one LLM at the time.

- General notebook to see benchmark results: `benchmark.ipynb`
  - Radar charts
    - Overall radar chart
    - Radar charts per object
    - Radar charts per model
  - Other plots
    - Univariate analysis: kdeplots, boxplots, violinplots
    - Multi-variate analysis: multi-kde, heatmaps
  - Further analysis
    - POS tagging
    - Topic modeling

