# LLM_creativity_aut

Semester Project in the REACT Group at EPFL (Feb. 2024 - June 2024)

## 🌐 Overview

### 📚 Context: Enhancing Creativity in Education

- In recent years, AI-powered chatbots, particularly large language models (LLMs) such as GPT models, have transformed various domains, including education. These models not only provide information and answer queries but also have the potential to foster creativity among students by offering diverse perspectives and generating numerous ideas. This capability is particularly valuable in educational settings, where encouraging creative thinking is essential for developing problem-solving skills and innovation.
- Creative thinking involves both divergent and convergent phases. This project explores how LLMs can assist students in both phases, widening their ideation space and supporting them in evaluating creative ideas.
- To assess LLMs' creative potential, we utilize the Alternative Uses Test (AUT), a widely recognized measure of creative thinking.

### 🚧 Challenge: Measuring Creativity
- Assessing creativity is inherently challenging due to its subjective nature and traditional methods can be time-consuming and expensive.

### ✨ Solution Proposed: Leveraging LLMs for Creative Ideation and NLP techniques for Automated Evaluation
- Designed a comprehensive benchmark and an automated evaluation pipeline to assess the creativity of ideas generated on the AUT.
- Tested different prompting strategies with LLMs

## 🔍 Research Question

If and how can LLMs assist students to be more creative?

## 📊 Methodology
- **Benchmark and Evaluation Pipeline:**

  - Designed to assess creativity using the AUT.
  - Evaluates multiple creativity dimensions, including originality, elaboration, dissimilarity, and flexibility.
  - Provides a framework for comparing human and LLM-generated responses.
    
- **Prompting Strategies:**

  - Experimented with various LLMs (closed and open-source) using different prompting strategies.
  - Analyzed the impact of prompt structures on the creativity of generated ideas.

## 🔁 Reproducibility

- **Pipeline to generate new ideas with GPT models**: `nb_script_LLM_main.ipynb`
  1. Call OpenAI API
  2. Call API OCSAI to evaluate. See [website](https://openscoring.du.edu/scoringllm) and their [API](https://openscoring.du.edu/docs)
  4. Save ideas in dataframe

- **Pipeline to generate new ideas with open source LLMs**: see code on local server

**Note:** For now, generation of ideas is done **one object at a time** and **one LLM at a time**. All ideas generated during this project can be found in the folder `dataset`.

- **General notebook to see benchmark results of GPT models**: `prompts_analysis.ipynb`
    - Univariate analysis
    - Multi-variate analysis
  - 🔍 **Further analysis**
    - POS tagging
    - Topic modeling
    - Open source models
    - Consistency check

The full report of the project can be found [here](https://github.com/Maximelel/LLM_creativity_aut/blob/main/pdf_report.pdf).
