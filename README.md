# IGANN-IT Model

This project builds on the original **IGANN model** by *Mathias Kraus* ([GitHub Repository](https://github.com/MathiasKraus/igann)).  
Reference:  
Kraus, M., Tschernutter, D., Weinzierl, S., & Zschech, P. (2024). *Interpretable generalized additive neural networks*. *European Journal of Operational Research*, 317(2), 303–316. [https://doi.org/10.1016/j.ejor.2023.06.032](https://doi.org/10.1016/j.ejor.2023.06.032)

---

The model has been extended to **IGANN-IT** (Interpretable Generalized Additive Neural Network with Interaction Terms).  

The implementation is provided in the `igann/` package.  
The module `igann_core.py` contains the extended version of IGANN with **interaction terms (IT)**.  
All modifications are marked with `# Veitl:` in the code.

---

This repository is available at: [https://github.com/MaximilianVeitl/igann-it](https://github.com/MaximilianVeitl/igann-it)

---

## Repository Structure
- 01_demo_igann_it_final.ipynb # Demo of IGANN-IT on Canadian auto insurance use case
- 02_igann_it_evaluation_final.ipynb # Evaluation of IGANN-IT on 12 datasets
- 03_survey_evaluation_final.ipynb # Evaluation of user study on interpretability

- igann/ # IGANN-IT model implementation
    - init.py
    - igann_core.py # IGANN-IT core model (with interaction terms)

- README.md # Project description
- setup.py # Package setup

---

## Dataset for Demo

- Dataset: [catelematic13](https://www2.math.uconn.edu/~valdez/data.html)  
- Reference:  
  Banghee So, Jean-Philippe Boucher, and Emiliano A. Valdez (2021), *Synthetic Dataset Generation of Driver Telematics*, *Risks* 9:58, [doi:10.3390/risks9040058](https://doi.org/10.3390/risks9040058)

---

## Datasets for Evaluation

The datasets used for the evaluation of IGANN-IT (in `02_igann_it_evaluation_final.ipynb`) are based on the datasets provided by the [whiteboxai/datasets](https://github.com/whiteboxai/datasets) repository.  

Link: [https://github.com/whiteboxai/datasets](https://github.com/whiteboxai/datasets)
