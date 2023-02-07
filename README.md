# REGNUM

This repository contains the data and implementation of the submission  **REGNUM**: **Generating Logical Rules with Numerical Predicates in Knowledge Graphs.**

## Requirements
<details>
<summary>Install Stardog, the instructions can be found via:</summary>
    
    [https://docs.stardog.com/get-started/install-stardog/](https://docs.stardog.com/get-started/install-stardog/)
</details>
<details>

<summary>An uncommercial free license can be requested via:</summary>
    
    [Stardog Trial License Request | Stardog](https://www.stardog.com/license-request/)

</details>
<details>

<summary>Run the following to set up your virtual environment and install the Python requirements:</summary>

    python3.8 -m venv regnum_env
    source regnum_env/bin/activate
    pip install -r requirements.txt
    
</details>
<details>

<summary>Install Java >= 8</summary>
</details>
## Quick start

Start from the run_process.py and pass arguemnt to enrich the rules

For ex:
    
    python src/run_process.py
    --train_path data/datasets/sample_small/train_dl.tsv
    --numerical_path data/datasets/sample_small/numericals.tsv
    --path_result data/results/result_sample

ARGS:

    --train_path :
    --numerical_path :
    --path_result :
    --MINHC :
    --MINCONF :
    
    
