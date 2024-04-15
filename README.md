# Optimizing the preventive maintenance frequency with causal machine learning </br><sub><sub>T. Vanderschueren, R. Boute, T. Verdonck, B. Baesens, W. Verbeke [[2023]](https://doi.org/10.1016/j.ijpe.2023.108798)</sub></sub>

<div align="justify">
We propose a prescriptive framework for preventive maintenance optimization. We leverage causal inference and machine learning to learn the effect of preventive maintenance on an asset's overhaul and failure rates, based on asset-specific characteristics, from observational data on assets that were maintained in the past. The learned effects allow for finding each asset's optimal preventive maintenance frequency, minimizing the combined cost of failures, overhauls, and preventive maintenance. We compare an individualized, causal approach to non-individualized and non-causal ablations.
</div>

## Repository structure
This repository is organised as follows:
```bash
|- scripts/
    |- main_IJPE.py
    |- main.py
|- src/
    |- data/
        |- generate_data.py
    |- methods/
        |- nn_supervised.py
        |- SCIGAN.py
    |- utils/
        |- evaluation_utils.py
        |- evaluation.py
        |- model_utils.py
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Disclaimer
Due to reasons of confidentiality, we unfortunately cannot share raw data and related preprocessing code. We provide comments in the code to guide researchers and practitioners interested in applying our methodology to their own data.

## Acknowledgements
We build upon the code from [SCIGAN](https://github.com/ioanabica/SCIGAN/tree/main) [1]. 

[1] Bica, I., Jordon, J., & van der Schaar, M. (2020). Estimating the effects of continuous-valued interventions using generative adversarial networks. _Advances in Neural Information Processing Systems_, 33, 16434-16445.

## Citing
Please cite our paper and/or code as follows:

```tex

@article{vanderschueren2023maintenance,
  title={Optimizing the preventive maintenance frequency with causal machine learning},
  author={Vanderschueren, Toon and Boute, Robert and Verdonck, Tim and Baesens, Bart and Verbeke, Wouter},
  journal={International Journal of Production Economics},
  volume={258},
  pages={108798},
  year={2023},
  publisher={Elsevier}
}


```
