# FIRMBOUND
This repository contains the official PyTorch implementation of **FIRMBOUND**, a framework for **optimal early classification of sequential data** under finite-horizon constraints. **FIRMBOUND** builds upon the **Sequential Probability Ratio Test (SPRT)** but extends it to handle **finite decision deadlines** by learning dynamic, time-dependent decision thresholds.

<div align="center">
<figure>
<img src="./figures/Closing_Boundary_4github.GIF" width="70%">  
</figure>
</div>

<p align="center">(Wait for a while to load) Figure 1: FIRMBOUND compared with the classic SPRT.</p>

---

## Key Features
- **Finite-Horizon Decision Making:** Unlike traditional SPRT, which assumes an infinite time horizon, **FIRMBOUND** adapts decision thresholds dynamically within a predefined limit.
- **Convex Function Learning (CFL) & Gaussian Process (GP) Regression:** Provides two approaches for solving backward induction to estimate optimal stopping rules.
- **Density Ratio Estimation (DRE):** Uses **SPRT-TANDEM** to estimate log-likelihood ratios (LLRs) consistently.
- **Extensive Benchmarking:** Supports synthetic (Gaussian) and real-world, non-i.i.d., multiclass datasets (e.g., SiW, HMDB51, UCF101).
- **PyTorch Implementation:** Modular and efficient code for training and inference.

## Quickstart Guide

### 1. **Generate Sequential Gaussian Dataset**
Run `generate_sequential_gaussian_as_lmdb.ipynb` to create train, validation, and test LMDB datasets.
- Modify the header section to specify dataset parameters.
- Repeat three times to generate different splits.


### 2. Train the Log-Likelihood Ratio (LLR) Estimator
Run `density_ratio_estimation_main.py` to use the SPRT-TANDEM framework to estimate log-density ratios. 
- Update `./config/config_dre.py` to include paths to the LMDB datasets before running.
- For more details, check:
  - [SPRT-TANDEM-PyTorch](https://github.com/Akinori-F-Ebihara/SPRT-TANDEM-PyTorch)
  - [SPRT-TANDEM tutorial](https://github.com/Akinori-F-Ebihara/SPRT-TANDEM_tutorial)  

### 3. Train Convex Function Learning (CFL) or Gaussian Process (GP) regression to estimate the optimal thresholds
Run either `backward_induction_CFL_train.py` or `backward_induction_GP_train.py` to use respectively.
- Update the following parameters:
  - `savedir`: Directory to save model results.
  - `subproject`: Name matching Step 2 output.
  - `cost_pool`: Set the sampling cost value.

### 4. Test the Model
Run inference on the generated dataset by either `backward_induction_GP_test.py` or `backward_induction_CFL_test.py`.

## Tested Environment
```
python      3.8.10
torch       2.0.0
notebook    6.5.3
optuna      3.1.0
```

## Example results
<div align="center">
<figure>
<img src ="./figures/Binary_optimal_boundary.png" width=70%>  
</figure>
</div>
<p align="center">Figure 2: Two-class Gaussian dataset. The estimated optimal decision threshold is shown in orange. The red and blue trajectories represent different classes. Correct decision is made when the red and blue curves hit the positive or negative side of the decision threshold, respectively.</p>


<div align="center">
<figure>
<img src ="./figures/rotation_animation_gray.gif" width=70%>  
</figure>
</div>
<p align="center">Figure 3: Three-class Gaussian dataset. The estimated optimal decision threshold is depicted as a yellow envelope. The stopping points for different classes (red, blue, yellow) are marked with black crosses.</p>

## Citation
If you use this code, please cite our paper:

```
@inproceedings{FIRMBOUND,
  title={Learning the Optimal Stopping for\\Early Classification within Finite Horizons\\via Sequential Probability Ratio Test},
  author={Akinori F Ebihara and Taiki Miyagawa and Kazuyuki Sakurai and Hitoshi Imaoka},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=SRghq20nGU}
}
```
