# UAUAR-GPHH
Genetic Programming Hyper-Heuristic (GPHH) for uncertain agricultural UAV arc routing with battery constraints.

---

## 1. License
This project is distributed under the **MIT License**.  
See the `LICENSE` file for full text.

---

## 2. Dependencies
- Python 3.8 or later
- `numpy`
- `deap`

Install dependencies:
```bash
pip install numpy deap
```

## 3. Repository Structure & File Description

| Folder/File | Description |
|-------------|-------------|
| `/GPHH` | Core GPHH implementation for UAUAR |
| `/H1` `/H2` `/H3` `/H4` `/H5` | Manual heuristic baselines |
| `/Improved Astar` | Improved A* baseline implementation |
| `/Simulation dataset` | Simulated benchmark datasets |
| `/real-world dataset` | Real-world farmland dataset |
| `README.md` | Project documentation |
| `LICENSE` | MIT License text |

---

## 4. Installation & Basic Usage

### 4.1 Installation
 Clone repository:
   ```bash
   git clone https://github.com/maxiangd/UAUAR-GPHH.git
   ```


### 4.2 Basic Usage
Run the main script:
```bash
python main.py
```

---

## 5. Core Method Pipeline

1. Generate uncertain training/test graphs from static graph data  
2. Evolve GP individuals as routing policies  
3. Evaluate each policy by UAV simulation (objective: total power consumption)  
4. Keep best policy   
5. Test best policy on unseen test dataset  

---

## 6. User Guide: Inputs / Outputs (Core)

### 6.1 Core Inputs
| Input | Description |
|------|-------------|
| `graph_structure` | Base graph (node-edge structure) |
| `MAX_BATTERY` | UAV battery capacity |
| `TRAIN_SEED_BASE` / `TEST_SEED_BASE` | Seeds for train/test split |
| GP params | population size, generations, crossover/mutation rates |

### 6.2 Core Outputs
| Output | Description |
|--------|-------------|
| `best_individual` | Best evolved GP policy |
| `train_fitness_log` | Evolution statistics per generation |
| `test_average_consumption` | Average power consumption on test set |

---


## 7. Notes on Reproducibility

- This project is stochastic; slight differences across runs are expected.
- Train/test seeds are separated to reduce data leakage.
- Multiprocessing is used for faster fitness evaluation.

---

## 8. Contact

For questions about code/paper, please contact:

- **Author**: Xiangdong Ma  
- **Email**: xiangdongma_181@stu.qau.edu.cn  
- **Affiliation**: College of Science and Information, Qingdao Agricultural University, Qingdao, China
