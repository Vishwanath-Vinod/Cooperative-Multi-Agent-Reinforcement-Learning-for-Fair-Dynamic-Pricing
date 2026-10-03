# FairSwarm: A Cooperative Multi-Agent Reinforcement Learning for Fair Dynamic Pricing

Official implementation of **“Cooperative Multi-Agent Reinforcement Learning for Fair Dynamic Pricing”**, presented at **The 14th Computing Conference 2026**.

Work done by Vishwanath Vinod under the guidance of Professor Rachel Kalpana Kalaimani for an Undergraduate Research Project at IIT Madras.

---
## Paper
*Title*:  Cooperative Multi-Agent Reinforcement Learning for Fair Dynamic Pricing \\
*Authors:* Vishwanath Vinod and Rachel Kalpana Kalaimani \\
*Institution:* Indian Institute of Technology Madras \\
*Conference:* The 14th Computing Conference 2026, London, United Kingdom, 9–10 July 2026 \\
*Link to Paper:* Computing Conference Proceedings \href{https://link.springer.com/chapter/10.1007/978-3-032-24804-6_11}{(here)}

---

## Overview

Dynamic pricing allows multiple agents to adjust prices in response to changing demand and inventory conditions. However, optimizing each agent independently can lead to significant disparities in the resulting profits.

This work proposes a **cooperative multi-agent reinforcement learning framework** that incorporates fairness directly into the learning process.

The proposed approach is based on a **decomposed MADDPG architecture**, where agents have access to:

* Local critics that capture individual profit.
* A global fairness critic that captures the fairness objective.
* A shared cooperative learning objective that balances profitability and fairness.

The experiments evaluate the approach under different numbers of agent groups and compare the resulting pricing, profit, and fairness behaviour.

---

## Method

The environment consists of multiple customer groups and pricing agents. Each agent dynamically selects prices based on the current market state.

The learning framework combines:

1. **Individual Profit Objective**
   Each agent optimizes its own revenue/profit.

2. **Global Fairness Objective**
   A fairness objective based on the **Jain's Fairness Index** is incorporated to encourage a more equitable distribution of profits.

3. **Cooperative Multi-Agent Learning**
   The individual and global objectives are jointly used within a decomposed MADDPG framework.

### Architecture

<p align="center">
  <img src="images/architecture.png" width="800">
</p>

*Overview of the proposed cooperative multi-agent reinforcement learning framework.*

> Replace `images/architecture.png` with the relevant figure from the paper.

---

## Results

The framework is evaluated using simulated dynamic-pricing environments with varying numbers of customer groups.

<p align="center">
  <img src="images/results.png" width="800">
</p>

*Example results from the paper showing the relationship between profitability and fairness.*

Additional figures from the paper can be added to the `images/` directory and referenced here.

---

## Repository Structure

```text
.
├── README.md
├── customer.py
├── dec_maddpg_dual.py
├── demand.py
├── env.py
├── fairness_metric.py
├── main.py
├── networks.py
├── noise.py
├── plot_fairness.py
└── replay_buffer.py
```

### File Descriptions

| File                 | Description                                       |
| -------------------- | ------------------------------------------------- |
| `main.py`            | Main script for training and running experiments  |
| `env.py`             | Dynamic-pricing environment                       |
| `customer.py`        | Customer and customer-group modelling             |
| `demand.py`          | Demand-generation and demand modelling            |
| `dec_maddpg_dual.py` | Decomposed MADDPG implementation                  |
| `networks.py`        | Actor and critic network definitions              |
| `replay_buffer.py`   | Experience replay buffer                          |
| `noise.py`           | Exploration noise used during training            |
| `fairness_metric.py` | Fairness metrics, including Jain's Fairness Index |
| `plot_fairness.py`   | Utilities for generating fairness-related plots   |

---

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not included, install the dependencies required by the Python files in the repository.

---

## Running the Experiments

The main training script is:

```bash
python main.py
```

Experiment parameters such as the number of groups, training episodes, pricing range, inventory, and customer population can be configured in the corresponding scripts.

The experiments reported in the paper consider different numbers of customer groups, including:

```text
2 groups
4 groups
6 groups
```

The experimental setup uses simulated customer populations and inventory levels proportional to the number of groups.

---

## Generating Fairness Plots

After training, fairness-related results can be visualized using:

```bash
python plot_fairness.py
```

The plotting script can be adapted to visualize the fairness and profitability results obtained from the trained agents.

---

## Experimental Setup

The experiments use the following general configuration:

| Parameter         | Setting                |
| ----------------- | ---------------------- |
| Number of groups  | 2, 4, 6                |
| Pricing range     | 100–500                |
| Customers         | 600 × number of groups |
| Initial inventory | 300 × number of groups |
| Training episodes | 1000                   |
| Random seeds      | 3                      |

The exact experimental configuration and implementation details are described in the paper.

---

## Paper

**Vishwanath Vinod and Rachel Kalpana Kalaimani.**
*Cooperative Multi-Agent Reinforcement Learning for Fair Dynamic Pricing.*
The 14th Computing Conference 2026, London, United Kingdom, 9–10 July 2026.

### Citation

If you use this repository or the proposed methodology in your work, please cite:

```bibtex
@inproceedings{vinod2026cooperative,
  title     = {Cooperative Multi-Agent Reinforcement Learning for Fair Dynamic Pricing},
  author    = {Vinod, Vishwanath and Kalaimani, Rachel Kalpana},
  booktitle = {Proceedings of the 14th Computing Conference},
  year      = {2026},
  address   = {London, United Kingdom}
}
```

A DOI or publisher link can be added here once available.

---

## Acknowledgements

This work was carried out by **Vishwanath Vinod** under the guidance of **Prof. Rachel Kalpana Kalaimani** at the **Indian Institute of Technology Madras**, as part of an undergraduate research project.

---

## License

Add the repository license here, for example:

```text
MIT License
```

if the repository is released under the MIT License.

