# 🧪 Alchemy Perceptron  
**Molecular Property Prediction from Scratch using a Multi-Layer Perceptron**

---

## 📌 Overview

This repository presents a **from-scratch implementation of a multi-layer perceptron (MLP)** for **molecular property prediction**, applied to the **QM9 quantum chemistry dataset**.

The goal is to predict the **HOMO–LUMO energy gap** of molecules using:
- handcrafted molecular representations (Morgan fingerprints)
- a neural network implemented *without* deep-learning frameworks
- explicit forward propagation, backpropagation, batching, and optimization

This project is designed as a **scientific baseline** and a stepping stone toward:
- Graph Neural Networks (GNNs)
- Convolutional Neural Networks (CNNs)

---

## 🎯 Scientific Objective

Given a molecule represented by its chemical structure, we aim to learn a function:

$$\[f(\text{molecule}) \rightarrow \Delta E_{\text{HOMO–LUMO}}\]$$

where:
- $$\( \Delta E_{\text{HOMO–LUMO}} \)$$ is the energy gap between the highest occupied and lowest unoccupied molecular orbitals
- this quantity is physically meaningful and relates to molecular stability and electronic behavior

The focus of this project is **not just prediction accuracy**, but:
- understanding *how* neural networks learn from molecular data
- identifying the limitations of non-graph representations
- motivating more expressive models

---

## 🧠 Neural Network Model

### Architecture

The model is a **fully connected feedforward neural network**:

Input (2048) → 512 → 128 → 64 → Output (1)


- Hidden layers: ReLU activation  
- Output layer: Linear activation  
- Loss function: Mean Squared Error (MSE)  
- Optimizer: Mini-batch Stochastic Gradient Descent  

All components are implemented manually using **NumPy only**.

---

### Neuron-Level Computation

For a neuron $$\( i \)$$ in layer $$\( l \)$$:

$$\[z_i^{(l)} = \sum_j W_{ij}^{(l)} a_j^{(l-1)} + b_i^{(l)}\]$$

$$\[a_i^{(l)} = g(z_i^{(l)})\]$$

where:
- $$\( W^{(l)} \)$$ is the weight matrix
- $$\( b^{(l)} \)$$ is the bias vector
- $$\( g(\cdot) \)$$ is the activation function

---

### Activation Function (ReLU)

$$\[\text{ReLU}(z) = \max(0, z)\]$$

$$\[\frac{d}{dz}\text{ReLU}(z) =\begin{cases}1 & if\ z > 0 \, 0 \ if & z \le 0\end{cases}\]$$

---

### Loss Function (MSE)

$$\[\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2\]$$

---

## 🧬 Dataset: QM9 (Alchemy Context)

This project uses the **QM9 dataset**, a standard benchmark in molecular machine learning.

### Dataset characteristics:
- ~130,000 small organic molecules
- Properties computed using Density Functional Theory (DFT)
- Molecules provided as SMILES strings
- Multiple quantum-chemical targets

### Target used in this project:
- **HOMO–LUMO energy gap**

QM9 is widely used in:
- molecular ML benchmarks
- graph neural network research
- computational chemistry studies

---

## 🧪 Molecular Representation: Morgan Fingerprints

Neural networks require fixed-size numerical inputs.  
Each molecule is converted into a **Morgan fingerprint** using RDKit.

### What is a Morgan Fingerprint?

Morgan fingerprints (ECFP) are circular molecular descriptors that:
- encode local atomic environments
- capture bonding patterns and substructures
- produce a fixed-length vector representation

In this project:
- Radius = 2  
- Vector size = 2048  

molecule → x ∈ R^2048

---

### Why Fingerprints?

Morgan fingerprints provide:
- a simple and efficient baseline
- a well-established representation in cheminformatics
- a clear contrast to graph-based methods

However, they **do not encode**:
- explicit atom–atom relationships
- molecular geometry
- long-range electronic interactions

This limitation motivates future **graph-based models**.

---

## 🔄 Data Processing Pipeline

1. Load QM9 CSV data
2. Parse SMILES strings using RDKit
3. Generate Morgan fingerprints
4. Normalize target values:
$$\[y_{\text{norm}} = \frac{y - \mu}{\sigma}\]$$
5. Train / test split (80% / 20%)
6. Mini-batch training loop

---

## 📥 Dataset Download

This project uses the **QM9 molecular dataset**, a standard benchmark in molecular machine learning.

### Download Instructions

1. Download the QM9 dataset (CSV format) from an official source:

   https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv

2. Place the downloaded file in the project root directory or inside a `data/` folder:

alchemy_perceptron/
├── data/
│ └── qm9.csv
├── main.py
├── mlp_model.py
└── README.md

3. Ensure the path to `qm9.csv` in `main.py` matches the file location.

> **Note:**  
> The full QM9 dataset is **not included** in this repository due to size and licensing considerations.

## ▶️ How to Run the Code

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/alchemy_perceptron.git
cd alchemy_perceptron
```
### 2. Install Dependencies

This project requires Python 3.8+ and the following packages:
```bash
pip install numpy pandas matplotlib rdkit-pypi
```
### 3. (Optional) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```
### 4. Run the Training Script
```bash
python main.py
```

## 📊 Results

### Training Behavior

- Smooth and monotonic loss decrease
- Stable gradients
- No divergence or numerical instability

**Training Loss Curve:**
> *(Insert loss curve plot here)*

---

### Performance

- **Test RMSE ≈ 0.50 (normalized)**
- Strong linear correlation between predictions and targets
- Expected degradation at extreme values

---

### Parity Plot

The parity plot below compares predicted vs actual HOMO–LUMO gaps.

> *(Insert parity plot image here)*

Observed behavior:
- Predictions cluster around the ideal $$\( y = x \)$$ line
- Symmetric scatter
- Increased error for rare molecular configurations

This behavior is **expected for fingerprint-based MLP models** and reflects known representational limits.

---

## ⚠️ Limitations

- No explicit molecular graph structure
- No geometric (3D) information
- Limited expressivity for long-range interactions

These limitations naturally motivate **Graph Neural Networks** as the next step.

---

## 🚀 Future Work

Planned extensions include:
- Message-passing Graph Neural Networks on QM9
- Comparison: MLP vs GNN performance
- CNN-based models for astronomical datasets (MiraBest)
- Probabilistic neural networks for uncertainty estimation
- Stochastic system modeling

---

## 🛠️ Technologies Used

- Python
- NumPy
- RDKit
- Matplotlib
- Custom neural network implementation (no ML frameworks)

---

## 📚 References

- QM9 Dataset  
- Morgan Fingerprints (ECFP)  
- Deep Learning for Molecular Property Prediction  
- Graph Neural Networks in Chemistry  

---

## ✨ Final Remarks

This project serves as a **scientifically grounded baseline**, demonstrating:
- a deep understanding of neural network internals
- careful handling of real quantum-chemical data
- clear motivation for more expressive learning architectures

It is intended as a foundation for more advanced research-oriented models.
