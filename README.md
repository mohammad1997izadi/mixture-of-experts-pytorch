# Mixture of Experts (MoE) in PyTorch

This project implements a modular and extensible Mixture of Experts (MoE) architecture using PyTorch, trained on the MNIST dataset.

## 📁 Project Structure

```
moe_project/
├── data/              # MNIST dataset storage
├── models/            # Trained model output
├── scripts/           # Modular training and evaluation scripts
├── README.md
├── requirements.txt
└── main.py            # Entry point
```

## ⚙️ Features

- Fully automated pipeline
- Clean modular design with separate concerns
- Visual training metrics (loss and accuracy)
- Easily extendable number of experts
- GPU acceleration supported

## 🧠 Architecture

- **Expert Networks**: MLPs trained on MNIST.
- **Gating Network**: Learns weights for combining expert outputs.
- **Mixture Output**: Weighted sum of expert outputs.

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python main.py
```

## 📈 Output Example

Training progress is visualized with loss and accuracy plots saved to `models/`.

## 📜 License

Licensed under the MIT License.
