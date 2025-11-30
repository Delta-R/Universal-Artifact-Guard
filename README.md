#  Universal Artifact Guard (UAG)

> **A Robust Framework for Generalizable Deepfake Detection**

**Universal Artifact Guard (UAG)** is a general-purpose forgery detection framework for multi-source synthetic and real images. With the rapid iteration of various generative models, detectors trained on a single dataset or generator often suffer from a sharp performance drop when "switching to a different generator" or "switching to a different scene". The goal of UAG is to learn a **universal representation that is weakly correlated with specific generators but highly sensitive to forgery artifacts**, thereby maintaining stable detection capabilities across datasets, generators, and even unseen distributions.

---

## 🌟 Key Features

*   **🧠 Universal Representation Learning**: Explicitly models the difference between forgery artifacts and real content features, focusing on high-frequency details and texture distortions.
*   **🌐 Cross-Domain Generalization**: Introduces multi-source data mixing and differential contrast strategies to improve robustness across generators and scenes.
*   **⚖️ Structural Constraints**: Suppresses excessive reliance on semantic content to mitigate overfitting problems.
*   **🚀 Fully Open Source**: Provides a complete implementation from data preprocessing and training to inference.

---

## 🛠️ Installation

This project provides a complete environment configuration file. Please ensure that Anaconda or Miniconda is installed in your environment.

```bash
conda env create -f environment.yml
conda activate uag_env
```

---

## 📂 Code Functions

The code structure of this repository is carefully organized. Below are the functional descriptions of the main files and modules:

### 🚀 Core Workflow

*   **`train.py`**
    *   🔥 **Model Training Entry Point**. Responsible for initializing the model, loading data, and executing the training loop.
*   **`test.py`**
    *   📊 **Validation & Evaluation**. Used to evaluate model performance on the test set and calculate metrics such as accuracy.
*   **`inference_single.py`**
    *   🖼️ **Single Image Inference**. Performs rapid forgery detection on a single image and outputs the authenticity probability.
*   **`auto_stop.py`**
    *   🛑 **Early Stopping Mechanism**. Monitors validation set metrics to prevent model overfitting.

### ⚙️ Configuration

*   **`configuration/`**
    *   **`common_opts.py`**: Basic parameter configuration.
    *   **`training_opts.py`**: Training-specific parameters (e.g., learning rate, Batch Size).
    *   **`eval_opts.py`**: Testing and evaluation parameters.

### 🏗️ Architectures

*   **`architectures/`**
    *   Contains implementations of various backbone networks supported by UAG.
    *   **`clip_module/`**: Integrated CLIP-related model structures.
    *   **`vision_transformers.py` / `vit.py`**: Vision Transformer related implementations.
    *   **`resnet_model.py`**: ResNet backbone network.
    *   **`moco_legacy/`**: Support for early MoCo versions.

### 🧠 Core Networks

*   **`core_networks/`**
    *   **`training_engine.py`**: Encapsulates specific training logic and steps.
    *   **`abstract_model.py`**: Base model class definition.
    *   **`resnet_filter.py` / `low_pass.py`**: Network components involving filtering and specific frequency feature extraction.

### 💾 Data Processing

*   **`dataset/`**
    *   **`loader.py`**: Data loader, responsible for reading images and preprocessing.
*   **`data_settings.py`**: Dataset path and attribute configuration.
*   **`partition_dataset.py`**: Dataset partitioning script (train/validation/test).
*   **`update_labels.py`**: Label update and correction tool.

### 📦 Checkpoints

*   **`checkpoints/`**
    *   Used to store trained model weight files (e.g., `best_epoch_model.pth`).

---

## 🚀 Quick Start

1.  **Configure Environment**: Refer to the installation steps above.
2.  **Prepare Data**: Modify `data_settings.py` to configure your data paths and use `partition_dataset.py` for partitioning.
3.  **Start Training**:
    ```bash
    python train.py --name experiment_uag --gpu_ids 0
    ```
4.  **Model Evaluation**:
    ```bash
    python test.py --name experiment_uag --checkpoint_dir checkpoints/
    ```
5.  **Single Image Test**:
    ```bash
    python inference_single.py --image_path path/to/image.jpg
    ```

---

## 📝 Introduction

In terms of overall structure, UAG uses common Convolutional Networks or Vision Transformers as feature extraction backbones, on top of which it explicitly models the differences between **forgery artifact features** and **real content features**. Through specialized branches and constraints, the framework encourages the network to focus on forgery clues that are difficult for the human eye to perceive directly, such as high-frequency details, texture distortions, and statistical inconsistencies.

To improve cross-generator and cross-domain generalization capabilities, UAG introduces multi-source data mixing and differential contrast strategies during training: on the one hand, the model is jointly optimized on samples from different synthesis methods and various resolution/quality conditions; on the other hand, by explicitly pulling closer the "structural similarity between real images" and pushing away the "artifact differences between real and forged samples" in the feature space, the forgery discrimination boundary learned by the model becomes more robust.


