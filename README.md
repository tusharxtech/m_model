# 🦠 Malaria Detection using Deep Learning (CNN)

## 📖 Overview

This project implements a **Deep Learning-based Malaria Detection System** using Convolutional Neural Networks (CNNs). The model classifies cell images as:

* 🧬 Parasitized (Infected)
* 🧪 Uninfected (Healthy)

The dataset is sourced from:

* Hugging Face (`sulaimank/malaria`)
* TensorFlow Datasets (`malaria`)

---

## 🚀 Features

* Image classification using CNN
* TensorFlow + Keras pipeline
* Data preprocessing with `tf.data`
* Batch training with prefetch optimization
* Visualization of predictions
* Model saving for deployment

---

## 📂 Dataset

### Sources:

* Hugging Face Dataset: `sulaimank/malaria`
* TensorFlow Dataset: `tfds.load('malaria')`

### Classes:

* **Parasitized (0)**
* **Uninfected (1)**

---

## ⚙️ Installation

```bash
pip install tensorflow tensorflow-datasets datasets matplotlib numpy
```

---

## 🧠 Model Architecture

The CNN model consists of:

* Input Layer (255x255x3)
* Convolution Layers
* Batch Normalization
* Max Pooling
* Fully Connected Dense Layers
* Output Layer (Sigmoid)

---

## 🔄 Data Pipeline

1. Load dataset from Hugging Face / TFDS
2. Split into:

   * 80% Training
   * 10% Validation
   * 10% Testing
3. Resize images to 255x255
4. Normalize pixel values (0–1)
5. Batch & prefetch for performance

---

## 🏋️ Training

```python
malaria_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
```

---

## 📊 Evaluation

```python
malaria_model.evaluate(test_dataset)
```

---



---

## 💾 Model Saving

```python
malaria_model.save("malaria_model.keras")
```

---

## 📈 Example Output

* Binary classification (0 or 1)
* Sigmoid probability threshold = 0.5

---

## 🧪 Future Improvements

* Use Transfer Learning 
* Hyperparameter tuning



---

## ⚠️ Known Issues / Improvements Needed

* Mixing Hugging Face dataset and TFDS dataset (redundant)
* Inefficient dataset splitting method
* Hardcoded image size (255 instead of 224)
* High learning rate (0.01 may cause instability)
* No early stopping or callbacks

---

## 🤝 Contributing

Pull requests are welcome! Open an issue first for major changes.

---


