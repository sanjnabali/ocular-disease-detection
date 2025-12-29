

---

# Ocular Disease Detection (Cataract) using Deep Learning

This project implements a Deep Learning solution to detect cataracts from retinal fundus images. It utilizes transfer learning with the **EfficientNetB0** architecture to classify images into "Normal" or "Cataract" categories. The model is trained on a combined dataset sourced from ODIR-5K and the Kaggle Cataract Dataset.

## 🚀 Overview

Cataracts are a leading cause of vision loss worldwide. Early detection through automated screening of fundus images can significantly assist healthcare providers. This pipeline includes:

* Data integration from multiple clinical sources.
* Handling class imbalance through **downsampling**.
* Image preprocessing (Resizing to 192x256 and Normalization).
* Transfer learning using **EfficientNetB0** for feature extraction and classification.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning Framework:** TensorFlow, Keras
* **Image Processing:** OpenCV
* **Data Analysis:** Pandas, Numpy
* **Visualization:** Matplotlib, Seaborn

## 📊 Dataset Detail

The project uses two primary datasets to ensure diversity:

1. **ODIR-5K:** Ocular Disease Intelligent Recognition dataset.
2. **CataractDataset:** A focused dataset for cataract classification.

**Class Balancing:**
Initial data showed a heavy bias toward "Normal" eyes (over 3,300 images per eye vs. ~150 cataract cases). To mitigate this, a **downsampling** function was applied to ensure the model does not become biased toward the majority class.

## 🏗️ Model Architecture

The model is built using the **EfficientNetB0** backbone:

* **Input Shape:** (192, 256, 3)
* **Preprocessing:** All pixels are normalized to the [0, 1] range.
* **Fine-Tuning:** Uses a Sequential model adding GlobalAveragePooling2D and a Dense output layer with Softmax activation.
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy

## 📈 Training Performance

* **Epochs:** 100
* **Batch Size:** 32
* **Splits:** 80% Training, 20% Testing (with 15% of training used for validation).

## 💻 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/sanjnabali/ocular-disease-detection.git
cd ocular-disease-detection

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Download Data:**
Ensure you have your `kaggle.json` API key to download the ODIR-5K and Cataract datasets as scripted in the notebook.

## 📜 License

This project is licensed under the MIT License.

---

**Author:** Sanjna Bali
**Email:** sanjnabali8@gmail.com
