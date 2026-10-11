* Reliability-Aware Multimodal Anomaly Detection
A multimodal industrial anomaly detection system that investigates how vision and textual inspection information can be fused reliably under missing, corrupted, and contradictory modalities.
Overview
Industrial inspection systems often have access to multiple information sources, such as images and textual inspection descriptions. However, these modalities may be incomplete, corrupted, or inconsistent.
This project evaluates multimodal anomaly detection under these conditions and implements a reliability-aware fusion approach that dynamically estimates the contribution of image and text representations.
The project uses the MMAD (Multi-Modal Anomaly Detection) dataset, with experiments conducted on the DS-MVTec subset.
Research Objective
The project compares:
- Vision-only anomaly/question classification
- Text-only classification
- Standard multimodal fusion
- Reliability-aware multimodal fusion
The multimodal systems are further evaluated under:
- Missing image
- Missing text
- Corrupted image
- Contradictory text
The objective is to determine whether reliability-aware multimodal fusion provides more robust predictions when one or more modalities become unreliable.

** Architecture
                    Industrial Image
                          │
                          ▼
                    ResNet-18
                    Image Features
                       512-D
                          │
                          ├──────────────┐
                          │              │
                          ▼              ▼
                   Image Reliability   Weighted
                       Estimator       Features
                          │              │
                          │              │
                          │         ┌────┘
                          │         │
                          ▼         ▼
                    Reliability-Aware
                         Fusion
                          ▲
                          │
                          │
                    Weighted Features
                          │
                          ▲
                          │
                     DistilBERT
                    Text Features
                       768-D
                          ▲
                          │
                Question + Options

** Standard Multimodal Fusion
Image and text representations are extracted independently and concatenated:

ResNet-18 (512) + DistilBERT (768)
                ↓
             1280-D
                ↓
          MLP Classifier
                ↓
             A / B / C / D

** Reliability-Aware Fusion
Separate reliability estimators produce image and text reliability scores. These scores are normalized and used to weight the corresponding modality representations before classification.
*** Dataset
The project uses the MMAD dataset and focuses on its DS-MVTec subset.
Dataset characteristics used in the project:
- 8,338 question-image samples
- 1,691 unique images
- 4 answer classes: A, B, C, D
- Industrial inspection imagery
- Question and multiple-choice textual information
The dataset was split at the image level to prevent questions belonging to the same image from appearing across different splits.
Models
Vision-only
- ResNet-18
- ImageNet pretrained backbone
- Frozen convolutional backbone
- 4-class classification head
Text-only
- DistilBERT
- Question + options as input
- 4-class classification head
Standard Multimodal
- ResNet-18 image encoder
- DistilBERT text encoder
- Feature concatenation
- MLP classifier
Reliability-Aware Multimodal
- ResNet-18 image encoder
- DistilBERT text encoder
- Image reliability estimator
- Text reliability estimator
- Reliability-weighted feature fusion
- MLP classifier

Results
Normal Test Set
Model	Accuracy	Macro F1	Weighted F1
Vision-only	30.46%	—	26.10%
Text-only	78.02%	78.34%	78.04%
Standard Multimodal	70.18%	70.14%	70.10%
Reliability-Aware Multimodal	76.58%	76.83%	76.55%


Robustness Evaluation
Condition	Standard Accuracy	Reliability-Aware Accuracy
Normal	70.18%	76.58%
Missing Image	70.82%	76.58%
Missing Text	31.41%	20.22%
Corrupted Image	70.82%	76.58%
Contradictory Text	26.38%	23.10%


The experiments show that reliability-aware fusion improves normal-condition multimodal performance, but its robustness is not uniformly better across all perturbation types.
In particular, the learned reliability mechanism strongly favors the text modality in the current implementation. This is an important finding rather than being treated as evidence that the reliability mechanism is universally robust.
Reliability Analysis
Average learned modality weights under perturbations:
Condition	Image Weight	Text Weight
Missing Image	7.83%	92.17%
Missing Text	2.03%	97.97%
Corrupted Image	1.09%	98.91%
Contradictory Text	1.22%	98.78%


These results indicate a strong text-modality bias in the current reliability estimator.
This also motivates future work involving explicit reliability supervision, modality dropout, and robustness-oriented training.
Project Structure
reliability-aware-multimodal-anomaly-detection/
│
├── notebooks/
│   └── factorymind_final.ipynb
│
├── models/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   └── evaluation.py
│
├── README.md
├── requirements.txt
└── .gitignore

Dataset files and trained model checkpoints are excluded from GitHub because of their size.

Experimental Environment
The models were trained using GPU resources in Google Colab. The trained checkpoints were subsequently transferred to a local VS Code Jupyter environment for evaluation, robustness experiments, visualization, and inference demonstration.
The final workflow therefore separates:
Google Colab
    │
    ├── Model training
    └── Checkpoint generation
             │
             ▼
       Trained .pth files
             │
             ▼
VS Code Jupyter
    │
    ├── Model loading
    ├── Evaluation
    ├── Robustness testing
    ├── Reliability analysis
    └── Custom inference demonstration

Installation
Clone the repository:
git clone https://github.com/<YOUR_USERNAME>/reliability-aware-multimodal-anomaly-detection.git
cd reliability-aware-multimodal-anomaly-detection

Create a virtual environment:
python -m venv .venv

Activate it on Windows:
.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Running the Project
Open the final Jupyter notebook:
notebooks/factorymind_final.ipynb

The notebook contains the complete experimental workflow:
1. Dataset preparation
2. Exploratory data analysis
3. Image-level splitting
4. Baseline models
5. Multimodal fusion
6. Reliability-aware fusion
7. Robustness experiments
8. Reliability analysis
9. Performance comparison
10. Custom image inference
Limitations
The current implementation has several limitations:
- The reliability-aware model was trained primarily on normal training data rather than explicit modality-corruption examples.
- The learned reliability weights show a strong preference for textual information.
- Reliability-aware fusion does not outperform standard fusion under every robustness condition.
- The text-only model performs strongly, indicating that question/options may contain substantial predictive information.
- Contradictory textual inputs are experimentally generated rather than collected from real industrial inspection workflows.
These limitations are considered part of the experimental findings and provide directions for future research.
Future Work
Potential improvements include:
- Reliability-aware training with explicit modality corruption
- Modality dropout during training
- Learned uncertainty estimation
- More realistic image corruption models
- Image-grounded contradictory text generation
- Calibration of reliability scores
- Evaluation on additional industrial anomaly datasets
- Larger vision-language foundation models
- Explainable modality-level decision analysis
Project
FactoryMind — Reliability-Aware Multimodal Industrial Anomaly Detection Under Missing and Contradictory Modalities
Built as an experimental research project in Multimodal AI, Industrial Anomaly Detection, and Robust Machine Learning.