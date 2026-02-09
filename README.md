# HAPM: Hierarchical Adaptive Perception Based Multimodal Named Entity Recognition for Agricultural Diseases and Pests Question Answering
<img width="3729" height="2300" alt="HAPM" src="https://github.com/user-attachments/assets/27d09902-628c-4023-a124-d1001684abae" />
The architecture of the HAPM model. The model comprises three stages: (1) Feature Extraction Module: Utilizing frozen weights of Swin-T and BERT to extract image and text features respectively; (2) Hierarchical Cross-Modal Interaction Module: Implementing deep modal interaction through a 3-layer stacked cross-attention mechanism; (3) Adaptive Quality-Aware Aggregation Module: (a) Layer Weights, used to learn the importance of different fusion layers; (b) Quality Estimation Network, used to generate quality-aware weights β based on modal contribution; (c) Dynamic Gating, used to balance the complementary contributions of the dual modalities.

# 🌟 Overview
**Accurate named entity recognition (NER)** is crucial for knowledge graph question answering (KGQA), as it identifies the named entity, i.e., the head entity from the given utterance to initiate knowledge graph reasoning. However, in the field of agricultural NER, existing methods are primarily designed for textual questions and falter in multimodal scenarios.
<img width="3989" height="1983" alt="Example of Agricultural KGQA Process Integrating Multimodal Named Entity Recognition" src="https://github.com/user-attachments/assets/34e88ceb-bc5d-40ca-9905-4ae88e511dcd" />

HAPM addresses three key challenges:

**Information segregation between modalities**: Inability to extract named entities hidden only in images.
**Rigid Entity Recognition**: Inability to dynamically recognize different entities in the same image under different questions.
**Fixed Modal Weights**: Inability to adaptively balance contributions from visual and textual inputs.

# ✨ Key Features

**Hierarchical Cross-modal Interaction (HCI)**: Multi-scale feature interaction mechanism that progressively extracts semantic features at different granularities through multi-layer cross-modal interaction.

**Hierarchical Adaptive Weighting (HAW)**: Dynamically adjusts the importance of entities at different hierarchical levels based on different questions for the same image, achieving question-driven entity recognition.

**Modal Quality-Aware (MQA)**: Automatically evaluate the contributions of visual and textual modalities, guiding the model to focus on more informative and higher-quality modalities.

**AgriMNE Dataset**: A multimodal agricultural pest and disease named entity recognition dataset containing 92 named entity categories and 10,425 image-text question pairs.

# 📊 AgriMNE Dataset
We have provided the AgriMNE dataset, which is a dataset used for agricultural multimodal named entity recognition.
<img width="3740" height="993" alt="Example of image annotation" src="https://github.com/user-attachments/assets/4995493e-e48e-4fc0-a00b-b5f136ac1d46" />

# 📈 Results
Our HAPM model achieves state-of-the-art performance on the AgriMNE dataset:

| Model | Text | Image | F1 | Recall | Precision | ACC@3 | Accuracy |
|-------|:----:|:-----:|:--:|:------:|:---------:|:-----:|:--------:|
| BERT | ✓ | ✗ | 37.45 | 30.28 | 67.95 | 49.45 | 38.34 |
| RoBERTa | ✓ | ✗ | 33.87 | 28.20 | 61.84 | 50.15 | 37.37 |
| Sentence Transformers | ✓ | ✗ | 10.14 | 9.94 | 18.65 | 41.37 | 22.73 |
| ViT | ✗ | ✓ | 30.20 | 31.37 | 31.31 | 97.56 | 34.83 |
| ResNet50 | ✗ | ✓ | 35.91 | 37.73 | 37.20 | 95.90 | 39.80 |
| EfficientNet-B7 | ✗ | ✓ | 42.02 | 46.24 | 42.60 | 96.49 | 43.41 |
| ConvNext | ✗ | ✓ | 33.74 | 34.94 | 36.35 | 94.54 | 42.24 |
| CLIP | ✓ | ✓ | 81.07 | 82.75 | 82.92 | 97.76 | 87.12 |
| ALBEF | ✓ | ✓ | 86.23 | 88.14 | 87.65 | 98.24 | 92.29 |
| BLIP | ✓ | ✓ | 85.56 | 86.26 | 89.03 | 94.44 | 91.02 |
| ViLT | ✓ | ✓ | 84.26 | 86.59 | 85.58 | 97.37 | 90.73 |
| VisualBERT | ✓ | ✓ | 83.09 | 84.00 | 84.43 | 96.98 | 91.02 |
| CoCa | ✓ | ✓ | 85.17 | 86.97 | 86.52 | 98.34 | 91.80 |
| Qwen-VL-Max | ✓ | ✓ | 46.53 | 50.90 | 51.97 | 76.76 | 56.54 |
| Llama 4 | ✓ | ✓ | 39.79 | 41.30 | 53.75 | 69.27 | 51.51 |
| **HAPM (Ours)** | ✓ | ✓ | **90.92** | **92.02** | **91.75** | **98.83** | **94.54** |

> **Note**: ✓ indicates the modality is used, ✗ indicates it is not used. Bold numbers represent the best performance.

# 🖼️ Visualization
Case analysis and weight visualization of the modal quality-aware mechanism
<img width="3559" height="1272" alt="Case analysis and weight visualization of the modal quality-aware mechanism" src="https://github.com/user-attachments/assets/77579327-8649-4fc9-ab79-9df1853c08b0" />

Performance comparison of different models in the problem of information separation between modalities
<img width="3022" height="1526" alt="Performance comparison of different models in the problem of information separation between modalities" src="https://github.com/user-attachments/assets/fc1000ca-2e3c-4eb2-a26c-4edee59b39a1" />

# ⚠️ Note
**As the project is still in progress, this repository currently contains a portion of the dataset. The complete dataset will be released after the project concludes.**
