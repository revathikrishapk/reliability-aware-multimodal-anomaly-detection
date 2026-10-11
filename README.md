Reliability-Aware Multimodal Anomaly Detection
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
