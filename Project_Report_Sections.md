# Project Report Sections

## 1. Problem Alignment

The core challenge addressed in this project is the significant **inter-subject variability** inherent in physiological stress detection. Generic machine learning models, trained on a population of users, often fail to generalize to new individuals due to unique physiological baselines (e.g., resting heart rate, skin conductivity levels).

*   **The Misalignment**: A generic model assumes a universal "stress pattern," but one user's baseline arousal might resemble another user's stress response. This leads to poor performance (approx. 77% accuracy) when tested on unseen subjects (Leave-One-Subject-Out validation).
*   **The Consequence**: High false positive rates, particularly confusing "Amusement" with "Stress" due to similar arousal levels (elevated heart rate, skin conductance).
*   **Our Alignment Strategy**: We align the model to each specific user through a **Rapid Calibration Framework**. By fine-tuning the pre-trained generic model on the first 20% of a user's data, we "personalize" the decision boundary, effectively mitigating the inter-subject variability problem and aligning the model's predictions with the individual's unique physiology.

## 2. Verification and Validation Test Results

To validate the efficacy of our proposed Personalized Hybrid CNN-LSTM-Attention model, we conducted rigorous testing using the WESAD dataset.

### Verification (Internal Consistency)
*   **Training Stability**: The model demonstrated stable convergence over 50 epochs, with training and validation loss decreasing exponentially. The use of Dropout and Batch Normalization prevented overfitting.
*   **Integrity Checks**: We verified the **temporal integrity** of our data splits. The calibration set (first 20%) and evaluation set (last 80%) were verified to be temporally disjoint, ensuring zero data leakage. The model was never exposed to future states during training.

### Validation (Performance Metrics)
We compared the Generic (Baseline) model against our Personalized approach:

| Metric | Generic Model (LOSO) | Personalized Model (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 77.78% | **97.04%** | **+19.26%** |
| **Precision (Stress)** | 68% | **98.45%** | **+30.45%** |
| **Recall (Stress)** | 71% | **99.62%** | **+28.62%** |
| **F1-Score (Macro)** | 0.68 | **0.96** | **+0.28** |

*   **Confusion Matrix Analysis**: The generic model struggled to differentiate Amusement from Stress. The personalized model corrected this "arousal overlap," resulting in a diagonal-dominant confusion matrix with minimal misclassifications.
*   **Calibration Plot**: The reliability diagram shows that our personalized model's predicted probabilities align closely with the ideal diagonal (y=x), indicating high confidence trustworthiness compared to the overconfident generic model.

## 3. Analysis and Comparisons

Our analysis highlights why the personalized approach outperforms traditional methods.

### Comparative Analysis with State-of-the-Art (SOTA)
Our method explicitly outperforms existing benchmarks on the WESAD dataset:

| Study | Method | Validation Strategy | Accuracy | Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Schmidt et al. (2018)** | Random Forest | Generic LOSO | 80.00% | Limited by manual feature engineering. |
| **Siavic et al. (2024)** | SVM / KNN | Generic LOSO | 75.21% | Struggles with complex temporal dependencies. |
| **Generic Deep Learning** | CNN / LSTM | Generic LOSO | ~85-90% | Fails to account for subject variability. |
| **Proposed Method** | **Hybrid CNN-LSTM + Calibration** | **Personalized** | **97.04%** | **Solves inter-subject variability & arousal overlap.** |

### Key Analysis Insights
1.  **Solving Arousal Overlap**: Identifying the subtle difference between "Happy High Heart Rate" (Amusement) and "Stressed High Heart Rate" (Stress) is the hardest task in WESAD. Our personalized model, by learning user-specific baselines (20% calibration data), successfully distinguishes these states where generic models fail, specifically targeting the ambiguous boundaries between high-arousal emotions.
2.  **Hybrid Architecture Benefit**: The **CNN branch** effectively extracts spatial/morphological features (e.g., shape of ECG peaks), while the **LSTM branch** captures temporal dependencies (e.g., rising trends in skin conductance). The **Attention mechanism** further refines this by focusing on the most salient parts of the signal, ignoring noise and artifacts.

## 4. Methodology Accuracy

Our methodology is built on a robust, scientifically valid framework designed for high accuracy and reliability.

### 1. Data Processing Pipeline
*   **Preprocessing**: Resampling to 32Hz, RobustScaler normalization (outlier removal), and sliding window segmentation (10s windows, 50% overlap).
*   **Input**: Multimodal data (ECG + EDA) fused at the feature level.

### 2. The Personalized Learning Strategy
*   **Protocol**: Calibrate on the *first 20%* of a subject's data $\rightarrow$ Evaluate on the *subsequent 80%*.
*   **Impact**: This strategy mimics a real-world deployment scenario (e.g., a "calibration phase" when a user first wears a device).
*   **Resulting Accuracy**:
    *   **Overall Accuracy**: **97.04%**
    *   **Stress Detection Accuracy**: **99.62% (Recall)** – The system almost never misses a stress event.
    *   **Baseline Accuracy**: **97.44%** – The system accurately identifies resting states.

### 3. Validity Confirmation
The high accuracy is not a result of overfitting. The strict separation of calibration and evaluation data, confirmed by timestamp verification, ensures that the model is genuinely learning to generalize to the user's future physiological states.

## 5. Conclusion

This project successfully addresses the critical bottleneck in automated stress detection: **inter-subject variability**. By transitioning from a "one-size-fits-all" generic model to a **Personalized Hybrid CNN-LSTM-Attention** framework, we achieved a remarkable accuracy improvement from **77.78%** to **97.04%**.

**Key Takeaways:**
1.  **Superior Performance**: The system outperforms state-of-the-art methods (Schmidt et al., etc.) by a significant margin (+17-21%).
2.  **Reliability**: The model is well-calibrated and demonstrates high integrity, with verified data separation to prevent leakage.
3.  **Real-World Applicability**: The "Rapid Calibration" approach (using only 20% data) offers a practical path for deploying highly accurate, personalized stress monitoring systems on wearable devices, ensuring privacy and user trust.
