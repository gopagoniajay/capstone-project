# Technical Architecture & Training Strategy

## 1. How the Training Works (The Strategy)
The system uses a **Personalized Transfer Learning** strategy to achieve high accuracy. It does not just "train once"; it adapts.

### Phase 1: Generic Training (The Foundation)
*   **Data Used**: 93% of the total dataset (All subjects *except* the target user).
*   **Goal**: Learn universal physiological patterns (e.g., "Heart rate rises during stress").
*   **Technique**: Standard supervised learning using Class Weighted Cross-Entropy Loss (to handle imbalance).

### Phase 2: Personalization (Fine-Tuning)
*   **Data Used**: Small "calibration" set (20% of the target user's data).
*   **Goal**: Adapt the model to the specific user's biology.
*   **Technique**: Even though the model is already trained, we continue training it for a few more epochs on the user's data with a **Low Learning Rate**. This adjusts the "decision boundaries" just enough to fit the user without forgetting the general rules.

---

## 2. Model Architecture (The Brain)
The model is a **Hybrid Ensemble** that combines three powerful deep learning techniques. It processes the signal in two parallel branches and then fuses the information.

### Branch A: 1D CNN (Convolutional Neural Network)
*   **Role**: *Feature Extractor* (Spatial).
*   **How it works**: It scans the biosignals looking for local patterns, like sharp peaks or specific shapes in the waveform, similar to how eyes recognize edges in an image.
*   **Components**: 2 layers of 1D Convolutions -> Batch Normalization -> ReLU.

### Branch B: LSTM (Long Short-Term Memory)
*   **Role**: *Sequence Learner* (Temporal).
*   **How it works**: It remembers what happened in the past. It understands context, like "Heart rate has been slowly rising for 10 seconds."
*   **Components**: 2-layer LSTM with Dropout.

### The Enhancer: Self-Attention Mechanism
*   **Role**: *Focus*.
*   **How it works**: Not all moments are equally important. Attention assigns a "weight" to every second of the signal. If a specific spike (e.g., a scare) is the key to detecting stress, the Attention layer highlights it and ignores the noise around it.

### The Fusion (Ensembling)
*   **Step**: The output of the CNN (64 features) and the Attention-weighted LSTM (64 features) are concatenated (glued together).
*   **Result**: A rich representation vector (128 features) that contains both local shapes and long-term context.
*   **Classification**: A final Neural Network layer predicts the class.

---

## 3. Stress Types (The Output)
The model detects **3 distinct states**. It is a multiclass classification problem.

| Class | Name | Description |
| :--- | :--- | :--- |
| **0** | **Baseline** | The neutral state. The user is relaxed. This is the most common state. |
| **1** | **Stress** | Technical/Social stress. The user is under pressure or anxiety (e.g., giving a speech, doing mental math). |
| **2** | **Amusement** | Positive excitation. The user is watching a funny video. *Crucial distinction*: Both Stress and Amusement raise heart rate, but the model learns to tell them apart. |

---

## Summary Diagram
```
Input Signal (GSR + ECG)
      |
      +-----------------------+
      |                       |
   [1D CNN]                [LSTM]
      |                       |
(Extracts Shapes)       (Extracts Time)
      |                       |
      |                 [Self-Attention]
      |                       |
      +-------[FUSION]--------+
              (Concat)
                 |
          [Classifier]
                 |
    Prediction (0, 1, or 2)
```
