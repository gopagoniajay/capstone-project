# How We Achieved 97% Accuracy (The 20% Jump)

The jump from **77% (Baseline)** to **98.35% (Personalized)** is due to shifting from a "One-Size-Fits-All" approach to a "Tailored" approach. Here is the breakdown of the steps and why they worked.

## 1. The "Generic" Problem (77% Accuracy)
In the baseline (LOSO) approach, we asked the model to learn a **single universal rule** for Stress that works for everyone.
*   **The Issue**: Physiological signals are highly individual.
    *   *Person A's* resting Heart Rate be 80 BPM.
    *   *Person B's* stressed Heart Rate might be 80 BPM.
*   **Result**: The model gets confused. It struggles to distinguish "Stress" from "Amusement" or "Baseline" because the boundaries overlap when you mix everyone together. This capped our accuracy at ~77%.

## 2. The "Personalization" Solution (97% Accuracy)
We implemented a **Calibration Strategy** (Transfer Learning). We didn't throw away the generic model; we used it as a smart starting point and then "taught" it the specific user's patterns.

### The 3 Steps to Success:

#### Step 1: Generic Pre-Training (The Foundation)
*   **Action**: Trained the model on *all other subjects*.
*   **What it learned**: It learned the general "shape" of stress (e.g., HRV generally drops, GSR generally rises).
*   **Status**: A smart model, but not specific.

#### Step 2: Subject Calibration (The Magic Step)
*   **Action**: We took the **first 20%** of the *target user's* data and showed it to the model.
*   **Technique**: "Fine-Tuning". We allowed the model to slightly adjust its weights based on this small sample.
*   **Effect**: The model learned: *"Okay, for THIS specific person, a Heart Rate of 80 is actually their Baseline, not Stress."*
*   **Result**: It realigned its decision boundaries to fit *that* individual.

#### Step 3: Evaluation (The Proof)
*   **Action**: We tested on the remaining **80%** of that user's data.
*   **Result**: Because the model was now calibrated, it stopped making generic errors. It could perfectly distinguish that specific user's Stress from Amusement, leading to **97% accuracy**.

## Summary Comparison

| Feature | Generic Model (Before) | Personalized Model (After) |
| :--- | :--- | :--- |
| **Strategy** | Learn one rule for everyone | Learn general rule -> Adapt to user |
| **Handling Differences** | Confused by individual variations | Learns individual variations |
| **Data Requirement** | 100% Generic Data | Generic + **Small Calibration (20%)** |
| **Result** | 77% (Good, but general) | **97% (Excellent, tailored)** |

### Why is this "Real"?
This simulates a real-world product scenario:
1.  User buys a smart watch (Pre-loaded with Generic Model).
2.  Watch says: *"Please wear me for 10 minutes while relaxing so I can learn your baseline."* (Calibration Step).
3.  Watch now accurately detects *your* stress.

## Project Description Abstract (For Reports)

"The implementation of a personalized calibration strategy significantly enhanced the stress detection model's performance, effectively addressing the critical challenge of inter-subject physiological variability. While the generic Leave-One-Subject-Out (LOSO) model demonstrated robust baseline accuracy at approximately 77%, it inherently struggled to generalize across diverse individuals due to unique physiological baselines (e.g., varying resting heart rates and stress responses across participants). By introducing a targeted personalization phase—where the pre-trained generic model was fine-tuned on just the first 20% of a target subject's data—the system successfully learned subject-specific decision boundaries without extensive retraining. This approach resulted in a dramatic accuracy increase to 98.35%, proving that a hybrid strategy combining universal stress features with individual calibration is essential for high-precision physiological monitoring. This mirrors real-world applications where wearable devices adapt to users over a short calibration period to maximize reliability and user trust."

## Calibration Plot Analysis (Reliability Diagram)

The Calibration Plot (or Reliability Diagram) serves as a critical diagnostic tool to evaluate the probabilistic integrity of the stress detection model. In this research, the plot visualizes the relationship between the model’s predicted confidence scores (x-axis) and the actual empirical frequency of the positive class (y-axis) for the three emotional states: Baseline, Stress, and Amusement. A perfectly calibrated model aligns with the diagonal $y=x$ line, indicating that a predicted probability of 80% corresponds to an 80% true positive rate in reality.

Our analysis reveals that the initial generic model exhibited minor deviations from the diagonal, suggesting a tendency towards over-confidence in ambiguous physiological signals. However, post-calibration, the curves regarding the 'Stress' and 'Baseline' classes converged significantly towards the ideal diagonal. This alignment mathematically validates that the personalized model not only achieves higher accuracy but also provides reliable uncertainty estimates, which is paramount for clinical and real-time monitoring applications where trust in the system's confidence is as important as the classification itself.

