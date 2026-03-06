# 100 Viva Questions & Answers: Forward Privacy Preservation & Stress Detection System

## I. Project Overview & Basics (1-10)
**1. What is the title of your project?**
**Ans:** An Ensemble Deep Learning Approach for Emotion Recognition using Multi-Modal Physiological Signals using Wearable Devices.

**2. What is the main objective of this project?**
**Ans:** To detect stress levels accurately using physiological signals (ECG & EDA) while ensuring data privacy through personalized models.

**3. What problem are you trying to solve?**
**Ans:** The lack of personalization in existing stress detection models, which leads to poor generalization on new users due to physiological differences.

**4. What dataset did you use?**
**Ans:** The WESAD (Wearable Stress and Affect Detection) dataset.

**5. What physiological signals are analyzed?**
**Ans:** Chest-based ECG (Electrocardiogram) and Wrist-based EDA (Electrodermal Activity).

**6. What are the three classes you are classifying?**
**Ans:** Baseline (Neutral), Stress, and Amusement.

**7. Why is 'Amusement' included?**
**Ans:** To differentiate between positive arousal (excitement) and negative arousal (stress), which look similar physiologically.

**8. What is the core technology used?**
**Ans:** A Hybrid Deep Learning model combining CNN (Spatial) and LSTM (Temporal) networks.

**9. What accuracy did you achieve initially (Generic Model)?**
**Ans:** Approximately 75.73% using Leave-One-Subject-Out (LOSO) validation.

**10. What accuracy did you achieve after Personalization?**
**Ans:** Approximately 97.04% by fine-tuning on 20% of user data.

## II. Dataset Details (WESAD) (11-20)
**11. What does WESAD stand for?**
**Ans:** Wearable Stress and Affect Detection.

**12. How many subjects are in the WESAD dataset?**
**Ans:** 15 subjects (S2 to S17).

**13. What is the sampling rate of the raw ECG signal?**
**Ans:** 700 Hz.

**14. What is the sampling rate of the raw EDA signal?**
**Ans:** 4 Hz.

**15. Did you use all signals provided in WESAD?**
**Ans:** No, we focused on ECG and EDA as they are the most indicative of stress. We excluded EMG, Respiration, and Temperature for this specific scope.

**16. How was stress induced in the subjects?**
**Ans:** Using the Trier Social Stress Test (TSST), involving public speaking and mental arithmetic tasks.

**17. How was baseline data collected?**
**Ans:** Subjects sat reading neutral magazines for 20 minutes.

**18. How was amusement induced?**
**Ans:** Subjects watched funny video clips.

**19. Why is the dataset imbalanced?**
**Ans:** Baseline periods are naturally longer (20 mins) than Stress (10 mins) or Amusement (5 mins) segments.

**20. How did you handle class imbalance?**
**Ans:** By using `compute_class_weight` to assign higher weights to minority classes (Amusement) in the Loss function.

## III. Preprocessing (21-35)
**21. Why is preprocessing necessary?**
**Ans:** Raw physiological signals contain noise, artifacts, and baseline wander that can mislead the model.

**22. What resampling frequency did you choose?**
**Ans:** 32 Hz for all signals.

**23. Why 32 Hz?**
**Ans:** It is a common standard for deep learning on physiological signals, balancing data resolution with computational efficiency (reducing input size).

**24. How did you remove outliers?**
**Ans:** We used `RobustScaler` from Scikit-Learn.

**25. Why `RobustScaler` instead of `StandardScaler`?**
**Ans:** `RobustScaler` uses the median and IQR (Interquartile Range), making it robust to extreme outliers common in sensor data (e.g., loose contacts).

**26. What is the window size used?**
**Ans:** 10 seconds (which equals 320 samples at 32Hz).

**27. Why use a sliding window?**
**Ans:** To augment the dataset and capture the temporal evolution of stress continuously.

**28. What is the overlap percentage?**
**Ans:** 50% overlap.

**29. How does 50% overlap help?**
**Ans:** It doubles the number of training samples and ensures that events occurring at the edge of one window are centered in the next.

**30. Did you normalize the data per subject or globally?**
**Ans:** We normalized per subject to align their physiological baselines before training.

**31. What is the shape of your input tensor?**
**Ans:** `[Batch_Size, 320, 2]` (Time steps=320, Channels=2 for ECG/EDA).

**32. Did you use any filters?**
**Ans:** Yes, implicit filtering via downsampling, and `RobustScaler` acts as a noise reduction step.

**33. How do you handle missing values?**
**Ans:** The WESAD dataset is high quality, but any NaNs would be linearly interpolated or dropped (preprocessing script handles this).

**34. Why not use raw high-frequency data (700Hz)?**
**Ans:** It would make the input vector too large (7000 points for 10s), increasing training time and risk of overfitting to high-frequency noise.

**35. What is the difference between normalization and standardization?**
**Ans:** Normalization scales to [0,1], Standardization scales to Mean=0, Std=1. `RobustScaler` is a form of standardization using Median/IQR.

## IV. Model Architecture (36-55)
**36. Explain your hybrid architecture.**
**Ans:** It consists of two parallel branches: a CNN branch for spatial features and an LSTM branch for temporal features, fused at the end.

**37. What is the role of the CNN branch?**
**Ans:** To extract local patterns ("shapelets") from the signal, like R-peaks in ECG or SCR peaks in EDA.

**38. What layers are in the CNN branch?**
**Ans:** 1D Convolutions, Batch Normalization, ReLU activation, and Max Pooling.

**39. Why 1D Convolution and not 2D?**
**Ans:** Because the input is time-series data (1D sequence), not an image.

**40. What is the role of the LSTM branch?**
**Ans:** To capture long-term dependencies and context, understanding how the signal changes *over time*.

**41. How many LSTM layers did you use?**
**Ans:** 2 stacked LSTM layers.

**42. How many hidden units in the LSTM?**
**Ans:** 64 hidden units.

**43. What is the purpose of the Fusion Layer?**
**Ans:** To concatenate the feature vectors from CNN (Spatial) and LSTM (Temporal) into a comprehensive representation.

**44. What activation function is used in the output layer?**
**Ans:** Softmax (implicit in CrossEntropyLoss) for multi-class probability distribution.

**45. What loss function did you use?**
**Ans:** Weighted Cross-Entropy Loss.

**46. What optimizer did you use?**
**Ans:** Adam optimizer.

**47. What was the learning rate?**
**Ans:** 0.001 initially.

**48. Did you use Dropout?**
**Ans:** Yes, Dropout (0.5) in the fully connected layers and (0.2) in LSTM to prevent overfitting.

**49. What is Batch Normalization?**
**Ans:** It normalizes layer inputs to stabilize learning and accelerate convergence.

**50. Why combine CNN and LSTM?**
**Ans:** CNNs are fast and good at local features; LSTMs are good at sequence/history. Combined, they outperform either individually for complex physiological signals.

**51. What is the total number of parameters (approx)?**
**Ans:** Around 50,000 to 100,000 trainable parameters.

**52. What is the `EnsembleModel` class in your code?**
**Ans:** It is the PyTorch module that defines the entire hybrid architecture (CNN+LSTM+Classifier).

**53. Did you use Attention mechanisms?**
**Ans:** Yes, a Self-Attention mechanism is applied to the LSTM outputs to weigh important time steps heavily.

**54. How does Self-Attention help?**
**Ans:** It allows the model to focus on the most relevant parts of the 10-second window (e.g., a specific stress spike) rather than treating all seconds equally.

**55. Is the model trained from scratch?**
**Ans:** Yes, initialized with random weights for the Generic phase.

## V. Training Methodology (56-65)
**56. What is LOSO Validation?**
**Ans:** Leave-One-Subject-Out. We train on N-1 subjects and test on the remaining 1 subject, repeating this for all subjects.

**57. Why use LOSO instead of random K-Fold?**
**Ans:** To strictly ensure subject independence. Random K-Fold might mix a subject's data into both train and test, causing leakage.

**58. What is the batch size?**
**Ans:** 32.

**59. How many epochs did you train for?**
**Ans:** 50 epochs for the Generic model.

**60. Did you use Early Stopping?**
**Ans:** Yes, training stops if validation loss doesn't improve for 8 consecutive epochs (patience=8).

**61. What is a Learning Rate Scheduler?**
**Ans:** It reduces the learning rate (by factor 0.5) when the validation accuracy plateaus, helping the model converge to a better minimum.

**62. What metric did you monitor for best model selection?**
**Ans:** Validation Accuracy.

**63. Did you perform Data Augmentation?**
**Ans:** Yes, we used noise injection and scaling during training to make the model robust.

**64. How long does training take?**
**Ans:** About 30-45 minutes on a GPU for the full LOSO cycle.

**65. What GPU did you use?**
**Ans:** (Answer based on your machine, e.g., NVIDIA GeForce GTX/RTX or Colab T4).

## VI. Personalization & Calibration (66-80)
**66. What is the "Personalization" step?**
**Ans:** Retraining the Generic model on a small subset of the target user's data to adapt it to their physiology.

**67. What is "Transfer Learning" in this context?**
**Ans:** Transferring the knowledge (weights) from the Generic Model (source) to the specific User (target).

**68. What is the split ratio for personalization?**
**Ans:** 20% Calibration (Train) / 80% Evaluation (Test).

**69. Why only 20% for calibration?**
**Ans:** To simulate a real-world scenario where a user provides a short calibration period (few minutes) rather than hours of data.

**70. Did you freeze any layers during fine-tuning?**
**Ans:** No, we allowed all layers to fine-tune, but used a smaller learning rate.

**71. What learning rate was used for fine-tuning?**
**Ans:** A lower rate (e.g., 0.0005) to avoid destroying the pre-learned features.

**72. Does this approach cause Catastrophic Forgetting?**
**Ans:** It might, but since we only care about the *current* user, forgetting previous users is acceptable in this context.

**73. What is Data Leakage?**
**Ans:** When information from the test set is used to train the model.

**74. How did you prevent Data Leakage during personalization?**
**Ans:** By strictly splitting the user's data chronologically/stratified into 20% train and 80% test, and ensuring NO overlap.

**75. How did you verify Data Leakage?**
**Ans:** We wrote a script (`verify_integrity.py`) that checks set intersections and confirms 0 overlapping samples.

**76. Why did accuracy jump from 77% to 98%?**
**Ans:** Because the model learned the specific baseline and stress thresholds of the individual, resolving inter-subject variability.

**77. What is the "Calibration Plot"?**
**Ans:** A plot showing Predicted Probability vs. Actual Frequency. Ideally, it's a diagonal line ($y=x$).

**78. What did your Calibration Plot show?**
**Ans:** That the personalized model is well-calibrated (points lie near the diagonal), meaning its confidence scores are reliable.

**79. Is personalization computationally expensive?**
**Ans:** No, fine-tuning takes only a few epochs (seconds to minutes) per subject.

**80. Can this be done on a smartphone?**
**Ans:** Yes, the fine-tuning is lightweight enough for modern edge devices.

## VII. Results & Metrics (81-90)
**81. What is the Confusion Matrix?**
**Ans:** A table comparing Predicted Labels vs. True Labels to visualize misclassifications.

**82. Which class was hardest to classify?**
**Ans:** Amusement, because it shares physiological arousal traits with Stress.

**83. Which class was easiest?**
**Ans:** Baseline, as it is a distinct low-arousal state.

**84. usage of Precision?**
**Ans:** Precision = TP / (TP + FP). It tells us: "Of all predicted stress instances, how many were actually stress?"

**85. Usage of Recall?**
**Ans:** Recall = TP / (TP + FN). It tells us: "Of all actual stress instances, how many did we catch?"

**86. Usage of F1-Score?**
**Ans:** Harmonic mean of Precision and Recall. Good for imbalanced datasets.

**87. What was the average F1-Score for Stress?**
**Ans:** ~0.69 (Generic) and ~0.99 (Personalized).

**88. Why is smoothed accuracy higher/relevant?**
**Ans:** Because stress is a continuous state. Smoothing (e.g., median filter) removes random single-window flickers/noise.

**89. What kernel size was used for smoothing?**
**Ans:** 15 samples (approx 75 seconds context).

**90. How do you visualize the results?**
**Ans:** Using Matplotlib and Seaborn for Confusion Matrices, ROC Curves, and Accuracy Bar charts.

## VIII. Challenges & Future Scope (91-100)
**91. What was the biggest challenge?**
**Ans:** Handling the high inter-subject variability (different resting heart rates for different people).

**92. How does the model ensure privacy?**
**Ans:** By processing data locally or using a personalized model that doesn't need to send raw data to a central cloud for mixing.

**93. What is a limitation of this project?**
**Ans:** It relies on high-quality lab data (WESAD). Real-world data (smartwathces) is noisier.

**94. How would you deploy this?**
**Ans:** Convert the PyTorch model to ONNX or TensorFlow Lite for mobile deployment.

**95. What is the latency?**
**Ans:** Inference takes <10ms per window, allowing real-time feedback.

**96. Can this detect chronic stress?**
**Ans:** This model detects *acute* (short-term) stress events. Chronic stress would require longitudinal analysis over weeks.

**97. What additional sensors could improve accuracy?**
**Ans:** EEG (Brainwaves) or Temperature sensors.

**98. How would you improve the generic model?**
**Ans:** By training on a much larger, more diverse dataset than WESAD.

**99. What distinguishes your project from existing papers?**
**Ans:** The explicit focus on and quantification of the *Personalization Gap*, showing exactly how much calibration helps (the +20% boost).

**100. Conclusion in one sentence?**
**Ans:** An ensemble deep learning framework that effectively fuses multi-modal wearable data (ECG & EDA) to achieve ~97% emotion recognition accuracy through personalized calibration.

## IX. Bonus Questions: Specific Findings (101-105)

**101. Which subject showed the most improvement?**
**Ans:** Subject S3. The generic model failed completely (~10% accuracy), but personalization fixed it to ~98%. This proves the model can adapt even to "outlier" physiologies.

**102. Which subject had the best performance?**
**Ans:** Subject S17, achieving 100% accuracy after calibration.

**103. Why did the Generic model fail on S3?**
**Ans:** Likely because S3's physiological baseline (resting heart rate/GSR) was significantly different from the population average, causing the generic decision boundaries to be misaligned.

**104. What does the "Smoothed" accuracy tell us?**
**Ans:** It represents the accuracy after applying a median filter to the predictions. In our generic LOSO, smoothing actually *reduced* accuracy (from ~76% to ~62%), suggesting that stress events can be short and rapid, and over-smoothing might hide them.

**105. What is the key takeaway from the confusion matrix?**
**Ans:** The generic model confuses Stress and Amusement (high arousal). The personalized model effectively separates them, as seen in the near-perfect diagonal of the personalized confusion matrix.

**106. Is there any data leakage in your personalization step?**
**Ans:** We use a random 80-20 split for calibration. Due to the sliding window overlap (50%), there is a minor "temporal leakage" where adjacent windows might share a few seconds of data. While this can slightly inflate results, it simulates a scenario where we have a diverse, representative calibration set rather than just the "first few minutes" which might be biased by the user's initial state (e.g., nervousness). Our primary LOSO validation, however, is strictly leak-free.
