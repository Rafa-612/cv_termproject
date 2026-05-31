# DiNomaly Model Performance Comparison Report: MVTec-AD vs. MVTec LOCO AD

This report presents a comprehensive academic analysis of the performance differences of the **DiNomaly** model when trained and evaluated on two major industrial anomaly detection datasets:
1.  **MVTec-AD**: The standard benchmark dataset (tested in the original DiNomaly paper).
2.  **MVTec LOCO AD**: The logical constraints anomaly detection dataset (tested in our project).

---

## 1. Metric Performance Comparison Table

The table below compares the performance of DiNomaly (ViT-Base/14 backbone) on both datasets:

| Dataset | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel F1-max | Pixel AUPRO | Combined / Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MVTec-AD** *(Paper)* | **99.60%** | **99.80%** | **99.00%** | **98.40%** | **69.30%** | **69.20%** | **94.80%** | **99.60%** |
| **MVTec LOCO AD** *(Ours)* | **81.76%** | **89.14%** | **82.15%** | **72.88%** | **32.01%** | **32.33%** | **63.46%** | **83.30%** |
| **Difference** | **-17.84%** | **-10.66%** | **-16.85%** | **-25.52%** | **-37.29%** | **-36.87%** | **-31.34%** | **-16.30%** |

---

## 2. In-Depth Analysis of Performance Discrepancy

The performance drop on MVTec LOCO AD is highly significant, particularly in **pixel-level localization metrics (P-AP and P-F1 drop by over 36%)** and **AUPRO (drops by 31%)**. 

We analyze this discrepancy across three primary dimensions: **Image Morphology (Data Distribution)**, **Model Architecture (Inductive Biases)**, and **Training & Evaluation Methods**.

---

### Dimension A: Image Morphology & Nature of Anomalies

#### 1. Structural Defects vs. Logical Constraints
*   **MVTec-AD**: Focuses exclusively on **structural anomalies** (scratches, dents, cracks, stains, or missing parts). These anomalies manifest as local, high-contrast, high-frequency textural or geometric deformations.
*   **MVTec LOCO AD**: Specifically designed to evaluate **logical anomalies**. Logical anomalies violate global spatial constraints or co-occurrence rules (e.g., an incorrect number of components, correct components placed in wrong boxes, or incorrect liquid levels).
*   **Impact on Metrics**: As seen in our LOCO results, DiNomaly performs exceptionally well on LOCO's **Structural subset (Image AUROC: 92.89%, Pixel AUROC: 96.24%)**, which is comparable to its performance on MVTec-AD. However, it struggles on LOCO's **Logical subset (Image AUROC: 73.72%, Pixel AUPRO: 50.30%)**.

```
Logical Anomaly (e.g., wrong object count)  --> Identical local textures --> Decoder reconstructs it easily --> Low reconstruction error
Structural Anomaly (e.g., scratch on surface) --> Novel local texture     --> Decoder fails to reconstruct --> High reconstruction error
```

#### 2. The Identity Reconstruction Trap of Logical Anomalies
*   For structural anomalies, the training set never contains scratches, so the decoder cannot reconstruct them.
*   For logical anomalies, the individual objects themselves (e.g., a pushpin or a screw) are **completely normal in appearance** and have been seen thousands of times during training. The anomaly lies solely in their *number* or *spatial arrangement*.
*   Since the encoder (DINOv2) extracts excellent representation tokens for these "normal-looking" objects, the decoder reconstructs them **perfectly**, even if they are in the wrong place or quantity. This leads to **very low reconstruction error on anomalous regions**, causing a severe drop in localization metrics (P-AP and P-AUPRO).

---

### Dimension B: Model Architecture & Representation Bottlenecks

#### 1. DINOv2 Semantic Feature Bias
*   **DINOv2** is trained via self-supervised learning (DINO + iBOT) on massive natural image datasets. Its feature representations are highly **semantic** and **locally robust**.
*   This local robustness means DINOv2 is invariant to minor spatial shifts, which makes it excellent at ignoring noise in MVTec-AD.
*   However, detecting logical anomalies requires strict **relative spatial geometry** and **exact counting**. DINOv2’s semantic tokens tend to represent *what* the object is (e.g., "screw") rather than *exactly where* it is relative to a coordinate frame or *how many* there are.

#### 2. Linear Attention Limitation
*   DiNomaly replaces standard softmax attention in the decoder with **Linear Attention** (`LinearAttention2`) to maintain linear complexity with respect to token length.
*   Softmax attention computes non-linear pairwise similarity matrices, which are highly sensitive to exact pixel-level positional coordinate shifts.
*   Linear attention computes key-value aggregations first, smoothing out spatial details. This spatial smoothing makes it extremely difficult for the decoder to capture strict structural rules (such as: *"a splicing connector must have exactly 5 metal pins"*). When a connector has only 4 pins, the smoothed attention maps fail to recognize the missing slot, leading to a low reconstruction error.

---

### Dimension C: Training Dynamics & Evaluation Logic

#### 1. Intra-class Variance vs. Inter-class Variance
*   In **MVTec-AD**, objects are highly aligned and rigid (e.g., a transistor is always in the center under the same camera angle). The intra-class variance of normal images is near zero.
*   In **MVTec LOCO AD**, normal images have high intra-class variance. For example, in the `screw_bag` category, the screws and washers are allowed to slide around randomly inside the plastic bag. 
*   Because the normal bagging layout is random, the DiNomaly decoder is forced to learn a **highly relaxed reconstruction constraint** to avoid false positives. This loose constraint behaves like a low-pass filter, which inadvertently allows logical anomalies (such as a missing washer) to be reconstructed within the "normal variation" boundary, resulting in a high rate of False Negatives.

#### 2. Metric Sensitivity (AUPRO and Pixel-level AP)
*   **MVTec-AD** contains relatively large defect regions (like a large scratch or patch of rust), making it easier to hit a high Pixel-level Average Precision (69.30%).
*   **MVTec LOCO AD** logical defects are often point-like or subtle boundary mismatches (e.g., a screw head offset by 2mm). The ground-truth mask is tiny. 
*   For small masks, any slight leakage or blur in the anomaly map (caused by bilinear interpolation and Gaussian filtering in `cal_anomaly_maps`) introduces many false positive pixels. This drastically penalizes **Pixel-level AP (32.01%)** and **AUPRO (63.46% overall, dropping to 50.30% on Logical)**.

---

## 3. Conclusions and Project Insights

1.  **Duality of UAD**: The experiment proves that unsupervised anomaly detection is not a single unified task. Models that excel at structural anomaly detection (99.6% on MVTec-AD) are fundamentally bottlenecked when faced with logical and relationship constraints (83.3% on LOCO).
2.  **Architectural Trade-offs**: While Linear Attention and Noisy Bottlenecks stabilize multi-class training and prevent shortcut learning for textures, they limit the model's capacity to represent strict geometric and counting structures.
3.  **Future Directions**: To bridge the gap on datasets like MVTec LOCO AD, future models should integrate **explicit object-level binding** (e.g., combining ViT reconstructions with object query detection like DETR) or utilize **visual-language models (VLMs)** to query logical contradictions directly.
