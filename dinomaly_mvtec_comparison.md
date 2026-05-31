# Deep Comparative Analysis of DiNomaly: MVTec-AD vs. MVTec LOCO AD
*An In-Depth Study Based on Bergmann et al. (CVPR 2019 & CVPR 2022)*

This report provides a deep comparative analysis of the **DiNomaly** model's performance on the standard **MVTec-AD** dataset versus the **MVTec LOCO AD** dataset. It references the core theoretical foundations of the two dataset papers:
1.  **MVTec-AD Paper**: *"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection"* (CVPR 2019)
2.  **MVTec LOCO AD Paper**: *"Beyond Denting: Introducing the MVTec Logical Constraints Anomaly Detection Dataset"* (CVPR 2022)

---

## 1. Metric Performance Comparison Table

Below is the comparative metric breakdown of the DiNomaly model (ViT-Base/14 backbone) trained under the multi-class setting:

| Dataset | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel F1-max | Pixel AUPRO | Combined / Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MVTec-AD** *(Paper)* | **99.60%** | **99.80%** | **99.00%** | **98.40%** | **69.30%** | **69.20%** | **94.80%** | **99.60%** |
| **MVTec LOCO AD** *(Ours)* | **81.76%** | **89.14%** | **82.15%** | **72.88%** | **32.01%** | **32.33%** | **63.46%** | **83.30%** |
| **Difference** | **-17.84%** | **-10.66%** | **-16.85%** | **-25.52%** | **-37.29%** | **-36.87%** | **-31.34%** | **-16.30%** |

---

## 2. Mathematical Modeling of Anomalies in UAD

To understand why the same model performs so differently on these two datasets, we must model anomaly detection mathematically. 

Let an image X be represented as a set of local patches X = {p_1, p_2, ..., p_N}, where each patch p_i represents local texture and structure.

```
                  ┌──────────────────────────────────────────┐
                  │          Total Image Space (X)           │
                  │                                          │
                  │   ┌──────────────────────────────────┐   │
                  │   │     Global Joint Manifold        │   │
                  │   │      P_global(p_1,...,p_N)       │   │
                  │   │                                  │   │
                  │   │   ┌──────────────────────────┐   │   │
                  │   │   │  Local Patch Manifold    │   │   │
                  │   │   │       P_local(p_i)       │   │   │
                  │   │   └──────────────────────────┘   │   │
                  │   └──────────────────────────────────┘   │
                  └──────────────────────────────────────────┘
```

### 1. The Local Patch Manifold: P_local(p_i)
Unsupervised Anomaly Detection (UAD) models learn the probability distribution of normal local patches. If a patch p_i has a low probability under this distribution:
*   **P_local(p_i) < ε** (where ε is a very small threshold)
it is classified as a **structural anomaly** (e.g., a scratch, crack, or hole).

### 2. The Global Joint Manifold: P_global(p_1, p_2, ..., p_N)
Logical anomalies do not violate local patch distributions; they violate the joint probability distribution of all patches in the image. An image contains a logical anomaly if:
*   **P_global(p_1, p_2, ..., p_N) < ε**   *while*   **for all i, P_local(p_i) >= θ** (where θ is a normal threshold)
This means every single patch p_i is completely normal locally, but their co-occurrence or spatial arrangement is invalid.

---

## 3. Essential Differences Between the Datasets

### A. MVTec-AD (CVPR 2019): The Local Manifold Paradigm
In the CVPR 2019 paper, Bergmann et al. defined anomalies as **structural defects** (e.g., scratches on leather, broken wires in grid, or contamination on pills). 
*   **Textural and Structural OOD**: These anomalies introduce novel high-frequency textures, color changes, or structural breaks.
*   **Locally Out-of-Distribution (OOD)**: A patch containing a scratch is OOD relative to the normal local patch manifold $P_{\text{local}}$.
*   **Reconstruction Failure**: Because the model is trained exclusively on normal local manifolds, it lacks the capacity to reconstruct OOD textures. The reconstruction error at the scratch patch is naturally high, allowing easy detection.

### B. MVTec LOCO AD (CVPR 2022): The "Beyond Denting" Paradigm
In the CVPR 2022 paper, Bergmann et al. argued that industrial inspection must move **"beyond denting"** (beyond simple local structural defects). They introduced **Logical Constraints**:
*   **Visually Intact Components**: The individual components (e.g., a screw, a washpin, a capsule) are **visually intact and completely normal**.
*   **Locally In-Distribution (ID)**: Every local patch in a logical anomaly exists in the normal training set. For instance, in `breakfast_box`, the patch containing a nectarine is ID ($P_{\text{local}}(p_i) \geq \theta$) whether there is one nectarine or two in the box.
*   **Global Relational Violations**: The anomaly is OOD only under the global joint distribution $P_{\text{global}}$ (e.g., the co-occurrence of two nectarines with no energy bar).

---

## 4. Why the Same Model Performs Differently

When we train DiNomaly on both datasets, the architectural and training properties interact with these dataset differences in the following ways:

### 1. Decoder Over-Generalization (Reconstruction Capacity)
*   **In MVTec-AD**: The encoder (DINOv2) extracts OOD features for scratches. The decoder cannot map these OOD features back to the original image, leading to high reconstruction error.
*   **In MVTec LOCO AD (Logical)**: Because the misplaced or extra object is locally normal, DINOv2 extracts **completely normal semantic features**. The decoder, being a highly expressive Transformer network, easily reconstructs these normal features.
*   **The Paradox**: The neural network **over-generalizes**; it reconstructs the logically invalid object perfectly because it knows how to reconstruct that object from the training set. Since the reconstruction error is low, the anomaly remains undetected.

### 2. DINOv2 Semantic Representation Bias vs. Geometric Precision
*   **Semantic Invariance**: DINOv2 is trained via self-supervised learning to be invariant to minor spatial deformations, scaling, and background noise. It represents *what* an object is (e.g., "a pushpin") exceptionally well.
*   **Counting & Coordinate Blindness**: DINOv2 features are soft and semantic. They do not enforce hard coordinates or count objects. 
*   **Result**: While this semantic invariance helps DiNomaly achieve **99.6% AUROC** on MVTec-AD (ignoring minor alignment noise), it prevents it from detecting when there are 4 pushpins instead of 5, leading to poor logical performance on `pushpins` (Image AUROC: 59.84%) and `screw_bag` (Image AUROC: 55.78%).

### 3. Linear Attention Bottleneck
*   **Softmax Attention**: Standard Transformers compute pairwise softmax similarity, which acts as a routing mechanism that can preserve exact spatial offsets.
*   **Linear Attention**: DiNomaly utilizes `LinearAttention2` (which computes $\phi(Q)\phi(K)^T V$). While this achieves linear efficiency ($O(N)$), it acts as a **spatial low-pass filter**.
*   **Loss of Relational Constraints**: The spatial smoothing of linear attention makes the decoder blind to exact geometric boundaries and counts. It can reconstruct a smooth layout of objects but fails to enforce strict logical checks (e.g., *"this metal pin must align exactly with this slot"*).

### 4. High Intra-class Variance in Training
*   **MVTec-AD**: Normal images are strictly aligned (rigid registration). The intra-class variance is near zero. The model can learn a very tight boundary of normal features.
*   **MVTec LOCO AD**: Normal images contain random configurations. For example, in `screw_bag`, the screws are placed randomly inside a transparent bag.
*   **Relayed Constraints**: To prevent false positives on these randomly sliding screws, the model's loss function must relax its constraints. This relaxation behaves like a low-pass filter, allowing anomalies (e.g., a missing screw) to be reconstructed within the "normal variation" boundary, resulting in a high rate of False Negatives.

---

## 5. Summary of Key Insights for Project Presentation

To summarize this comparison for your term project, use this structured logic:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. MVTec-AD is "Local Texture UAD"                                          │
│    - Defects are OOD locally.                                               │
│    - DiNomaly excels (99.6% AUROC) because local OOD features cannot be     │
│      reconstructed.                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. MVTec LOCO AD is "Global Relational UAD"                                 │
│    - Defect parts are ID locally, OOD globally.                             │
│    - DiNomaly drops (83.3% Combined) because the powerful decoder           │
│      over-generalizes and reconstructs locally normal OOD objects.          │
└─────────────────────────────────────────────────────────────────────────────┘
```

This contrast highlights that **unsupervised reconstruction-based models have a fundamental theoretical limit**: they cannot easily distinguish between a locally normal valid assembly and a locally normal invalid assembly without explicit relational graph modeling or object-counting constraints.
