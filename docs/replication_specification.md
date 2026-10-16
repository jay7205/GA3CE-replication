# GA3CE Replication Specification

This document serves as the authoritative source of truth for all mathematical formulations, architectural hyperparameters, and data preprocessing rules extracted from the CVPR 2025 GA3CE paper and its supplementary material. 

**Rule:** Implementation must strictly follow these values. Any item marked as "Unresolved" must be investigated and resolved before code implementation for that component begins.

---

## 1. Coordinate Conventions & Reference Points
* **$t_{pose}$**: Defined as the average 3D position of the head keypoints (indices 24 to 29).
* **$t_{object}$**: Defined via backprojection using the center of the head bounding box $b_{head}$. 
  * **[UNRESOLVED]**: The exact coordinate frame and origin reference for $t_{object}$ requires source verification from the original baseline code to prevent offsets.

## 2. Pose Normalization
* **Scale ($s$)**: Computed as the reciprocal of the L2 norm between the 24th and 28th keypoints (which defines the head's width).
* **Keypoint Subsampling**: The 30 keypoints from the `smpl+head_30` format are subsampled using the exact indices `{0, 4–8, 12, 18–21, 24, 26, 28, 29}`, resulting in $N_{pose} = 15$.

## 3. Cyclotorsion Rotation (Egocentric Transformation)
To align the view direction $v = [v_x, v_y, v_z]$ to $z = [0, 0, 1]^T$ while minimizing roll ($\psi = 0$), the rotation $R = Euler(\theta, \phi, 0)$ is calculated as:
* $\phi = \text{arctan2}(-v_x, v_z)$
* $\theta = \text{arctan2}(v_y, v_z \cos\phi - v_x \sin\phi)$

The core object transformation is: $P'_{object} = R(P_{object} - t_{object})$

## 4. D3 Positional Encoding
* **Equation**: γ̃(p) = γ(p / ||p||) ⊕ γ(||p||).

* **Encoding function**:
  * The paper defines γ as the standard positional encoding used for the D3 formulation.
  * **[UNRESOLVED]**: The exact implementation and frequency parameters of γ are not specified in the supplementary material and must be verified from the cited/reference implementation.
  * Do not assume or invent the frequency bands, bandwidth, or maximum frequency.
* **Dimensions**:
  * $C_{dir\_keypoint} = 6$, $C_{dist\_keypoint} = 4 \implies C_{keypoint} = 10$.
  * $C_{dir\_latent} = 128$, $C_{dist\_latent} = 128 \implies C_{latent} = 256$.
* **Frequencies**: 
  * **[UNRESOLVED]**: The exact frequency bandwidth, maximum frequency, and number of frequency bands are not specified in the supplementary material (only broadly cited to Tancik et al.). This must be verified in the reference implementation.

## 5. Transformer Architecture
* **Module**: Standard PyTorch Transformer module (`torch.nn.Transformer`).
* **Encoder**: $N_{encoder} = 3$ layers.
* **Decoder**: $N_{decoder} = 3$ layers.
* **Dimensions**: Feedforward network dimension = 512.
* **Attention**: Multi-head attention uses 2 heads.
* **Other**: All other hyperparameters follow default PyTorch settings.

## 6. Gaze-Cone Attention Bias
* **Definition**: An additive attention bias $B \in \mathbb{R}^{N_{object}}$ added to the object features $F_{object}$ during cross-attention.
* **Equation**: Each element of the bias is defined exactly as the **cosine similarity** between $v$ and $p_{object} \in P_{object}$.
* No thresholds, scaling factors, or sign modifiers are specified or should be invented.

## 7. Maximum Object Positions & Masking
* **GFIE & CAD-120**: Object sequence padded to $N_{max\_object} = 168$.
* **GAFA**: Object sequence padded to $N_{max\_object} = 278$.
* **Masking**: Non-existent object positions are masked out during multi-head attention.

## 8. Residual Gaze MLP & Reconstruction
* **MLP**: The residual gaze direction $g'$ is decoded using a two-layer MLP with 512 hidden units and ReLU activation.
* **Reconstruction**:
  * **[UNRESOLVED]**: The exact transformation from the predicted residual gaze direction g' to the final camera-space gaze direction g must be verified from the original implementation/source.
  * This verification must establish the exact coordinate frame, the role of R and v, and whether any explicit unit-normalization is applied.

## 9. Training Configuration
* **Hardware**: Single A10G GPU.
* **Batch Size**: 32 for GFIE, 64 for GAFA.
* **Epochs**: 20.
* **Optimizer**: AdamW.
* **Learning Rate**: 0.0014.
* **Scheduler**: Cosine scheduling with a 4-epoch warm-up.
* **Regularization**: Weight decay = 0.1. Gradient clipping with an L2-norm threshold of 0.1.
* **GFIE-Only View Augmentation**: Noise $\epsilon \in [-0.5, 0.5]^3$ is added to $v$. $v_{noise} = (v + \epsilon) / ||v + \epsilon||$. (Not applied for GAFA).

## 10. Third-Party / Pretrained Dependencies
* **MeTRAbs Pose Estimator**: Frozen weights. Requires isolated TF inference.
* **ZoeDepth**: 
  * **[UNRESOLVED]**: The exact ZoeDepth variant and checkpoint used for GAFA is not specified in the supplementary material.
* **GAFA Appearance Estimator**: 7-temporal-frame Head and Body Network.

## 11. Baseline Modifications (Ablations Only)
* **Rule**: For the modified baseline models (`GFIE+3D` / `GAFA+3D`), object positions are normalized by $t_{object}$ **and scaled so that their largest extent equals one**. This rule does *not* apply to the core GA3CE architecture.
