# AN2DL-challenges-2025-26

## Challenge 1: Time Series Classification


This challenge involved classifying multivariate time series data to predict pain status (no pain, low pain, high pain) using Recurrent Neural Network architectures. Each sequence consisted of 160 measurements from multiple channels, optimizing for F1-score performance.

## Dataset

- **Training:** 661 labeled subjects
- **Test:** 1,323 unlabeled subjects
- **Features:**
  - **Joint curves:** 31 continuous body joint angle measurements over time
  - **Pain surveys:** 4 rule-based sensor aggregations (values: 0, 1, 2)
  - **Subject characteristics:** 3 categorical variables (missing limbs/eyes)


## Architecture

**Hybrid Multi-Branch Model:**

1. **Joint Features Branch:** Bidirectional LSTM with attention mechanism
2. **Pain Surveys Branch:** Separate LSTM module
3. **Static Features Branch:** Fully-connected layer for subject characteristics
4. **Fusion Layer:** Merged outputs before final classifier

**Total Parameters:** 1,090,248

| Model | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| Baseline (Majority Class) | 0.6741 | 0.5975 | 0.7730 |
| **Final Model** | **0.9501** | **0.9584** ± 0.0095 | **0.9576** ± 0.0092 |


## Conclusion

The final model achieved **0.9501 F1-score** on the public test set, representing significant improvement over the baseline. While competitive, state-of-the-art models in the competition exceeded 0.9700, indicating room for optimization through enhanced feature engineering, data augmentation, and architectural refinements.
On the private test set, it achieved a **0.96268 F1-Score**, boosting our ranking from 107th to 14th place.


---

## Challenge 2: Image Classification


This challenge involved classifying low-magnification Whole Slide Imaging (WSI) of human tissue into four molecular subtypes corresponding to potential diseases. The task utilized Convolutional Neural Network architectures, optimizing for F1-score performance.

## Dataset

- **Training:** 691 labeled samples
- **Test:** 477 unlabeled samples
- **Sample Structure:**
  - **Image:** RGB image (~1024×1024) of tissue
  - **Mask:** Binary mask identifying regions most likely to contain diseased tissue

### Data Cleaning

Initial inspection revealed corrupted images (Shrek memes and green-stained artifacts), which were removed from the training set.

### Patch Extraction Strategy

To focus on informative regions and reduce background noise:
- Images divided into **128×128 patches** with stride of 64 pixels
- Retained only patches where mask contained ≥70% positive values
- At least one patch per image guaranteed (highest mask overlap selected)
- Prevents information loss while maintaining fine-grained details

![Patch extraction visualization showing mask and cropped patches]

## Architecture

**Backbone:** ConvNeXt-Tiny (pre-trained on ImageNet)

ConvNeXt modernizes traditional CNN architecture by integrating Vision Transformer design principles:
- Layer normalization
- GELU activations
- Inverted bottleneck blocks
- Retains convolutional inductive biases

**Classifier:** Linear layer mapping 768 features → 4 classes


## Results

| Model | Validation F1 | Test F1 |
|-------|---------------|---------|
| Random Classifier | --- | 0.2932 |
| SimpleCNN | 0.3081 | 0.1612 |
| EfficientNet-B0 | 0.3194 | 0.3535 |
| EfficientNet-B1 | 0.3224 | 0.3212 |
| EfficientNet-B2 | 0.3045 | 0.3200 |
| ResNet18 | 0.4485 | 0.4162 |
| ResNet50 | 0.3856 | 0.3803 |
| **ConvNeXt-Tiny** | **0.4269** | **0.4300** |
| ConvNeXt-Small | 0.4558 | 0.4171 |
| ConvNeXt-Base | 0.4039 | 0.4011 |

**Final Test F1-Score: 0.4300** (+0.1368 over random baseline)

### Per-Class Performance

The model shows variable performance across breast cancer subtypes:

- **Luminal B:** Best performance, often predicted with high confidence
- **HER2(+) & Luminal A:** Frequent high-confidence errors (overconfidence issue)
- **Triple Negative:** Severe bottleneck with extremely low recall (0.21)

## Discussion

The model's performance is primarily limited by the **Triple Negative subtype** (recall=0.21), which remains the bottleneck despite weighted cross-entropy. The overconfidence in HER2(+) and Luminal A predictions suggests the need for better calibration techniques.


## Conclusions

The final model achieved an **F1-score of 0.4300**, significantly outperforming the random baseline (0.2932) but falling short of top competition performance (0.46+). The model successfully learns morphological patterns but is severely limited by the Triple Negative class and shows overconfidence in certain predictions.


---
