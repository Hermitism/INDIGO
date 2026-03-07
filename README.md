# INDIGO: INdependent DIssimilarity of GNN-embeddings for Outlier-querying

**INDIGO** is a novel **Active Learning** strategy designed for on-the-fly Molecular Dynamics (MD) simulations. It addresses the challenge of insufficient generalization in Machine Learning Force Fields (MLFF) when encountering Out-of-Distribution (OOD) samples.

Unlike traditional Query by Committee (QBC) methods which require training multiple models, INDIGO utilizes **embedding features** from the penultimate layer of a single Graph Neural Network (GNN). By calculating the **Mahalanobis Distance**, it measures the dissimilarity between the current atomic structure and the training set to efficiently select high-value samples for First-Principles (DFT) labeling.

##  Key Features

*   ** Minimal Computational Overhead**: Only requires training and maintaining a single model. The cost of calculating embedding features and Mahalanobis distance is negligible compared to model inference.
*   ** High Compatibility**: Compatible with most modern GNN models that feature embeddings and linear output layers (e.g., MACE, Allegro, SevenNet) without architectural changes.
*   ** Precise Outlier Detection**: Leverages physical information extracted by deep neural networks to accurately capture changes in local atomic environments (e.g., phase transitions, melting).
*   ** Efficient Data Selection**: Achieves screening accuracy on par with mainstream QBC methods and outperforms them on certain OOD samples.

##  How It Works

INDIGO strikes a balance between model-based uncertainty and data distribution analysis:

1.  **Feature Extraction**: Extracts network activation values from the layer preceding the energy summation layer (usually the penultimate layer). This layer contains rich physical information highly correlated with the final output.
2.  **Dissimilarity Metric**: Uses **Mahalanobis Distance** to measure the similarity between the current MD step's atomic environment and the training set. This normalizes and decorrelates the high-dimensional feature space.
3.  **Active Learning Loop**:
    *   **Predict**: The MLFF predicts energy and forces.
    *   **Query**: If the Mahalanobis Distance exceeds a `threshold`, the model is deemed uncertain.
    *   **Label**: The system calls a DFT calculator (e.g., VASP) for ground truth labels.
    *   **Train**: Once enough new structures (`num_stru_train`) are accumulated, the model is fine-tuned.

##  Configuration

Key hyperparameters for INDIGO-driven MD simulations:

| Parameter | Description | Suggestion |
| :--- | :--- | :--- |
| `threshold` | **Mahalanobis Distance Threshold**. Values > `threshold` trigger DFT labeling; values < `threshold` use MLFF inference. | Adjust based on desired accuracy vs. cost trade-off. |
| `num_stru_train` | **Training Trigger**. The number of accumulated labeled structures required to start model fine-tuning. | Dependent on available compute resources. |
| `k` | **Replay Ratio**. The proportion of old training data sampled and mixed with new data to prevent catastrophic forgetting. | Recommended > 0. |
| `sample_interval` | **Sampling Interval**. Minimum steps between adding two sampled structures to the training set to avoid overfitting. | - |
| `skip_iter` | **Warm-up Steps**. Number of initial steps to force MLFF usage, avoiding frequent DFT calls if the initial model is unstable. | Use if initial model accuracy is decent. |
