INDIGO: INdependent DIssimilarity of GNN-embeddings for Outlier-querying
INDIGO is an efficient Active Learning strategy designed for Molecular Dynamics (MD) simulations. It aims to address the issue of insufficient generalization capability in Machine Learning Force Fields (MLFF) when facing Out-of-Distribution (OOD) samples.

Unlike traditional Query by Committee (QBC) methods, INDIGO does not require training multiple models. Instead, it utilizes the embedding features from the penultimate layer of a Graph Neural Network (GNN), combined with the Mahalanobis Distance, to measure the dissimilarity between the structure being predicted and the training set. This allows for the efficient selection of high-value samples for First-Principles (DFT) labeling.


Minimal Computational Overhead: Unlike QBC, INDIGO only requires training and maintaining a single model. The cost of calculating embedding features and Mahalanobis distance is negligible compared to the model inference itself.
High Compatibility: It is compatible with most modern GNN models that feature embeddings and a linear output layer (such as MACE, Allegro, SevenNet, etc.) without requiring changes to the model architecture.
Precise Outlier Detection: By leveraging physical information extracted by deep neural networks, it accurately captures changes in the local atomic environment (e.g., phase transitions, melting).
Efficient Data Selection: It achieves screening accuracy on par with mainstream QBC methods and even outperforms them when handling certain out-of-distribution samples.

The core idea of INDIGO is to strike a balance between classification based on model information and classification based on data distribution.

Feature Extraction: It extracts the network activation values from the layer preceding the energy summation layer (usually the penultimate layer) of the GNN model as embedding features. Information in this layer is highly correlated with the final output and contains rich physical information.
Dissimilarity Metric: It uses the Mahalanobis Distance to measure the similarity between the atomic environment in the current MD step and the atomic environments in the training set.
The Mahalanobis Distance normalizes and decorrelates the feature space via the covariance matrix, effectively solving distance metric issues in high-dimensional feature spaces.
Active Learning Loop:
Prediction: Use the current MLFF to predict energy and forces.
Judgment: Calculate the Mahalanobis Distance. If the distance exceeds the threshold, the model is considered uncertain about the current structure.
Labeling: Call DFT (e.g., VASP) to calculate ground truth labels.
Training: Trigger model fine-tuning once enough new structures have been accumulated.

