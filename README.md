# PCA for Face Recognition

This project demonstrates the use of Principal Component Analysis (PCA) and its variants for face recognition. PCA is a powerful statistical technique that reduces the dimensionality of data while preserving as much variance as possible, making it particularly effective for image datasets.

## Overview

PCA identifies the principal components that capture the greatest variability in data. These components are orthogonal, ensuring they represent distinct features of the data, such as pixel variations in images.

### Implementations

#### 1. Traditional PCA
- **Setup**: The initial implementation involves setting up the image matrix, labels, and defining the quality percentage of variance to retain.
- **Mean Centering**: Each image is centered around the mean face by subtracting the mean from the original image matrix.
- **Dimensionality Reduction**: Singular Value Decomposition (SVD) is used to decompose the image matrix into principal components. A function calculates the number of components needed to retain the desired variance.
- **Projection**: Images are projected onto the new bases formed by these principal components.
- **Recognition**: A test image is projected onto the same principal components used in training. The `recognize_face` function computes the Euclidean distance between the test face's PCA coordinates and the mean coordinates of each class, returning the closest class.

#### 2. 2D PCA
- **Concept**: This method considers the natural 2D structure of images instead of flattening them into 1D vectors. By preserving spatial relationships between pixels, 2D PCA enhances recognition accuracy.
- **Implementation**: The mean face is computed by averaging across all columns of the image matrix. Each column (image vector) is mean-centered by subtracting the mean face. The covariance matrix is then computed from these mean-centered columns.
- **Dimensionality Reduction**: Eigenvalues and eigenvectors of the covariance matrix identify the principal components. Components are selected based on their eigenvalues to retain a significant percentage of variance, such as 95%.
- **Performance**: On the ORL dataset, 2D PCA achieves 2% higher accuracy than traditional PCA due to better spatial preservation.

#### 3. 2D Squared PCA
- **Concept**: Extends 2D PCA by considering covariance along both rows and columns more intricately. This approach captures detailed spatial relationships within the image data.
- **Implementation**: Covariance matrices are computed separately for rows and columns, capturing relationships in both dimensions. Eigenvalues and eigenvectors of these matrices determine the principal components representing significant variations.
- **Dimensionality Reduction**: This dual approach allows for a nuanced reduction in dimensionality while retaining crucial spatial information.
- **Performance**: 2D Squared PCA preserves structural information inherent in the pixel layout, providing a more accurate representation of image variations.

## Comparison of Methods

- **PCA**: Treats images as flat vectors, potentially losing important spatial information. Useful but limited in capturing complex spatial structures.
- **2D PCA**: Considers rows and columns separately, preserving more spatial structure and relationships, leading to improved feature extraction.
- **2D Squared PCA**: Analyzes covariance matrices along both dimensions for a more detailed capture of spatial features, outperforming other methods in recognition accuracy.

## Limitations and Considerations

- **Memory Usage**: Retaining more information requires storing higher-dimensional data, consuming more memory and potentially leading to scalability issues.
- **Computational Cost**: Higher dimensionality increases the computational expense of operations like covariance matrix computation and eigen decomposition.
- **Risk of Overfitting**: Retaining too much information can lead to overfitting, where the model captures noise and irrelevant features.

## Future Directions

- **Enhancement of PCA Models**: Consider improving the PCA model by maintaining more spatial structures. Experiment with different dimensionality reduction techniques that balance between preserving information and computational efficiency.
- **Scalability**: Address memory and processing constraints for larger datasets or higher-resolution images.

## Getting Started

1. Clone the repository: 
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```
2. Follow the instructions in each file (PCA.py, TwoD_PCA.py, TwoD_Squared_PCA.py) to understand the implementation details and how to run the code.
3. Experiment with different datasets and parameters to see the impact on recognition accuracy.

