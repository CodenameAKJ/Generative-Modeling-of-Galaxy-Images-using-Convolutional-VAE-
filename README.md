# Generative-Modeling-of-Galaxy-Images-using-Convolutional-VAE-


###  Project Goal
The goal of this project is to develop a **Convolutional Variational Autoencoder (VAE)** that learns a robust, low-dimensional latent space representation of complex galactic morphologies from the **Sloan Digital Sky Survey (SDSS)** data.  
This latent space forms the foundation for **efficient data compression** and **unsupervised anomaly detection** in astronomical imaging.

---

### Project Status

| Component          | Status                                            | Next Step                                                                 |
|--------------------|---------------------------------------------------|---------------------------------------------------------------------------|
| **Data Pipeline**  |  Complete (Loading, Normalization, Batching)     | N/A                                                                       |
| **VAE Model**      |  Complete (Custom CNN Encoder/Decoder)           | N/A                                                                       |
| **Training **       |  Complete (Loss minimization & Latent space adherence) | N/A                                                                 |
| **Hyper parameter tuning**       | In Progress                        | N/A                                                                 |
| **Anomaly Scoring**|  In Progress                                     | Implement calculation of anomaly scores using reconstruction loss.        |

---

###  Core Achievements & Technical Focus

- **Developed a Convolutional Variational Autoencoder (VAE)** for generative modeling and analysis of galaxy image data, processing over **200,000 images**.  
- **Modeled Disentangled Latent Space:** Implemented a custom CNN encoder-decoder to learn continuous, disentangled latent features of galactic morphology.  
- **Applied Mathematical Constraints:** Utilized **Kullback–Leibler (KL) divergence** loss to ensure the latent space adheres to a Gaussian prior, enabling smooth interpolation.  
- **Dimensionality Reduction & Anomaly Framework:** The learned latent space enables efficient dimensionality reduction and forms the basis for anomaly detection using reconstruction probability.  
- **VAE Design Trade-Offs:** Model prioritizes reconstruction of major morphological features over fine textures, highlighting trade-offs between fidelity and generalization.

---

###  Technologies Used

| Category | Tools & Libraries |
|-----------|-------------------|
| **Deep Learning** | Python, TensorFlow 2.x (Keras) |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Version Control** | Git |

---

### 🪐 Data Source

- **Dataset:** Sloan Digital Sky Survey (SDSS) galaxy images  
- **Size:** ~200,000 processed galaxy images  
- **Link:** [https://www.sdss.org/]  

---

###  Future Work

- Implement **Anomaly Scoring Module** to identify statistical outliers based on reconstruction probability.  
- Explore **latent space interpolation** for galaxy morphology transitions.  
- Evaluate **performance metrics** and visualize learned representations using t-SNE/UMAP.

---



