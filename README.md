
# ImmersiveDepth: Advanced 360-Degree Depth Estimation

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Totoro97/PeRF)
<!--![visitors](https://visitor-badge.glitch.me/badge?page_id=Totoro97/PeRF)-->
<tr>
    <img src="YOUR_IMAGE_LINK_HERE" width="90%"/>
</tr>

## Overview

ImmersiveDepth is a state-of-the-art project that aims to provide robust and accurate monocular depth estimation (MDE) from 360-degree images. By integrating advanced models like MiDaS v3.1 and Depth Anything V2, the project addresses challenges such as scale variations, complex lighting conditions, and the inherent distortion of 360-degree imagery. The methodology leverages the strengths of both models to create precise distance maps for applications in VR, 3D mapping, and data visualization.

## Key Features

- **Model Integration**: Combines MiDaS v3.1 and Depth Anything V2 to leverage the strengths of both models for improved depth prediction accuracy.
- **Tangent Image Projection**: Uses an icosahedron-based approach to project spherical images into perspective tangent images, allowing for precise depth map creation.
- **Global Disparity Alignment**: Aligns predicted disparity maps using advanced affine transformation techniques to ensure consistent global alignment.
- **Robust Preprocessing**: Employs a range of image enhancement techniques such as Gaussian filtering, CLAHE, and dehazing to improve depth map quality.
- **Efficient Workflow**: Optimized for GPU processing with minimal hardware requirements, providing faster and more efficient depth estimation.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sarshardorosti/ImmersiveDepth.git
cd ImmersiveDepth
pip install -r requirements.txt



Recommended Setup: Python 3.9, CUDA 12.1, and PyTorch 2.3.

Step 2: Download Pre-trained Checkpoints
Download the necessary pre-trained model checkpoints as shown here.

Usage
Train on Example Data
Use the following command to train the model on example data:
python core_exp_runner.py --config-name nerf dataset.image_path=$(pwd)/example_data/hotel/hotel.png device.base_exp_dir=$(pwd)/exp



Open 3D Model
To visualize and interact with the generated 3D models, run the following command:
python core_exp_runner.py --config-name nerf dataset.image_path=$(pwd)/example_data/kitchen/image.png device.base_exp_dir=$(pwd)/exp mode=render_dense is_continue=true



Methodology
Initialization and Configuration
The initialization phase sets up the environment, loads configuration files, and prepares the system for depth estimation. It defines parameters such as dataset paths, model configurations, and hyperparameters.

core_exp_runner.py: Utilizes Hydra for dynamic configuration management, allowing easy switching between experimental setups via config.yaml, depth.yaml, and normal.yaml. Sets a random seed for reproducibility and creates an instance of DepthEstimator to manage the entire depth estimation process.
Data Loading and Preprocessing
This phase involves loading datasets and performing initial preprocessing on input images. It handles reading images, resizing, normalizing pixel values, and preparing them for model input.

dataset.py: Manages the loading of panoramic images and associated reference data (distance and normal maps). Preprocesses images by resizing and normalizing them.

Data Augmentation and Transformation Modules: Includes transforms.py, augmentation.py, and refocus_augmentation.py, which apply transformations such as resizing, normalization, and refocusing to enhance dataset diversity and model robustness.

Multi-Task Dataset Handling: ImmersiveDepth_dataset.py supports multi-task learning with datasets like Taskonomy and Replica, ensuring each sample has matching tasks across different domains.

Model Architecture and Processing
This step leverages advanced model architectures for depth estimation, including feature extraction using Vision Transformers, multi-scale feature fusion, and depth prediction from spherical and perspective images.

Depth Prediction Models: Includes dpt_depth.py, midas_net.py, midas_net_custom.py, and depth_model.py. Implements models like DPTDepthModel and MidasNet for depth prediction, utilizing Vision Transformers for feature extraction and custom decoders for depth map generation.

Vision Transformer Components: vit.py utilizes Vision Transformers (ViTs) for feature extraction, transforming image patches into sequences of tokens processed through transformer layers.

Depth Anything V2 Model Components: depth_anything_v2/ directory implements components like attention.py, block.py, layer_scale.py, and others to provide a robust architecture for depth estimation tasks.

Prediction and Refinement
This stage involves predicting depth and normal maps and refining them for geometric consistency. It includes integrating multiple models for enhanced prediction and optimizing results through refinement techniques.

Depth and Normal Prediction: pano_joint_predictor.py, ImmersiveDepth_predictor.py, ImmersiveDepth_normal_predictor.py integrate models for depth and normal map prediction, combining outputs for enhanced results.

Geometry Refinement: pano_geo_refiner.py, geo_utils.py refine depth and normal maps by optimizing geometric consistency, aligning predictions across multiple views, and iteratively improving results.

Mask Generation and Instance Segmentation
This step involves generating masks to indicate valid geometry regions and performing instance segmentation to extract and visualize object instances from images.

Mask Generation: masks.py, segment_instance.py generate masks identifying valid geometry within the image, focusing models on important areas during prediction.
Visualization and Output Handling
The final step involves visualizing and saving predicted depth and normal maps, allowing for evaluation and analysis of the results.

Visualization Utilities: utils.py, viz_utils.py visualize predicted depth and normal maps using functions like show_batch_images, providing graphical representations of the results.
Interactive Testing with Gradio
An interactive web-based interface allows users to test the model with uploaded images, providing real-time demonstration and feedback of depth estimation capabilities.

Gradio Interface: app.py launches a Gradio interface for users to upload images and receive depth maps as output, demonstrating the system's effectiveness in real-time.
Results and Discussion
Depth Detection System from 360-Degree Images
The developed system has demonstrated superior performance in creating precise distance maps from 360-degree images. By combining outputs from both MiDaS v3.1 and Depth Anything V2, the project achieves high accuracy and stability in depth predictions. The results indicate significant improvements over existing models, with enhanced detail recognition and resilience to environmental variations.

Key Findings
Improved Accuracy: The integrated model outperforms standalone models in various metrics, indicating enhanced depth prediction accuracy.
Efficient Processing: Optimized for GPU processing, the system provides efficient depth estimation with reduced hardware requirements.
Comprehensive Solution: The project offers a robust solution for depth estimation, applicable in VR, 3D mapping, and other domains.
Visualization and Analysis
The final depth maps and normal maps are saved in accessible formats for further analysis and visualization. The project supports 3D model generation, enabling detailed examination of depth data in a visual context.

References
Ainaz Eftekhar, Sax, A., Malik, J., & Zamir, A. (2021). Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans. 2021 IEEE/CVF International Conference on Computer Vision (ICCV). arxiv.org
Birkl, R., Wofk, D., & Müller, M. (2023). MiDaS v3.1 -- A Model Zoo for Robust Monocular Relative Depth Estimation. arxiv.org
Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2022). Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence. arxiv.org
Rey-Area, M., Yuan, M., & Richardt, C. (2022). 360MonoDepth: High-Resolution 360° Monocular Depth Estimation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Wang, G., Wang, P., Chen, Z., Wang, W., Loy, C. C., & Liu, Z. (2023). PERF: Panoramic Neural Radiance Field from a Single Panorama. arxiv.org
Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything V2. arxiv.org """
