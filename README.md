# ImmersiveDepth: Advanced 360-Degree Depth Estimation

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Totoro97/PeRF)

<div align="center">
    <img src="MY---------------------" width="90%"/>
</div>

## Overview

ImmersiveDepth is a cutting-edge project designed to provide robust and accurate monocular depth estimation (MDE) from 360-degree images. By integrating advanced models such as MiDaS v3.1 and Depth Anything V2, this project addresses the challenges of depth perception under varying environmental conditions, scale variations, and image complexities. It leverages the strengths of both models to enhance overall performance, offering a comprehensive solution for applications in VR, 3D mapping, and data visualization.

## Features

- **Model Integration**: Combines MiDaS v3.1 and Depth Anything V2 to leverage the strengths of both models for improved depth prediction accuracy.
- **Tangent Image Projection**: Uses an icosahedron-based approach to project spherical images into perspective tangent images, allowing for precise depth map creation.
- **Global Disparity Alignment**: Aligns predicted disparity maps using advanced affine transformation techniques to ensure consistent global alignment.
- **Robust Preprocessing**: Employs a range of image enhancement techniques such as Gaussian filtering, CLAHE, and dehazing to improve depth map quality.
- **Efficient Workflow**: Optimized for GPU processing with minimal hardware requirements, providing faster and more efficient depth estimation.

## Table of Contents

- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [References](#references)

## Project Structure

├── 3D
│ ├── 3dopen.py
│ └── exportmesh.py
├── configs
│ └── device
│ ├── featureize.yaml
│ ├── local.yaml
│ └── nerf.yaml
├── example_data
│ ├── BU1
│ ├── BU2
│ ├── Matter1
│ ├── Matter2
│ └── kitchen
├── modules
│ ├── dataset
│ │ ├── pycache
│ │ ├── dataset.py
│ │ └── sup_info.py
│ ├── fields
│ │ └── networks.py
│ └── geo_predictors
│ └── ImmersiveDepth
│ ├── config
│ │ ├── depth.yaml
│ │ └── normal.yaml
│ ├── data
│ │ └── splits
│ │ ├── metadata_images_split_scene.csv
│ │ ├── test_hypersim_orig.csv
│ │ ├── train_hypersim_orig.csv
│ │ ├── train_val_test_blendedMVS.csv
│ │ ├── train_val_test_debug.csv
│ │ ├── train_val_test_full.csv
│ │ ├── train_val_test_fullplus.csv
│ │ ├── train_val_test_gso.csv
│ │ ├── train_val_test_hypersim.csv
│ │ ├── train_val_test_medium.csv
│ │ ├── train_val_test_replica.csv
│ │ ├── train_val_test_tiny.csv
│ │ └── val_hypersim_orig.csv
│ ├── ImmersiveDepth_dataset.py
│ ├── init.py
│ ├── augmentation.py
│ ├── masks.py
│ ├── refocus_augmentation.py
│ └── splits.py
│ ├── dataloader
│ │ ├── component_datasets
│ │ │ ├── blended_mvg
│ │ │ ├── hm3d
│ │ │ ├── hypersim
│ │ │ ├── replica
│ │ │ ├── replica_gso
│ │ │ └── taskonomy
│ │ ├── task_configs.py
│ │ ├── taskonomy_dataset.py
│ │ ├── transforms.py
│ │ ├── init.py
│ │ ├── ImmersiveDepth_dataset.py
│ │ ├── README.md
│ │ ├── masks.py
│ │ ├── pytorch3d_utils.py
│ │ ├── pytorch_lightning_datamodule.py
│ │ ├── scene_metadata.py
│ │ ├── segment_instance.py
│ │ ├── splits.py
│ │ ├── task_configs.py
│ │ ├── transforms.py
│ │ └── viz_utils.py
│ ├── losses
│ │ ├── init.py
│ │ ├── masked_losses.py
│ │ ├── midas_loss.py
│ │ └── virtual_normal_loss.py
│ ├── modules
│ │ ├── midas
│ │ │ ├── init.py
│ │ │ ├── base_model.py
│ │ │ ├── blocks.py
│ │ │ ├── dpt_depth.py
│ │ │ ├── midas_net.py
│ │ │ ├── midas_net_custom.py
│ │ │ ├── transforms.py
│ │ │ └── vit.py
│ │ ├── init.py
│ │ ├── channel_attention.py
│ │ └── unet.py
│ ├── tools
│ │ ├── download_depth_models.sh
│ │ └── download_surface_normal_models.sh
│ └── depth_anything_v2
│ ├── dinov2_layers
│ │ ├── init.py
│ │ ├── attention.py
│ │ ├── block.py
│ │ ├── drop_path.py
│ │ ├── layer_scale.py
│ │ ├── mlp.py
│ │ ├── patch_embed.py
│ │ └── swiglu_fn.py
│ └── util
│ ├── blocks.py
│ ├── transform.py
│ ├── init.py
│ ├── dinov2.py
│ ├── dpt.py
│ ├── ImmersiveDepth_normal_predictor.py
│ ├── ImmersiveDepth_predictor.py
│ ├── init.py
│ ├── app.py
│ ├── depth_model.py
│ ├── geo_predictor.py
│ ├── pano_fusion_inv_predictor.py
│ ├── pano_fusion_normal_predictor.py
│ ├── pano_geo_refiner.py
│ ├── pano_joint_predictor (copy).py
│ ├── pano_joint_predictor.py
│ ├── init.py
│ └── pre_checkpoints
│ ├── README.md
│ ├── utils
│ ├── camera_utils.py
│ ├── debug_utils.py
│ ├── geo_utils.py
│ ├── utils.py
│ ├── config.yaml
│ ├── core_exp_runner.py
│ └── requirements.txt




## Methodology

### 1. Initialization and Configuration

The initialization phase involves setting up the environment, loading configuration files, and preparing the system for depth estimation. Key files in this stage include:

- **`core_exp_runner.py`**: This script uses `Hydra` to load configurations from `config.yaml`, `depth.yaml`, and `normal.yaml`, setting parameters such as model paths and data directories. It initializes the `DepthEstimator` class, which manages the entire estimation process.

    ```python
    @hydra.main(config_path=".", config_name="config")
    def main(cfg: DictConfig):
        set_random_seed(cfg.get("seed", 0))
        depth_estimator = DepthEstimator(cfg)
        depth_estimator.estimate_depth()
    ```

- **`config.yaml`**: This file contains global settings for the project, including paths to models and datasets, as well as various processing options.

### 2. Data Loading and Preprocessing

This stage handles the loading of datasets and initial preprocessing of images, including resizing and normalization.

- **`dataset.py`**: Manages dataset loading and

 preprocessing. The `WildDataset` class reads panoramic images and associated reference data, performing resizing and normalization to prepare data for model input.

    ```python
    class WildDataset(Dataset):
        def __init__(self, image_path, image_resize, ref_dist_map=None, ref_normal_map=None):
            self.image_path = image_path
            self.image_resize = image_resize
            self.images = self._load_images()
            self.ref_dist_map = ref_dist_map
            self.ref_normal_map = ref_normal_map

        def _load_images(self):
            # Load images from the specified path
            images = []
            for img_file in sorted(os.listdir(self.image_path)):
                img_path = os.path.join(self.image_path, img_file)
                img = cv2.imread(img_path)
                images.append(img)
            return images
    ```

- **Data Augmentation**: Implemented in files like `transforms.py`, `augmentation.py`, and `refocus_augmentation.py`, these modules apply transformations such as resizing, normalization, and refocusing to enhance data diversity and model robustness.

    ```python
    class Resize(object):
        def __call__(self, sample):
            width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
            sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)
            return sample

    class NormalizeImage(object):
        def __call__(self, sample):
            sample["image"] = (sample["image"] - self.__mean) / self.__std
            return sample
    ```

### 3. Model Architecture and Processing

This phase leverages advanced model architectures for depth estimation, using Vision Transformers and multi-scale feature fusion.

- **Depth Prediction Models**: Files such as `dpt_depth.py`, `midas_net.py`, `midas_net_custom.py`, and `depth_model.py` implement models like `DPTDepthModel` and `MidasNet`, utilizing Vision Transformers for feature extraction and custom decoders for depth map generation.

    ```python
    class DPTDepthModel(DPT):
        def forward(self, x):
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
            path_4 = self.scratch.refinenet4(layer_4)
            path_3 = self.scratch.refinenet3(path_4, layer_3)
            path_2 = self.scratch.refinenet2(path_3, layer_2)
            path_1 = self.scratch.refinenet1(path_2, layer_1)
            out = self.scratch.output_conv(path_1)
            return out.squeeze(dim=1)
    ```

- **Vision Transformer Components**: Implemented in `vit.py`, these components transform image patches into sequences of tokens processed through transformer layers.

    ```python
    def forward_vit(pretrained, x):
        b, c, h, w = x.shape
        glob = pretrained.model.forward_flex(x)
        layer_1 = pretrained.activations["1"]
        layer_2 = pretrained.activations["2"]
        layer_3 = pretrained.activations["3"]
        layer_4 = pretrained.activations["4"]
        ...
    ```

- **Depth Anything V2 Model Components**: Implemented in the `depth_anything_v2/` directory, this module includes components like `attention.py`, `block.py`, `layer_scale.py`, and others, providing a robust architecture for depth estimation.

### 4. Prediction and Refinement

In this stage, the system predicts depth and normal maps, refining them for geometric consistency.

- **Joint Prediction**: Files such as `pano_joint_predictor.py`, `ImmersiveDepth_predictor.py`, and `ImmersiveDepth_normal_predictor.py` integrate models for depth and normal map prediction, combining outputs for enhanced results.

    ```python
    class PanoJointPredictor(GeoPredictor):
        def __init__(self):
            super().__init__()
            self.depth_predictor1 = ImmersiveDepthPredictor()  # Initialize first depth predictor
            self.depth_predictor2 = DepthAnything2(encoder='vitl')  # Initialize second depth predictor with Vision Transformer Large encoder
            self.normal_predictor = ImmersiveDepthNormalPredictor()  # Initialize normal predictor

        def __call__(self, img, ref_distance, mask, gen_res=384, ...):
            ...
            for i in range(n_pers):
                with torch.no_grad():
                    intri = {'fx': fx[i].item(), 'fy': fy[i].item(), 'cx': cx[i].item(), 'cy': cy[i].item()}
                    pred_depth1 = self.depth_predictor1.predict_depth1(pers_imgs[i: i+1], intri=intri).clip(0., None)
                    pred_depth2 = self.depth_predictor2.predict_depth2(img_pil).clip(0., None)
                    pred_normals = self.normal_predictor.predict_normal(pers_imgs[i: i+1])
                    ...
    ```

- **Geometry Refinement**: Implemented in `pano_geo_refiner.py` and `geo_utils.py`, this process refines depth and normal maps by optimizing geometric consistency, aligning predictions across multiple views, and iteratively improving results.

    ```python
    class PanoGeoRefiner:
        def refine(self, distances, normals):
            # Iteratively refine depth and normal maps for accuracy
            ...

    def align_scale(data, reference):
        # Aligns scales between data and reference for consistency
        ...
    ```

### 5. Mask Generation and Instance Segmentation

This step generates masks to indicate valid geometry regions and performs instance segmentation to extract and visualize object instances from images.

- **Mask Generation**: Implemented in `masks.py`, this module generates masks that identify valid geometry within the image, focusing models on important areas during prediction.

    ```python
    def make_mask(tensor, task):
        # Generates a mask to indicate valid pixels
        ...
    ```

- **Instance Segmentation**: Implemented in `segment_instance.py`, this module extracts object instances using techniques like bounding box extraction and mask application, supporting tasks like semantic segmentation and object recognition.

    ```python
    def extract_instances(img, return_masks=False):
        # Extracts instances from an image, returning bounding boxes, labels, and masks
        ...
    ```

### 6. Visualization and Output Handling

The final step involves visualizing and saving predicted depth and normal maps, allowing for evaluation and analysis of the results.

- **Visualization Utilities**: Implemented in `utils.py` and `viz_utils.py`, these modules visualize predicted depth and normal maps using functions like `show_batch_images`, providing graphical representations of the results.

    ```python
    def show_batch_images(batch, batch_idx, view_idxs=None, keys=('rgb', 'depth_euclidean'), figsize=None):
        # Visualizes a batch of images using Matplotlib
        ...
    ```

- **Output Management**: Saves results to disk using utilities like `write_image`, facilitating thorough evaluation and analysis of model outputs.

    ```python
    def write_image(save_path, image):
        # Saves an image to the specified path
        ...
    ```

### 7. Interactive Testing with Gradio

An interactive web-based interface allows users to test the model with uploaded images, providing real-time demonstration and feedback of depth estimation capabilities.

- **Gradio Interface**: Implemented in `app.py`, this module launches a `Gradio` interface for users to upload images and receive depth maps as output, demonstrating the system's effectiveness in real-time.

    ```python
    def predict_depth(image):
        return model.infer_image(image)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        gr.Markdown("### Depth Prediction demo")

        with gr.Row():
            input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
            depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
        submit = gr.Button(value="Compute Depth")
        gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
        raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download",)

        def on_submit(image):
            original_image = image.copy()
            depth = predict_depth(image[:, :, ::-1])
            raw_depth = Image.fromarray(depth.astype('uint16'))
            tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            raw_depth.save(tmp_raw_depth.name)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

            gray_depth = Image.fromarray(depth)
            tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            gray_depth.save(tmp_gray_depth.name)

            return [(original_image, colored_depth), tmp_gray_depth.name, tmp_raw_depth.name]

        submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file, raw_file])
    ```

## Installation

### Step 1: Clone the Repository

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/sarshardorosti/ImmersiveDepth.git
cd ImmersiveDepth
pip install -r requirements.txt
```

**Recommended Setup**: Python 3.9, CUDA 

12.1, and PyTorch 2.3.

### Step 2: Download Pre-trained Checkpoints

Download the necessary pre-trained model checkpoints as shown [here](./pre_checkpoints).

## Usage

### Training and Evaluation

#### Train on Example Data

Use the following command to train the model on example data:

```bash
python core_exp_runner.py --config-name nerf dataset.image_path=$(pwd)/example_data/hotel/hotel.png device.base_exp_dir=$(pwd)/exp
```

This command initializes the training process, loading the example dataset and preparing the system for depth estimation.


#### Open 3D Model

To visualize and interact with the generated 3D models, run the following command:

```bash
python core_exp_runner.py --config-name nerf dataset.image_path=$(pwd)/example_data/kitchen/image.png device.base_exp_dir=$(pwd)/exp mode=render_dense is_continue=true
```

This command processes the input image to generate and render a dense 3D model.


## Evaluation Metrics

The project's performance is evaluated using metrics such as Absolute Relative Error (AbsRel), Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and more. These metrics ensure that depth predictions are accurate and reliable, aligning closely with ground-truth values.

- **Absolute Relative Error (AbsRel)**: Measures the average of the absolute differences between the predicted depth and the ground-truth depth, relative to the ground-truth depth.

    \[
    \mathrm{AbsRel}=\frac{1}{N}\sum_{i=1}^{N}\frac{\left|z_i-z_i^\ast\right|}{z_i^\ast}
    \]

- **Mean Absolute Error (MAE)**: Calculates the average of the absolute differences between the predicted depth and the ground-truth depth.

    \[
    \mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}\left|z_i-z_i^\ast\right|
    \]

- **Root Mean Square Error (RMSE)**: Measures the square root of the average of the squared differences between the predicted depth and the ground-truth depth.

    \[
    \mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(z_i-z_i^\ast\right)^2}
    \]

- **Log Root Mean Square Error (RMSE(log))**: Similar to RMSE but operates in the logarithmic space, measuring the square root of the average of the squared logarithmic differences.

    \[
    \mathrm{RMSE(log)}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\log_{10}{z_i}-\log_{10}{z_i^\ast}\right)^2}
    \]

- **Square Relative Error (SqRel)**: Calculates the average of the squared differences between the predicted depth and the ground-truth depth, relative to the ground-truth depth.

    \[
    \mathrm{SqRel}=\frac{1}{N}\sum_{i=1}^{N}\frac{\left(z_i-z_i^\ast\right)^2}{z_i^\ast}
    \]


## Results

The developed system has demonstrated superior performance in creating precise distance maps from 360-degree images. By combining outputs from both MiDaS v3.1 and Depth Anything V2, the project achieves high accuracy and stability in depth predictions. The results indicate significant improvements over existing models, with enhanced detail recognition and resilience to environmental variations.


### Key Findings

- **Improved Accuracy**: The integrated model outperforms standalone models in various metrics, indicating enhanced depth prediction accuracy.
- **Efficient Processing**: Optimized for GPU processing, the system provides efficient depth estimation with reduced hardware requirements.
- **Comprehensive Solution**: The project offers a robust solution for depth estimation, applicable in VR, 3D mapping, and other domains.


## References

- Ainaz Eftekhar, Sax, A., Malik, J., & Zamir, A. (2021). Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans. 2021 IEEE/CVF International Conference on Computer Vision (ICCV). [arxiv.org](https://arxiv.org/abs/2110.04994)
- Birkl, R., Wofk, D., & Müller, M. (2023). MiDaS v3.1 -- A Model Zoo for Robust Monocular Relative Depth Estimation. [arxiv.org](https://arxiv.org/abs/2307.14460)
- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2022). Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence. [arxiv.org](https://arxiv.org/pdf/1907.01341v3.pdf)
- Rey-Area, M., Yuan, M., & Richardt, C. (2022). 360MonoDepth: High-Resolution 360° Monocular Depth Estimation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- Wang, G., Wang, P., Chen, Z., Wang, W., Loy, C. C., & Liu, Z. (2023). PERF: Panoramic Neural Radiance Field from a Single Panorama. [arxiv.org](https://arxiv.org/abs/2310.16831v2)
- Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything V2. [arxiv.org](https://arxiv.org/abs/2406.09414)




