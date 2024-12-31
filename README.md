# Image ML Pod

**A modular framework to simplify your image dataset workflows.**

Image ML Pod is a ready-to-use framework designed to make image-based machine learning pipelines easier, faster, and more scalable. From preprocessing to training, inference, and deployment, this pod provides you with tools and templates to focus on building your models instead of managing workflows.

---

## **Why Image ML Pod?**

- **Prebuilt Kedro Pipelines**: Modular workflows for preprocessing, training, inference, and postprocessing.
- **Seamless Integration**: Built-in support for HuggingFace datasets, MLFlow, FastAPI, and Docker.
- **Cutting-Edge Features**:
  - Out-of-Distribution (OOD) detection.
  - Conformal predictions for reliability.
  - Explainability with Integrated Gradients.
- **Scalable Deployment**: Easily bootstrap APIs or explore microservices architecture.
- **Time-Saving**: Spend less time on setup and more on experimentation.

---

## **Key Features**

- **Modular Design**: Use only the pipelines you need, customize nodes, and add new ones effortlessly.
- **Automatic OOD Detection**: Ensure robustness with templates for MSP, RMD, and MultiMahalanobis detectors.
- **Experiment Tracking**: MLFlow integration lets you log hyperparameters, metrics, and models.
- **FastAPI Integration**: Bootstrap APIs directly from inference pipelines.
- **Docker Support**: Build and deploy your applications seamlessly with GPU compatibility.
- **Conformal Predictions**: Generate reliable prediction sets with torchcp.
- **Explainability**: Use Captum’s Integrated Gradients to interpret your model’s decisions.

---

## **How It Works**

### Framework Overview

1. **Data Handling**: HuggingFace datasets integration for seamless loading and processing of image datasets.
2. **Preprocessing**: Ready-to-use pipelines for image transforms, OOD detection, and data augmentation.
3. **Training**: Kedro pipelines with placeholders for custom models and training logic.
4. **Inference**: FastAPI server integration for real-time inference.
5. **Postprocessing**: Enhance predictions with conformal methods and explainability tools.

---

## **Demos**

1. **Prebuilt Pipelines**

   - Load an image dataset with HuggingFace’s `ImageFolder`.
   - Train a model and log results with MLFlow.
   - Deploy the inference pipeline as a FastAPI server.

2. **Example Commands**

   ```bash
   # Generate conformal predictions
   kedro run --pipeline=inf_pred_postprocessing

   # Launch the FastAPI server
   uvicorn src.image_ml_pod_fastapi.app:app --host 0.0.0.0 --port 8000
   ```

---

## **Customization**

### Adding Custom Nodes

- Modify existing Kedro nodes or add new ones in the pipeline YAML files.
- Use the provided templates for:
  - **Data Preprocessing**: Add torchvision transforms or custom logic.
  - **OOD Detection**: Train your own detectors.
  - **Postprocessing**: Implement explainability or custom logging.

### Extending Pipelines

- Add or remove nodes by editing the `catalog.yml` and pipeline configuration files.
- Example:
  ```yaml
  my_image_dataset:
    type: image_ml_pod.datasets.HFImageFolderDataSet
    data_dir: data/01_raw/images
  ```

---

## **Deployment**

### Running Locally

Run the FastAPI server for inference:

```bash
uvicorn src.image_ml_pod_fastapi.app:app --host 0.0.0.0 --port 8000
```

### Dockerized Deployment

Build the Docker image:

```bash
docker build -t image-ml-pod .
```

Run the Docker container with GPU support:

```bash
docker run -p 8000:8000 --gpus all image-ml-pod
```

---

## **Documentation**

To view the full documentation, visit the [docs](docs) folder.

Or you can access it by running the following command:

```bash
quarto preview
```

and navigating to the "Documentation" section.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

Built with love using Kedro, HuggingFace, MLFlow, FastAPI, and more. Special thanks to the open-source community for providing the tools that made this possible.
