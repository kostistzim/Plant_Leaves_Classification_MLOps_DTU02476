# Plant_Leaves_Classification

An MLOps (DTU02476) project work.

## Group 96 - Team Members

Kostis Tzimoulias - s242796@dtu.dk
Xhino Mullaymeri - s223481@dtu.dk
Dimitris Papantzikos - s242798@dtu.dk
Jan Šulíček - s246733@dtu.dk
Michail Dikaiopoulos - s242816@dtu.dk

## Goal

Our goal for the project is to implement a binary classification model for classifying a dataset of plant leaves as being in healthy or diseased condition. We aim to use a pretrained Convolutional Neural Network implementation and finetune it to achieve the best results. Our main focus is to deliver and deploy a model that is well organized, fully reproducible, easy-to-interact with and aligned with the overall direction of the course. To do that, we are going to use all the tools and techniques that we are currently learning in the course.

## Framework

We will be using PyTorch, with the addition of other third party libraries. PyTorch Image Models (timm) contains pre-trained image classification neural networks. Torchvision will be used to handle image augmentation, both to enhance the size of the dataset and to make the model more robust in the end. Hydra provides us an elegant way to handle our hyperparameters.

## Data

As for the dataset, we are going to use [Plant Leaves for Image Classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification) from Kaggle.The dataset contains 4502 images of healthy and unhealthy plant leaves with an approximate size of 7.3GBs. We will start by scaling down the images from (6000*4000) to the model's proper input dimensions. In case we notice that the size of the dataset requires computational expensive training, we will utilize a subset of the initial data. On the other hand, in case that the dataset is small for the task, we will augment it by utilizing frameworks such as torchvision or TissueImageAnalytics/tiatoolbox.

## Models

We are going to use CNN based model(s), more specifically [timm/efficientnet_b1.ra4_e3600_r240_in1k - Hugging Face](https://huggingface.co/timm/efficientnet_b1.ra4_e3600_r240_in1k).The idea is to use the CNN body of this pretrained model and blend it with our own fully connected classification layer(s). This will allow us to at least partially utilize the pre-trained weights of the model, while the custom fully connected layer handles the specific needs of our problem.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


# Project Setup and Commands

> Please go through the commands one by one.

## Setup Commands

### 1. Create a New Conda Environment
This command creates a new Conda environment for the project with the specified Python version.

`invoke create-environment`

### 2. Install Project Requirements
This command installs the required dependencies for the project.

`invoke requirements`

### 3. Install Development Requirements
This command installs the development dependencies for the project.

`invoke dev-requirements`

### 4. Install Pre-commit Hooks
This command installs the pre-commit hooks for the project.

`invoke precommit`

## Project Commands

### 1. Download Data
This command downloads the project dataset.

`invoke download-data`

### 2. Preprocess Data
This command preprocesses the raw data into processed data.

`invoke preprocess-data`

### 3. Train Model
This command trains the model for the project.

`invoke train`

## Deployment

### **Train**:
#### Local deployment:
1. Build the train image locally by running: `docker build -f dockerfiles/<your_dockerfile> -t <your_image_name:tag> .` (e.g. `docker build -f dockerfiles/train.dockerfile -t plants/train:v1.0 .`)
2. Run the container locally: `docker run --name <experiment_name> <your_image_name:tag>` (e.g. `docker run --name train_experiment plants/train:v1.0`)
#### Google Cloud deployment:
1. Build the train image on Google Cloud: `gcloud builds submit --config=<your_cloudbuild_config> .` (e.g. `gcloud builds submit --config=configs/cloud/cloudbuild.yaml .`)
2. Run the container with VertexAI: `gcloud ai custom-jobs create --region=<choose_region> --display-name=<choose_name> --config=<local_config_path>` (e.g. `gcloud ai custom-jobs create --region=europe-west1 --display-name=test-run --config=configs/cloud/vertex_config_cpu.yaml`). This expects a GCP Storage with the name `oxygen-o2` and the data to exist in the path `gcs/oxygen-o2/data/processed`.

### **Application**:
#### Local deployment:
- Simply use the docker-compose file by running `docker-compose up`.
#### Google Cloud deployment:
1. Build the train images of both frontend and backend on Google Cloud: `gcloud builds submit --config=<your_cloudbuild_config> .` (e.g. `gcloud builds submit --config=configs/cloud/cloudbuild_frontend.yaml .`)
2. Run the containers from Cloud Run in GCP by navigating to Cloud Run webpage and creating a new service, specifying the latest image builds that were triggered either manually or from Github Actions. We managed to make the frontend and backend communicate by letting the `BACKEND_URI` be an environment variable and setting it upon creation of the frontend in the Cloud Run webpage.


## API Overview
- **`root/` Endpoint**: Serves as a health check to ensure the API is up and running.
- **`predict/` Endpoint**: Handles POST requests, accepts a user's PNG image as input, and returns a JSON object containing:
  - `image_label`: The predicted label for the image.
  - `confidence`: The confidence score of the prediction.
  - `status_code`: The HTTP response code.
- **`metrics` Endpoint**: Displays several dev-defined metrics regarding latency, number of calls, hits/misses etc.


<!--### 4. Run Tests
This command runs the tests for the project and generates a coverage report.

invoke test

### 5. Build Docker Images
This command builds the Docker images for the project, one for training and one for the API.

invoke docker-build

## Documentation Commands

### 1. Build Documentation
This command builds the project documentation using MkDocs.

invoke build-docs

### 2. Serve Documentation
This command serves the project documentation locally using MkDocs.

invoke serve-docs -->
