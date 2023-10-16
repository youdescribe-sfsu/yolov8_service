# Object Detection Service

This repository contains code for an object detection service that uses the YOLOv8 model for detecting objects in images. The service provides two endpoints: `detect_single_file` and `detect_batch_folder`.

## Installation

To use this service, you need to install the following dependencies:

- Python 3.x
- [FastAPI](https://fastapi.tiangolo.com/)
- [ultralytics](https://docs.ultralytics.com/)

You can install the dependencies by running the following command:

```sh
pip install -r requirements.ts
```

## Running the Service

You can run the service using Uvicorn. Here's an example command:
```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

You can also specify the CUDA device to use by setting the `CUDA_DEVICE` environment variable:

```sh
CUDA_DEVICE=1 uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```


## API Endpoints

The object detection service provides the following API endpoints:

### `docs`

The "docs" endpoint is part of the FastAPI framework and is used to serve the Swagger documentation for your API. It allows you to explore and interact with your API, view the available endpoints, and understand how to use them. This endpoint is particularly useful during the development and testing phases.

To access the "docs" endpoint, you typically open a web browser and navigate to the URL where your FastAPI application is running, followed by "/docs". For example, if your FastAPI application is running on "http://localhost:8087", you can access the documentation at "http://localhost:8087/docs".

### `detect_single_file`

This endpoint accepts a POST request with the following parameters:

- `filename`: The path to the image file to be processed.

**Example request:**

```sh
curl -X POST -d "filename=/path/to/single/image.jpg" http://localhost:8080/detect_single_file
```

**Example response:**

```json
{
  "status": "success",
  "results": [
    {
      "name": "person",
      "confidence": 0.95,
      "box": [10, 20, 100, 200]
    },
    {
      "name": "car",
      "confidence": 0.85,
      "box": [50, 60, 150, 250]
    }
  ]
}
```

### `detect_batch_folder`

This endpoint accepts a POST request with the following parameters:

- `folder_path`: The path to the folder containing the image files to be processed.

**Example request:**

```sh
curl -X POST -d "folder_path=/path/to/folder" http://localhost:8080/detect_batch_folder
```

**Example response:**

```json
{
  "status": "success",
  "results": [
    {
      "file_path": "/path/to/folder/image1.jpg",
      "frame_number": 1,
      "confidences": [
        {
          "name": "person",
          "confidence": 0.95,
          "box": [10, 20, 100, 200]
        },
        {
          "name": "car",
          "confidence": 0.85,
          "box": [50, 60, 150, 250]
        }
      ]
    },
    {
      "file_path": "/path/to/folder/image2.jpg",
      "frame_number": 2,
      "confidences": [
        {
          "name": "person",
          "confidence": 0.90,
          "box": [20, 30, 110, 210]
        },
        {
          "name": "car",
          "confidence": 0.80,
          "box": [60, 70, 160, 260]
        }
      ]
    }
  ]
}
```

## Logging

The service logs any errors that occur during the detection process to the `object_detection_service.log` file.