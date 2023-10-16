# Object Detection Service

This repository contains code for an object detection service that uses the YOLOv8 model for detecting objects in images. The service provides two endpoints: `detect_single_file` and `detect_batch_folder`.

## Installation

To use this service, you need to install the following dependencies:

- Python 3.x
- web.py
- ultralytics
- logging

You can install the dependencies by running the following command:

```sh
pip install -r requirements.ts
```

## API Endpoints

The object detection service provides the following API endpoints:

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