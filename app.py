import logging
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from yolov8_class import ObjectDetectionProcessor
from typing import List
from pydantic import BaseModel
import uvicorn
import argparse
# Set up logging
logging.basicConfig(filename="object_detection_service.log", level=logging.INFO)

app = FastAPI()
objectDetectionProcessor = None


class SingleFileRequest(BaseModel):
    filename: str


class BatchFolderRequest(BaseModel):
    folder_path: str


class ObjectDetectionResult(BaseModel):
    name: str
    confidence: float
    box: List[int]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "person", "confidence": 0.95, "box": [10, 20, 100, 200]}
            ]
        }
    }


class SingleFileResponse(BaseModel):
    status: str
    results: List[ObjectDetectionResult]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "results": [
                        {
                            "name": "person",
                            "confidence": 0.95,
                            "box": [10, 20, 100, 200],
                        },
                        {
                            "name": "car",
                            "confidence": 0.85,
                            "box": [50, 60, 150, 250],
                        },
                    ],
                }
            ]
        }
    }


class BatchFolderConfidence(BaseModel):
    name: str
    confidence: float
    box: List[int]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "person", "confidence": 0.95, "box": [10, 20, 100, 200]}
            ]
        }
    }


class BatchFolderResult(BaseModel):
    file_path: str
    frame_number: int
    confidences: List[BatchFolderConfidence]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "file_path": "/path/to/folder/image1.jpg",
                    "frame_number": 1,
                    "confidences": [
                        {
                            "name": "person",
                            "confidence": 0.95,
                            "box": [10, 20, 100, 200],
                        },
                        {
                            "name": "car",
                            "confidence": 0.85,
                            "box": [50, 60, 150, 250],
                        },
                    ],
                }
            ]
        }
    }


class BatchFolderResponse(BaseModel):
    status: str
    results: List[BatchFolderResult]

    model_config = {
        "json_schema_extra": {
            "examples": [
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
                                    "box": [10, 20, 100, 200],
                                },
                                {
                                    "name": "car",
                                    "confidence": 0.85,
                                    "box": [50, 60, 150, 250],
                                },
                            ],
                        },
                        {
                            "file_path": "/path/to/folder/image2.jpg",
                            "frame_number": 2,
                            "confidences": [
                                {
                                    "name": "person",
                                    "confidence": 0.90,
                                    "box": [20, 30, 110, 210],
                                },
                                {
                                    "name": "car",
                                    "confidence": 0.80,
                                    "box": [60, 70, 160, 260],
                                },
                            ],
                        },
                    ],
                }
            ]
        }
    }


@app.post("/detect_single_file", response_model=SingleFileResponse)
async def detect_single_file(request: SingleFileRequest):
    try:
        filename = request.filename
        if not filename:
            raise HTTPException(status_code=400, detail="No filename in the request")
        results = objectDetectionProcessor.process_single_file(filename)
        return {"status": "success", "results": results}
    except Exception as e:
        logging.error(f"Error in SingleFileDetectionHandler: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred")


@app.post("/detect_batch_folder", response_model=BatchFolderResponse)
async def detect_batch_folder(request: BatchFolderRequest):
    try:
        folder_path = request.folder_path
        if not folder_path:
            raise HTTPException(status_code=400, detail="No folder_path in the request")
        results = objectDetectionProcessor.process_directory(folder_path)
        return {"status": "success", "results": results}
    except Exception as e:
        logging.error(f"Error in BatchFolderDetectionHandler: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Service")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--cuda_device", type=int, default=2, help="CUDA device number")
    objectDetectionProcessor = ObjectDetectionProcessor(cuda_device=2)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
