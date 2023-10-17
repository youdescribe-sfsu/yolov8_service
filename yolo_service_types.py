from pydantic import BaseModel
from typing import List


class SingleFileRequest(BaseModel):
    filename: str
    threshold: float
    model_config = {
        "json_schema_extra": {
            "examples": [{"filename": "/path/to/folder/image1.jpg", "threshold": 0.05}]
        }
    }

class MultipleFileRequest(BaseModel):
    files_path: List[str]
    threshold: float
    
    model_config = {
        "json_schema_extra": {
            "examples": [{"files_path": ["/path/to/folder/image1.jpg","/path/to/folder/image2.jpg"], "threshold": 0.05}]
        }
    }

class BatchFolderRequest(BaseModel):
    folder_path: str
    threshold: float

    model_config = {
        "json_schema_extra": {
            "examples": [{"folder_path": "/path/to/folder/*", "threshold": 0.05}]
        }
    }


class ObjectDetectionResult(BaseModel):
    name: str
    confidence: float
    box: List[float]

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
    box: List[float]

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
