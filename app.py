import logging
from fastapi import FastAPI
from fastapi import HTTPException
from yolov8_class import ObjectDetectionProcessor
import os
from yolo_service_types import SingleFileRequest, BatchFolderRequest, SingleFileResponse, BatchFolderResponse


# Set up logging
logging.basicConfig(filename="object_detection_service.log", level=logging.INFO)

app = FastAPI()
YOLO_CUDA_DEVICE = int(os.getenv('YOLO_CUDA_DEVICE', 2))  # Use environment variable, or default to 2
objectDetectionProcessor = ObjectDetectionProcessor(cuda_device=YOLO_CUDA_DEVICE)

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


# if __name__ == "__main__":
#     YOLO_CUDA_DEVICE = int(os.getenv('YOLO_CUDA_DEVICE', 2))  # Use environment variable, or default to 2
#     objectDetectionProcessor = ObjectDetectionProcessor(cuda_device=YOLO_CUDA_DEVICE)
#     # YOLO_CUDA_DEVICE=3 uvicorn app:app --host 0.0.0.0 --port 8087 --workers 4