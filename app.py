import web
import json
import logging
from yolov8_class import ObjectDetectionProcessor

# Set up logging
logging.basicConfig(filename='object_detection_service.log', level=logging.INFO)

urls = (
    '/detect_single_file', 'SingleFileDetectionHandler',
    '/detect_batch_folder', 'BatchFolderDetectionHandler'
)

app = web.application(urls, globals())
objectDetectionProcessor = ObjectDetectionProcessor()

class SingleFileDetectionHandler:
    def POST(self):
        try:
            data = web.data()  # Use web.input() to get POST data
            if 'filename' not in data:
                return json.dumps({'status': 'error', 'message': 'No filename in the request'})
            file_path = data.filename
            results = objectDetectionProcessor.process_single_file(file_path)
            return json.dumps({'status': 'success', 'results': results})
        except Exception as e:
            logging.error(f"Error in SingleFileDetectionHandler: {str(e)}")
            return json.dumps({'status': 'error', 'message': 'An error occurred'})

class BatchFolderDetectionHandler:
    def POST(self):
        try:
            data = web.data()
            if 'folder_path' not in data:
                return json.dumps({'status': 'error', 'message': 'No folder_path in the request'})
            folder_path = data.folder_path
            results = objectDetectionProcessor.process_directory(folder_path)
            return json.dumps({'status': 'success', 'results': results})
        except Exception as e:
            logging.error(f"Error in BatchFolderDetectionHandler: {str(e)}")
            return json.dumps({'status': 'error', 'message': 'An error occurred'})

if __name__ == "__main__":
    app.run()
