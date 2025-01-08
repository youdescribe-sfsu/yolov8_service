import json
import time
from ultralytics import YOLO
import re
from yolo_service_types import ObjectDetectionResult

class ObjectDetectionProcessor:
    def __init__(self, model_path='yolov8m.pt', cuda_device=2):
        print(f"Initializing ObjectDetectionProcessor with model: {model_path} on CUDA device: {cuda_device}")
        self.model = YOLO(model_path)
        self.cuda_device = cuda_device
        print("ObjectDetectionProcessor initialized successfully")

    def extract_frame_number(self, file_path):
        match = re.search(r'frame_(\d+).jpg', file_path)
        if match:
            return int(match.group(1))
        else:
            return 0

    def process_object(self, detected_object):
        try:
            name = detected_object['name']
            confidence = detected_object['confidence']
            box = (
                detected_object['box']['x1'],
                detected_object['box']['y1'],
                detected_object['box']['x2'],
                detected_object['box']['y2']
            )
            return ObjectDetectionResult(name=name, confidence=confidence, box=list(box))
        except KeyError as e:
            print(f"KeyError: {str(e)} in detected_object")
            return None

    def process_results_and_sort_by_filepath(self, results):
        print(f"Processing and sorting results for {len(results)} images")
        processed_results = []

        for result in results:
            try:
                path = result.path
                json_obj = json.loads(result.to_json())

                return_arr = [self.process_object(detected_object) for detected_object in json_obj]
                frame_number = self.extract_frame_number(path)

                processed_results.append({
                    "confidences": return_arr,
                    "file_path": path,
                    "frame_number": frame_number
                })

            except Exception as e:
                print(f"An error occurred while processing result: {str(e)}")

        sorted_results = sorted(processed_results, key=lambda x: x.get('frame_number', 0))
        print(f"Sorted {len(sorted_results)} results")
        return sorted_results

    def process_directory(self, input_folder, conf_threshold=0.25):
        print(f"Processing directory: {input_folder} with confidence threshold: {conf_threshold}")
        try:
            results = self.model.predict(source=input_folder, conf=conf_threshold, save_txt=False, device=self.cuda_device, verbose=False)
            print(f"YOLO model prediction completed. Processing results...")
            processed_objects = self.process_results_and_sort_by_filepath(results)
            print(f"Processed {len(processed_objects)} objects")
            return processed_objects
        except Exception as e:
            print(f"An error occurred in process_directory: {str(e)}")
            return []

    def process_multiple_files(self, input_files, conf_threshold=0.25):
        try:
            print("input_files :: ",input_files)
            results = self.model.predict(source=input_files,conf=conf_threshold, save_txt=False, device=self.cuda_device,verbose=False)
            processed_objects = self.process_results_and_sort_by_filepath(results)
            return processed_objects
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return []

    def process_single_file(self, input_file, conf_threshold=0.25):
        try:
            results = self.model.predict(source=[input_file],conf=conf_threshold, save_txt=False, device=self.cuda_device)
            if( len(results) > 0 and results[0].path == input_file):
                processed_objects = self.process_results_and_sort_by_filepath(results)
                return processed_objects[0]['confidences']
            else:
                return []
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return []

    def run_and_save_results(self, input_folder, output_json_file):
        start_time = time.time()
        return_data = self.process_directory(input_folder)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for directory processing: {elapsed_time} seconds")

        with open(output_json_file, 'w') as outfile:
            json.dump(return_data, outfile, indent=4)

        print("Directory processing done")

    def run_and_save_single_file(self, input_file, output_json_file):
        start_time = time.time()
        return_data = self.process_single_file(input_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for single file processing: {elapsed_time} seconds")

        with open(output_json_file, 'w') as outfile:
            json.dump(return_data, outfile, indent=4)

        print("Single file processing done")


if __name__ == "__main__":
    processor = ObjectDetectionProcessor()
    
    # Batch processing of a directory
    processor.run_and_save_results("/home/datasets/pipeline/6dxSsZ_Perw_files/frames/*.jpg", "data_yolov8n.json")
    
    # Processing a single file
    processor.run_and_save_single_file("/path/to/single/image.jpg", "single_file_yolov8n.json")
