import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import argparse
import logging
import os
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObjectDetectionSystem:
    """Enhanced real-time object detection system using YOLOv8 with GPU optimization."""

    def __init__(self, model_path: str = "yolov8x.pt", conf_threshold: float = 0.5,
                 use_cuda: bool = True, nms_threshold: float = 0.45,
                 half_precision: bool = True, config_file: str = None):
        """
        Initialize the detection system with GPU optimizations.

        Args:
            model_path: Path to the YOLOv8 model weights (using larger model for accuracy)
            conf_threshold: Confidence threshold for detections
            use_cuda: Whether to use CUDA for GPU acceleration
            nms_threshold: Non-maximum suppression threshold
            half_precision: Whether to use FP16 precision for faster inference
            config_file: Optional path to configuration JSON file
        """
        # Load configuration if provided
        config = {}
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    import json
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")

        # Apply config values or use defaults
        self.conf_threshold = config.get('conf_threshold', conf_threshold)
        self.nms_threshold = config.get('nms_threshold', nms_threshold)
        self.half_precision = config.get('half_precision', half_precision)
        model_path = config.get('model_path', model_path)

        # Target performance metrics
        self.target_fps = config.get('target_fps', 30.0)

        # Force CUDA to be visible
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Verify CUDA installation and print diagnostic info
        cuda_available = torch.cuda.is_available()
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available (PyTorch check): {cuda_available}")

        if cuda_available:
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

        # Device configuration - force CUDA if available
        if cuda_available and use_cuda:
            self.device = torch.device("cuda:0")
            self.cuda_available = True
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # Convert to GB
            logger.info(f"Using GPU: {gpu_name} with {gpu_mem:.2f}GB memory")
            # Set optimal CUDA settings for performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Additional check to ensure CUDA is working
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(self.device)
            logger.info(f"Test tensor device: {test_tensor.device}")
        else:
            if not cuda_available and use_cuda:
                logger.error(
                    "CUDA is not available despite PyTorch CUDA installation. Please check your NVIDIA drivers.")
            self.device = torch.device("cpu")
            self.cuda_available = False
            logger.warning("Using CPU. Performance will be limited.")

        # Load YOLOv8 model
        try:
            # Initialize YOLO model with explicit device setting - first without half precision
            self.model = YOLO(model_path)

            # Apply GPU optimizations
            if self.cuda_available:
                # Set model to GPU explicitly
                self.model.to(self.device)

                # Enable half-precision (FP16) for faster inference if supported
                if self.half_precision:
                    try:
                        # Try to convert to half precision
                        logger.info("Attempting to use half precision (FP16) for faster inference")
                        self.model.half()
                        # Test inference with half precision
                        dummy_input = torch.zeros((1, 3, 640, 640), dtype=torch.float16).to(self.device)
                        with torch.no_grad():
                            _ = self.model(dummy_input)
                        logger.info("Half precision (FP16) successfully enabled")
                    except Exception as e:
                        logger.warning(f"Half precision failed, falling back to full precision: {e}")
                        self.half_precision = False
                        # Reload the model in full precision
                        self.model = YOLO(model_path)
                        self.model.to(self.device)

            logger.info(f"Model loaded from {model_path} to {self.device} " +
                        f"using {'half' if self.half_precision else 'full'} precision")

            # Warm up the model to ensure CUDA initialization is complete
            if self.cuda_available:
                dtype = torch.float16 if self.half_precision else torch.float32
                dummy_input = torch.zeros((1, 3, 640, 640), dtype=dtype).to(self.device)
                with torch.no_grad():
                    for _ in range(3):  # Multiple warm-up runs
                        _ = self.model(dummy_input)
                torch.cuda.synchronize()  # Ensure warm-up is complete
                logger.info("Model warm-up complete")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Color palette for different classes (more visually appealing)
        self.color_palette = self._generate_color_palette()

        # Initialize variables for FPS calculation and performance metrics
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps_values = []  # Store recent FPS values for smoothing
        self.inference_times = []  # Store inference times for monitoring

        # Performance metrics
        self.total_frames = 0
        self.detection_count = 0

        # For heatmap visualization
        self.heatmap = None
        self.heatmap_alpha = 0.3
        self.show_heatmap = False

    def _generate_color_palette(self, num_colors: int = 80) -> List[Tuple[int, int, int]]:
        """Generate a visually distinct color palette for object visualization."""
        np.random.seed(42)  # For reproducibility
        colors = []
        for _ in range(num_colors):
            # Generate vibrant colors with good contrast (avoid dark colors)
            color = (
                int(np.random.randint(100, 256)),  # R
                int(np.random.randint(100, 256)),  # G
                int(np.random.randint(100, 256))  # B
            )
            colors.append(color)
        return colors

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model inference with GPU optimization.

        Args:
            frame: Input frame as numpy array

        Returns:
            Preprocessed tensor ready for inference
        """
        # The YOLO model handles preprocessing internally, but we can
        # ensure the frame is in optimal format for performance
        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List, float]:
        """
        Process a single frame for object detection with GPU optimization.

        Args:
            frame: The input frame as numpy array

        Returns:
            Tuple containing the processed frame, detections, and inference time
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame")
            return None, [], 0.0

        # Start timing for inference
        inference_start = time.time()

        # Create a copy for visualization
        visualization_frame = frame.copy()

        # Perform prediction with GPU optimization
        try:
            # Turn off gradient calculation for inference
            with torch.no_grad():
                # Ensure model is on correct device
                if self.cuda_available:
                    try:
                        if next(self.model.parameters()).device != self.device:
                            self.model.to(self.device)
                    except (StopIteration, AttributeError):
                        logger.warning("Couldn't check model device - model may not have accessible parameters")

                # Run inference with appropriate settings
                results = self.model(frame, conf=self.conf_threshold, iou=self.nms_threshold,
                                     verbose=False, device=0 if self.cuda_available else 'cpu')

            # Ensure CUDA operations are complete
            if self.cuda_available:
                torch.cuda.synchronize()

            # Calculate inference time
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 30:
                self.inference_times.pop(0)

            # Process results
            detections = []
            for r in results:
                boxes = r.boxes
                self.detection_count += len(boxes)

                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    })

                    # Get color for this class
                    color = self.color_palette[cls_id % len(self.color_palette)]

                    # Draw enhanced bounding box with gradient effect
                    cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), color, 2)

                    # Add filled background for text
                    text_size = cv2.getTextSize(f"{cls_name}: {conf:.2f}",
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(visualization_frame,
                                  (x1, y1 - text_size[1] - 10),
                                  (x1 + text_size[0] + 10, y1),
                                  color, -1)

                    # Add label with class name and confidence
                    cv2.putText(visualization_frame, f"{cls_name}: {conf:.2f}",
                                (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2)

                    # Update heatmap if enabled
                    if self.show_heatmap and self.heatmap is not None:
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(self.heatmap, (center_x, center_y), 30, (0, 0, 255), -1)

            # Calculate and display FPS
            self.curr_frame_time = time.time()
            fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
            self.prev_frame_time = self.curr_frame_time

            # Smooth FPS calculation
            self.fps_values.append(fps)
            if len(self.fps_values) > 10:
                self.fps_values.pop(0)
            avg_fps = sum(self.fps_values) / len(self.fps_values)

            # Calculate average inference time
            avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0

            # Create performance overlay
            self.total_frames += 1

            # Display GPU memory usage if using CUDA
            gpu_mem_info = ""
            if self.cuda_available:
                mem_allocated = torch.cuda.memory_allocated(0) / 1024 ** 2  # MB
                mem_reserved = torch.cuda.memory_reserved(0) / 1024 ** 2  # MB
                gpu_mem_info = f"GPU Mem: {mem_allocated:.0f}MB/{mem_reserved:.0f}MB"

            # Draw enhanced status bar with performance metrics
            cv2.rectangle(visualization_frame, (0, 0), (450, 120), (0, 0, 0), -1)
            cv2.putText(visualization_frame, f"FPS: {avg_fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization_frame, f"Inference: {avg_inference_time * 1000:.1f}ms", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization_frame, f"Detections: {len(detections)}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw GPU/CPU indicator
            device_text = f"GPU: {torch.cuda.get_device_name(0)}" if self.cuda_available else "CPU"
            text_color = (0, 255, 255) if self.cuda_available else (0, 165, 255)
            cv2.putText(visualization_frame, device_text, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            # Display GPU memory info if available
            if gpu_mem_info:
                cv2.putText(visualization_frame, gpu_mem_info,
                            (visualization_frame.shape[1] - 250, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return visualization_frame, detections, inference_time

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, [], 0.0

    def optimize_model_settings(self):
        """Dynamically adjust model settings based on performance metrics."""
        if not self.fps_values:
            return

        avg_fps = sum(self.fps_values) / len(self.fps_values)

        # If using GPU and performance is below target, try to optimize
        if self.cuda_available:
            if avg_fps < self.target_fps * 0.7:  # FPS is significantly below target
                if not self.half_precision:
                    # Switch to half precision
                    try:
                        self.half_precision = True
                        self.model.half()
                        logger.info("Switched to half precision for improved performance")
                    except Exception as e:
                        logger.error(f"Failed to switch to half precision: {e}")

    def get_performance_metrics(self) -> dict:
        """Get system performance metrics."""
        metrics = {
            "avg_fps": sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0,
            "avg_inference_time": sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0,
            "total_frames": self.total_frames,
            "total_detections": self.detection_count,
            "device": "GPU" if self.cuda_available else "CPU",
            "model": "YOLOv8"
        }

        # Add GPU metrics if available
        if self.cuda_available:
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
            metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / 1024 ** 2  # MB
            metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved(0) / 1024 ** 2  # MB

        return metrics

    def save_config(self, config_file: str) -> bool:
        """Save current configuration to file."""
        try:
            import json
            config = {
                'conf_threshold': self.conf_threshold,
                'nms_threshold': self.nms_threshold,
                'half_precision': self.half_precision,
                'target_fps': self.target_fps
            }

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
            return False

    def setup_video_recorder(self, output_path: str, fps: float, width: int, height: int):
        """Set up video recorder for saving processed frames."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def cleanup(self):
        """Release resources and clean up memory."""
        if self.cuda_available:
            torch.cuda.empty_cache()

    def run_detection(self, source: str = "0", record: bool = False,
                      output_path: str = "output/detection_output.mp4") -> None:
        """
        Run the detection system with OpenCV UI.

        Args:
            source: Video source (camera index, video file path, or RTSP URL)
            record: Whether to record output video
            output_path: Path to save recorded video if record=True
        """
        # Try to convert source to integer for webcam index
        try:
            source = int(source)
        except ValueError:
            pass

        video_recorder = None

        # Initialize video capture
        try:
            cap = cv2.VideoCapture(source)

            # Try opening the camera with a timeout
            read_timeout = time.time() + 5.0  # 5 seconds timeout
            success = False
            while time.time() < read_timeout:
                if cap.isOpened():
                    success = True
                    break
                time.sleep(0.1)

            if not success:
                raise ValueError(f"Failed to open video source: {source}")

            logger.info(f"Successfully opened video source: {source}")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Initialize heatmap
            self.heatmap = np.zeros((height, width, 3), dtype=np.uint8)

            # Set up video recorder if needed
            if record:
                video_recorder = self.setup_video_recorder(output_path, fps, width, height)
                logger.info(f"Recording video to {output_path}")

            # Create control panel/instructions overlay
            controls_text = [
                "Controls:",
                "ESC/Q - Quit",
                "+ - Increase confidence threshold",
                "- - Decrease confidence threshold",
                "S - Save screenshot",
                "H - Toggle help display",
                "R - Toggle recording",
                "M - Toggle heatmap",
                "C - Clear heatmap"
            ]

            show_help = True
            current_frame = None
            is_recording = record
            read_timeout = 5.0  # seconds

            while True:
                # Read frame with timeout handling
                frame_read_success = False
                start_read_time = time.time()

                while time.time() - start_read_time < read_timeout:
                    ret, frame = cap.read()
                    if ret:
                        frame_read_success = True
                        break
                    time.sleep(0.01)  # Short sleep to prevent CPU hogging

                if not frame_read_success:
                    logger.warning("Failed to receive frame within timeout period, stream may have ended")
                    break

                # Process frame
                processed_frame, detections, inference_time = self.process_frame(frame)
                current_frame = processed_frame.copy()

                # Draw help overlay if enabled
                if show_help:
                    # Create semi-transparent overlay for instructions
                    help_overlay = processed_frame.copy()
                    overlay_height = len(controls_text) * 25 + 20
                    cv2.rectangle(help_overlay,
                                  (width - 350, 10),
                                  (width - 10, overlay_height + 10),
                                  (0, 0, 0), -1)

                    # Add instructions text
                    for i, text in enumerate(controls_text):
                        cv2.putText(help_overlay, text,
                                    (width - 340, 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Blend overlay with main frame
                    alpha = 0.7
                    processed_frame = cv2.addWeighted(help_overlay, alpha, processed_frame, 1 - alpha, 0)

                # Combine with heatmap if enabled
                if self.show_heatmap and self.heatmap is not None:
                    # Apply Gaussian blur for smoother heatmap
                    blurred_heatmap = cv2.GaussianBlur(self.heatmap, (21, 21), 0)
                    # Blend with frame
                    processed_frame = cv2.addWeighted(processed_frame, 1.0, blurred_heatmap, self.heatmap_alpha, 0)

                # Display the frame
                cv2.imshow('Enhanced Object Detection', processed_frame)

                # Record video if enabled
                if is_recording and video_recorder is not None:
                    video_recorder.write(current_frame)

                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord('q'):  # ESC or Q
                    logger.info("Exiting...")
                    break

                elif key == ord('+'):  # Increase confidence threshold
                    self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
                    logger.info(f"Confidence threshold increased to {self.conf_threshold:.2f}")

                elif key == ord('-'):  # Decrease confidence threshold
                    self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                    logger.info(f"Confidence threshold decreased to {self.conf_threshold:.2f}")

                elif key == ord('s'):  # Save screenshot
                    screenshot_dir = "screenshots"
                    os.makedirs(screenshot_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{screenshot_dir}/detection_{timestamp}.jpg"
                    cv2.imwrite(filename, current_frame)
                    logger.info(f"Screenshot saved as {filename}")

                    # Show confirmation overlay
                    confirmation_overlay = current_frame.copy()
                    cv2.putText(confirmation_overlay, f"Screenshot saved: {filename}",
                                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Enhanced Object Detection', confirmation_overlay)
                    cv2.waitKey(1000)  # Show confirmation for 1 second

                elif key == ord('h'):  # Toggle help
                    show_help = not show_help

                elif key == ord('r'):  # Toggle recording
                    is_recording = not is_recording
                    if is_recording and video_recorder is None:
                        output_dir = os.path.dirname(output_path)
                        if output_dir:
                            os.makedirs(output_dir, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        new_output_path = f"output/detection_{timestamp}.mp4"
                        video_recorder = self.setup_video_recorder(new_output_path, fps, width, height)
                        logger.info(f"Started recording to {new_output_path}")
                    elif not is_recording and video_recorder is not None:
                        logger.info("Stopped recording")

                elif key == ord('m'):  # Toggle heatmap
                    self.show_heatmap = not self.show_heatmap
                    logger.info(f"Heatmap visualization: {'ON' if self.show_heatmap else 'OFF'}")

                elif key == ord('c'):  # Clear heatmap
                    self.heatmap = np.zeros((height, width, 3), dtype=np.uint8)
                    logger.info("Heatmap cleared")

                elif key == ord('p'):  # Save current configuration
                    if self.save_config("detection_config.json"):
                        logger.info("Configuration saved to detection_config.json")
                        # Show confirmation
                        cv2.putText(current_frame, "Configuration saved",
                                    (width // 2 - 100, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.imshow('Enhanced Object Detection', current_frame)
                        cv2.waitKey(1000)

        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
        finally:
            # Clean up
            if 'cap' in locals() and cap.isOpened():
                cap.release()

            if video_recorder is not None:
                video_recorder.release()

            cv2.destroyAllWindows()
            self.cleanup()
            logger.info("Detection ended")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Real-time Object Detection System with GPU Optimization")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (camera index, video file, or RTSP URL)")
    parser.add_argument("--model", type=str, default="yolov8x.pt",
                        help="Path to YOLOv8 model weights (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold for detections")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA (use CPU)")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable half precision (FP16) inference")
    parser.add_argument("--record", action="store_true",
                        help="Record output video")
    parser.add_argument("--output", type=str, default="output/detection_output.mp4",
                        help="Output video path (if recording)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration JSON file")
    args = parser.parse_args()

    # Create and run detection system
    try:
        detector = ObjectDetectionSystem(
            model_path=args.model,
            conf_threshold=args.conf,
            use_cuda=not args.no_cuda,
            half_precision=not args.no_half,
            config_file=args.config
        )

        detector.run_detection(source=args.source, record=args.record, output_path=args.output)

    except KeyboardInterrupt:
        logger.info("Detection stopped by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")