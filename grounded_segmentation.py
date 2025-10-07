"""Use grounding DINO + Segment Anything (SAM) to perform grounded segmentation on an image.

Based on: https://github.com/IDEA-Research/Grounded-Segment-Anything
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


@dataclass
class BoundingBox:
    """Bounding box representation."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        """Return bounding box coordinates.

        Returns:
            List[float]: coodinates: [xmin, ymin, xmax, ymax]

        """
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    """Detection result from Grounding DINO + Mask from SAM."""

    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        """Create a DetectionResult from a dictionary.

        Args:
            detection_dict (Dict): Detection result dictionary.

        Returns:
            DetectionResult: Detection result object.

        """
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """Use OpenCV to refine a mask by turning it into a polygon.

    Args:
        mask (np.ndarray): Segmentation mask.

    Returns:
        List[List[int]]: List of (x, y) coordinates representing the vertices of the polygon.

    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    return largest_contour.reshape(-1, 2).tolist()


def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """Convert a polygon to a segmentation mask.

    Args:
        polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
        np.ndarray: Segmentation mask with the polygon filled.

    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def load_image(image_str: str) -> Image.Image:
    """Load an image from a URL or file path.

    Args:
        image_str (str): URL or file path to the image.

    Returns:
        PIL.Image: Image object.

    """
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image


def _get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def _refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


device = "cuda" if torch.cuda.is_available() else "cpu"
detector_id = "IDEA-Research/grounding-dino-tiny"
print(f"load object detector pipeline: {detector_id}")
object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

segmenter_id = "facebook/sam-vit-base"
print(f"load segmentator: {segmenter_id}")
segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
print(f"load processor: {segmenter_id}")
processor = AutoProcessor.from_pretrained(segmenter_id)


def detect(image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion."""
    global object_detector, device
    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    return [DetectionResult.from_dict(result) for result in results]


def segment(
    image: Image.Image, detection_results: List[Dict[str, Any]], polygon_refinement: bool = False
) -> List[DetectionResult]:
    """Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes."""
    global segmentator, processor, device
    boxes = _get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks, original_sizes=inputs.original_sizes, reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = _refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results


def grounded_segmentation(
    image: Union[Image.Image, str], labels: List[str], threshold: float = 0.3, polygon_refinement: bool = False
) -> Tuple[torch.Tensor, List[DetectionResult]]:
    """Segment out the objects in an image given a set of labels.

    Args:
        image (Union[Image.Image, str]): Image to load/work on.
        labels (List[str]): Object labels to segment.
        threshold (float, optional): Segmentation threshold. Defaults to 0.3.
        polygon_refinement (bool, optional): Use polygon refinement on the segmented mask? Defaults to False.

    Returns:
        Tuple[torch.Tensor, List[DetectionResult]]: Image tensor and list of detection results.

    """
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold)
    if len(detections) == 0:
        return ToTensor()(image), []
    detections = segment(image, detections, polygon_refinement)

    return ToTensor()(image), detections
