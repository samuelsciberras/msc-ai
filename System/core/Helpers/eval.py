# Sys Imports
import math
import copy
# 3rd Party Imports
from shapely.geometry import Polygon


# Convert Class ID to Name
def ConvertClassIdToName(id):
    id = int(id)
    if id == 0:
        return "Car"
    elif id == 1:
        return "SUV"
    elif id == 2:
        return "Bus"

# Convert Class Name to ID
def ConvertClassNameToId(name):
    if name == "Car":
        return 0
    elif name == "SUV":
        return 1
    elif name == "Bus":
        return 2

# Function to convert Yolo Annotation to Pixel TLWH
def ConvertYoloAnnotationTo_PixelTLWH(Annotation, imgSize):
    imgSize = float(imgSize)
    try:
        correctedAnnotation = copy.deepcopy(Annotation)
        # Convert to px coordinates <x-Center> <y-Center> <W> <H>
        correctedAnnotation[1] = int(round(float(Annotation[1]) * imgSize, 0))
        correctedAnnotation[2] = int(round(float(Annotation[2]) * imgSize, 0))
        correctedAnnotation[3] = int(round(float(Annotation[3]) * imgSize, 0))
        correctedAnnotation[4] = int(round(float(Annotation[4]) * imgSize, 0))
        # Convert X to X-(W/2), Y to Y-(H/2)
        correctedAnnotation[1] = correctedAnnotation[1] - (int(round(correctedAnnotation[3]/2)))
        correctedAnnotation[2] = correctedAnnotation[2] - (int(round(correctedAnnotation[4]/2)))
        # If anything is out of boundaries, set to boundary
        if correctedAnnotation[1] < 0: 
            correctedAnnotation = 0
        if correctedAnnotation[2] < 0: 
            correctedAnnotation = 0
        if correctedAnnotation[1] > imgSize: 
            correctedAnnotation = imgSize
        if correctedAnnotation[2] > imgSize: 
            correctedAnnotation = imgSize
        return correctedAnnotation
    except Exception as ex:
        print(ex)
        return None
        
# Calculate IoU Between two bounding boxes
def CalculateIoU(det, ann):
    # Setup Bounding box coordinates for each corner
    try:
        det_tl = [det[2][0],det[2][1]]
        det_tr = [det[2][0]+det[2][2],det[2][1]]
        det_br = [det[2][0]+det[2][2],det[2][1]+det[2][3]]
        det_bl = [det[2][0],det[2][1]+det[2][3]]
        bb1 = [det_tl, det_tr, det_br, det_bl]
        ann_tl = [ann[1],ann[2]]
        ann_tr = [ann[1]+ann[3],ann[2]]
        ann_br = [ann[1]+ann[3],ann[2]+ann[4]]
        ann_bl = [ann[1],ann[2]+ann[4]]
        bb2 = [ann_tl, ann_tr, ann_br, ann_bl]
        
        # Calculate IoU
        poly_1 = Polygon(bb1)
        poly_2 = Polygon(bb2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    except Exception as ex:
        print(ex)
        return None

# Function to evaluate image for object detection
# Returns (TP, FP, FN)
def EvaluateObjectDetection(IoUThreshold, Annotations, detections):
    # Initialize metrics
    tp = 0
    fp = 0
    fn = 0

    # Make copies of the passed lists to not modify original
    unmatchedDetections = copy.deepcopy(detections)
    unmatchedAnnotations = copy.deepcopy(Annotations)

    # Check that for each of the annotations, there is a detection with the same label
    for k in range (0, Annotations.__len__()):
        ann = ConvertYoloAnnotationTo_PixelTLWH(Annotations[k], 416)
        best_IoU = None
        best_IoU_idx = None
        # Iterate all detections to match the annotation
        for i in range (0, unmatchedDetections.__len__()):
            det = unmatchedDetections[i]
            # Class matches, proceed to calculate IoU
            if ConvertClassIdToName(int(ann[0])) == ConvertClassIdToName(int(det[0])):
                # Calculate IoU
                IoU = CalculateIoU(det, ann)
                # IoU is above threshold, proceed to update best_IoU details
                if IoU >= IoUThreshold:
                    if best_IoU is None or best_IoU < IoU:
                        best_IoU = IoU
                        best_IoU_idx = i 
        # Found the best IoU that matches this annotation, remove from unmatched annotations, and unmatched detections
        if best_IoU is not None:
            # Add True Positive
            tp = tp + 1
            unmatchedDetections.pop(best_IoU_idx)
            unmatchedAnnotations.pop(k)
            
    # Unmatched detections -> False Positive
    fp = unmatchedDetections.__len__()
    # Unmatched annotations -> False Negative
    fn = unmatchedAnnotations.__len__()
        
    return tp, fp, fn
    

# Function to evaluate a set of detections
def EvaluateDetections(FileName, IoUThreshold, Annotations, detections, imgSize):
    # Return: [float:ConfidenceScore, str:tp/fp]
    return_arr = []

    # Make copies of the passed lists to not modify original
    unmatchedDetections = copy.deepcopy(detections)
    unmatchedAnnotations = copy.deepcopy(Annotations)

    # Check that for each of the annotations, there is a detection with the same label
    for k in range (0, Annotations.__len__()):
        ann = ConvertYoloAnnotationTo_PixelTLWH(Annotations[k], imgSize)
        best_IoU = None
        best_IoU_idx = None
        # Iterate all detections to match the annotation
        for i in range (0, unmatchedDetections.__len__()):
            det = unmatchedDetections[i]
            # Class matches, proceed to calculate IoU
            if ConvertClassIdToName(int(ann[0])) == ConvertClassIdToName(int(det[0])):
                # Calculate IoU
                IoU = CalculateIoU(det, ann)
                # IoU is above threshold, proceed to update best_IoU details
                if IoU >= IoUThreshold:
                    if best_IoU is None or best_IoU < IoU:
                        best_IoU = IoU
                        best_IoU_idx = i 
        # Found the best IoU that matches this annotation, remove from unmatched annotations, and unmatched detections
        if best_IoU is not None:
            # True Positive
            return_arr.append((FileName, detections[best_IoU_idx][1], "tp"))
            unmatchedDetections.pop(best_IoU_idx)
            unmatchedAnnotations.pop(k)

    # For any remaining unmatched detections, set them as False Positives
    for unmatchedDetection in unmatchedDetections:
        return_arr.append((FileName, unmatchedDetection[1], "fp"))
        
    return return_arr