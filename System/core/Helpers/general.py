
# Function to check if two bounding boxes overlap
import math

# Function to check if two bounding boxes overlap
def BoundingboxesOverlap(bb1, bb2):
    overlaps = False

    x1min = bb1[0]
    x1max = x1min + bb1[2]
    y1min = bb1[1]
    y1max = y1min + bb1[3]

    x2min = bb2[0]
    x2max = x2min + bb2[2]
    y2min = bb2[1]
    y2max = y2min + bb2[3]

    overlaps = (x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max)

    return overlaps

# Function to parse an annotation file
def ParseDatasetAnnotationFile(filepath):
    annotations = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            annotations.append(line.split())
    return annotations
        
# Function to check whether the vehicle is within range to be tracked
def FilterWithPadding(detections, imgSize, paddingPercentage):
    PadPxFromEdges = int(round((imgSize * (1-paddingPercentage)) / 2, 0))
    FilteredDetections = []
    for i in range (0, detections.__len__()):
        det = detections[i].tlwh
        # Get corners of bounding box
        det_tl = [det[0],det[1]]
        det_tr = [det[0]+det[2],det[1]]
        det_br = [det[0]+det[2],det[1]+det[3]]
        det_bl = [det[0],det[1]+det[3]]

        # Check that the vehicle is within range
        if not ((det_tl[0] >= PadPxFromEdges and det_tl[1] >= PadPxFromEdges) \
            and (det_tr[0] <= imgSize-PadPxFromEdges and det_tr[1] >= PadPxFromEdges) \
            and (det_br[0] <= imgSize-PadPxFromEdges and det_br[1] <= imgSize-PadPxFromEdges) \
            and (det_bl[0] <= imgSize-PadPxFromEdges and det_bl[1] >= PadPxFromEdges)):
            FilteredDetections.append(i)

    return FilteredDetections
