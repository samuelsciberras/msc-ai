
# Sys Imports
import math
import copy
# Custom Imports
import core.Helpers.general as generalfunctions

fps = 30

# Function to estimate speed
def EstimateSpeed(AllDetections, TrackedDetection):
    if AllDetections.__len__() < 2:
        return 0, None

    ResultingSpeed = 0
    # Setup a list of detecections for this specific track
    AllTrackDetections = []
    for det in AllDetections:
        if det.Id == TrackedDetection.Id:
            AllTrackDetections.append(det)
    AllTrackDetections.append(TrackedDetection)
    
    if AllTrackDetections.__len__() <= 1:
        return ResultingSpeed, None

    # Find the frame where the detection does not overlap the detection from current frame
    CurrentFrame = AllTrackDetections[-1]
    PreviousFrame = None
    ReversedDetectionList = reversed(AllTrackDetections)
    for det in ReversedDetectionList:
        if not generalfunctions.BoundingboxesOverlap(CurrentFrame.BoundingBox_tlwh, det.BoundingBox_tlwh):
            PreviousFrame = det
            break
    if PreviousFrame is not None:
        ResultingSpeed = EstimateSpeed_NoOverlap(CurrentFrame, PreviousFrame)
    
        
    return round(ResultingSpeed, 0), PreviousFrame

# Function to get Y coordinate at given X coordinate (pixels)
def GetYCoordinateAtX(slope, X, point_x, point_y):
    return ((slope * X) - (slope * point_x) + point_y)
    
# Function to get X coordinate at given Y coordinate (pixels)
def GetXCoordinateAtY(slope, Y, point_x, point_y):
    if slope == 0:
        return point_x
    return ((Y - point_y) + (slope * point_x))/slope

# Here we estimate speed based on the distance travelled relative to the vehicle length, by the centroid of the detected bounding box.
def EstimateSpeed_NoOverlap(vehicleDetection_curr, vehicleDetection_prev):
    # Find distance travelled in pixel coordinates
    x_pixels =  abs(vehicleDetection_prev.Centroid[0] - vehicleDetection_curr.Centroid[0])
    y_pixels =  abs(vehicleDetection_prev.Centroid[1] - vehicleDetection_curr.Centroid[1])
    Distance_pixels = math.sqrt(math.pow(x_pixels, 2) + math.pow(y_pixels, 2))

    # Slope = change in Y / change in X
    slope = (vehicleDetection_curr.Centroid[1] - vehicleDetection_prev.Centroid[1])/(vehicleDetection_curr.Centroid[0] - vehicleDetection_prev.Centroid[0])
    # Line -> y - y1 = slope(x - x1) : solve for any point we know
    # Get Y Coordinate where X is minimum and maximum
    bbtlwh = vehicleDetection_curr.BoundingBox_tlwh
    y_coord_min_x = GetYCoordinateAtX(slope, bbtlwh[0], vehicleDetection_curr.Centroid[0], vehicleDetection_curr.Centroid[1])
    y_coord_max_x = GetYCoordinateAtX(slope, bbtlwh[0] + bbtlwh[2], vehicleDetection_curr.Centroid[0], vehicleDetection_curr.Centroid[1])
    x_coord_min_y = GetXCoordinateAtY(slope, bbtlwh[1], vehicleDetection_curr.Centroid[0], vehicleDetection_curr.Centroid[1])
    x_coord_max_y = GetXCoordinateAtY(slope, bbtlwh[1] + bbtlwh[3], vehicleDetection_curr.Centroid[0], vehicleDetection_curr.Centroid[1])

    LineIntersectsBoundingbox = None
    # Centroid of previous frame has X coordinate less than current frame Centroid X coordinate -> Moved right
    if (vehicleDetection_prev.Centroid[0] <= vehicleDetection_curr.Centroid[0]):
        # Is Left edge?
        if (y_coord_min_x >= bbtlwh[1] and y_coord_min_x <= (bbtlwh[1]+bbtlwh[3])):
            LineIntersectsBoundingbox = "LEFT"
        if (LineIntersectsBoundingbox is None):
            # Is Top edge?
            if (vehicleDetection_prev.Centroid[1] <= vehicleDetection_curr.Centroid[1]):
                LineIntersectsBoundingbox = "TOP"
            # Is Bottom edge?
            else:
                LineIntersectsBoundingbox = "BOTTOM"
    else:
        # Is Right edge?
        if (y_coord_max_x >= bbtlwh[1] and y_coord_max_x <= (bbtlwh[1]+bbtlwh[3])):
            LineIntersectsBoundingbox = "RIGHT"
        if (LineIntersectsBoundingbox is None):
            # Is Top edge?
            if (vehicleDetection_prev.Centroid[1] <= vehicleDetection_curr.Centroid[1]):
                LineIntersectsBoundingbox = "TOP"
            # Is Bottom edge?
            else:
                LineIntersectsBoundingbox = "BOTTOM"

    intersection = (0, 0)
    if LineIntersectsBoundingbox == "LEFT":
        intersection = (bbtlwh[0], y_coord_min_x)
    elif LineIntersectsBoundingbox == "RIGHT":
        intersection = (bbtlwh[0]+bbtlwh[2], y_coord_max_x)
    elif LineIntersectsBoundingbox == "TOP":
        intersection = (x_coord_min_y, bbtlwh[1])
    elif LineIntersectsBoundingbox == "BOTTOM":
        intersection = (x_coord_max_y, bbtlwh[1]+bbtlwh[3])

    intersection_x, intersection_y = intersection

    # Get distance (x) and (y) between centroid of current frame and intersection between boundingbox of current frame and line
    # With this, we achieve the length of a vehicle in pixels 
    xP =  abs(intersection_x - vehicleDetection_curr.Centroid[0])
    yP =  abs(intersection_y - vehicleDetection_curr.Centroid[1])
    vehicle_length_px = math.sqrt(math.pow(xP, 2) + math.pow(yP, 2))

    # Map measurement (metres) to pixels -> since bounding box changes between one frame and another, we average both
    # Half length of vehicle equates to the length in pixels from centroid of BB to intersection point
    m_per_pixel = (vehicleDetection_curr.Length_m/2)/vehicle_length_px
    
    m_distanceTravelled = Distance_pixels * m_per_pixel
    
    # We know that 30 frames = 1 second (Dataset is generated this way)
    # speed = distance / time
    diffFrames = vehicleDetection_curr.FrameNumber - vehicleDetection_prev.FrameNumber
    speed_m_s = m_distanceTravelled/(diffFrames/fps)
    return speed_m_s


# Calculation is done per frame
def EstimateSpeed_CentroidPerFrame(current, previous):
    estimated_metres_per_second = 0

    # Find distance travelled in pixel coordinates
    x_pixels =  abs(previous.Centroid[0] - current.Centroid[0])
    y_pixels =  abs(previous.Centroid[1] - current.Centroid[1])
    distance_pixels = math.sqrt(math.pow(x_pixels, 2) + math.pow(y_pixels, 2))

    # Map Pixels to Metres
    metres_per_pixel = current.Length_m/current.BoundingBox_tlwh[2]

    distance_travelled_metres = distance_pixels * metres_per_pixel
    estimated_metres_per_second = distance_travelled_metres/(1/fps)
    return estimated_metres_per_second

# Exponential moving average = S(t) * (2/(n+1)) + ema(t-1) * (1-(2/(n+1))) 
# where 't' is current speed, 'n' is the number of previous speeds to consider
def ExponentialMovingAverage(lastSpeed, SpeedList, PrevEMA, n):
    ema = (lastSpeed * (2/(n+1))) + (PrevEMA * (1-(2/(n+1))))
    return ema
