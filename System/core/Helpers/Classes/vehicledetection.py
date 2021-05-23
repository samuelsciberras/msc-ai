import core.Helpers.vehicleData as vehdetFuncts


class VehicleDetection(object):
    def __init__(self, id, label, boundingboxtlwh, boundingboxtlbr, frame_number):
        self.ClassLabel = label
        self.BoundingBox_tlwh = boundingboxtlwh
        self.BoundingBox_tlbr = boundingboxtlbr
        self.Id = id
        self.Centroid = self.SetCentroid()
        self.HistoricalCentroids = []
        self.HistoricalSpeed = []
        self.Length_m, self.Height_m = vehdetFuncts.GetVehicleData(self.ClassLabel)
        self.Speed = 0
        self.FrameNumber = frame_number
        self.EmaSpeed = 0

    def SetCentroid(self):
        ctr = []
        # x = (top left X coordinate) + (1/2)*Width
        ctr_x = self.BoundingBox_tlwh[0] + int(round(self.BoundingBox_tlwh[2] / 2))
        ctr.append(ctr_x)
        # Y = (top left Y coordinate) + (1/2)*Height
        ctr_y = self.BoundingBox_tlwh[1] + int(round(self.BoundingBox_tlwh[3] / 2))
        ctr.append(ctr_y)
        return ctr
