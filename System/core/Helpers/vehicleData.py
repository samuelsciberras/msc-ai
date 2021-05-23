


# Constants - do not change (metres)
# This is the datastore
# Car
const_car_height = 1.9
const_car_length = 6.5
# SUV
const_suv_height = 2.5
const_suv_length = 6
# Bus
const_bus_height = 3.4
const_bus_length = 10


def GetCarData():
    return const_car_length, const_car_height

def GetSUVData():
    return const_suv_length, const_suv_height
    
def GetBusData():
    return const_bus_length, const_bus_height

def GetVehicleData(type):
    if (str.lower(type) == "car"):
        return GetCarData()
    elif (str.lower(type) == "suv"):
        return GetSUVData()
    elif (str.lower(type) == "bus"):
        return GetBusData()
    # No matches
    return None