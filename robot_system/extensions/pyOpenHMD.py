from robot_system.resource.definition import *
from openhmd import PyOpenHMD


def init_openhmd_driver():
    dev = PyOpenHMD()
    dev.printSensors()
    dev.printDeviceInfo()
