from Advanced_Function import *
from openhmd import PyOpenHMD


def init_openhmd_driver():
    dev = PyOpenHMD()
    dev.printSensors()
    dev.printDeviceInfo()
