import psychopy
import numpy as np
import pandas as pd
import os
import serial

from psychopy.hardware.pr import PR655
from psychopy import hardware

# import SpectraFunctions as SF
# port = '/dev/cu.usbmodem1423101'
# SF.openPR(port)
def openPR(port, action='open'):
    ser = serial.Serial(port)
    if action == 'open':
        ser.write(b'P')
        ser.write(b'H')
        ser.write(b'O')
        ser.write(b'T')
        ser.write(b'O')
        return
    else:
        ser.write(b'Q')
        return


def takeMeasurementsManual(port,saveP = '/Users/duffieldsj/Documents/GitHub/ColorCalibration/NIF'):
    myPR655 = PR655(port)
    nms = np.array([])
    powers = np.array([])
    XYZs = np.array([])
    xys = np.array([])
    measureNum = 0
    spectraDF = pd.DataFrame()
    XYZDF = pd.DataFrame()
    saveSpectraP = os.path.join(saveP,'spectra.csv')
    saveXYZP = os.path.join(saveP,'XYZ.csv')
    while True:
        karg = input('Enter to take measurement, press anything else to escape')
        if karg == "":
            myPR655.measure()  # make a measurement
            nm, power = myPR655.getLastSpectrum()
            XYZ = myPR655.getLastTristim()
            xy = myPR655.getLastXY()
            XYZ = [float(i) for i in XYZ[2:5]]
            xy = [float(i) for i in xy[2:5]]
            spectraDF_temp = pd.DataFrame(np.array([power]), columns=nm)
            spectraDF = spectraDF.append(spectraDF_temp,ignore_index=True)
            XYZDF_temp = pd.DataFrame(np.array([XYZ]), columns=['X','Y','Z'])
            XYZDF = XYZDF.append(XYZDF_temp,ignore_index=True)
            nms = np.append(nms,nm)
            powers = np.append(powers, power)
            XYZs = np.append(XYZs, XYZ)
            xys = np.append(xys, xy)

            print(XYZ)
            print(xy)
            print(measureNum)
            measureNum += 1
            spectraDF.to_csv(saveSpectraP)
            XYZDF.to_csv(saveXYZP)
        else:
            break

    return nms, powers, XYZs, xys, XYZDF, spectraDF

