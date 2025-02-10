import FliSdk_V2 as sdk  # make sure scripts that use the First Light SDK are run on on the lab desktop, not through remote SSH
from astropy.io import fits
import numpy as np
import time
import copy

# This takes and saves images to the L-plate Dark image folder
# Take a dark image at the same camera conditions (fps, temp, tint) as the data, but with the camera lens cap on so no light gets in
# This file also adds comments to the fits file header

# Setting context
context = sdk.Init()

print("Detection of grabbers...")
listOfGrabbers = sdk.DetectGrabbers(context)

if len(listOfGrabbers) == 0:
    print("No grabber detected, exit.")
    exit()

print("Done.")
print("List of detected grabber(s):")

for s in listOfGrabbers:
    print("- " + s)

print("Detection of cameras...")
listOfCameras = sdk.DetectCameras(context)

if len(listOfCameras) == 0:
    print("No camera detected, exit.")
    exit()

print("Done.")

cameraIndex = 0
print("Setting camera: " + listOfCameras[cameraIndex])
ok = sdk.SetCamera(context, listOfCameras[cameraIndex])

if not ok:
    print("Error while setting camera.")
    exit()

print("Setting mode full.")

ok = sdk.Update(context)
print("Updating...")
if not ok:
    print("Error Updating")
    exit()

res, mb, fe, pw, init_sensor_temp, peltier, heatsink = sdk.FliCredTwo.GetAllTemp(context)
if res:
    print("Initial Temp: {:.2f}C".format(init_sensor_temp))
else:
    print("Error reading temperature.")

# Querying sensor temperature
try:
    set_temp = input("Temperature to set? (between " + str(-55) + " C and " + str(20)+ " C) ")
    set_temp = float(set_temp)
    ok = sdk.FliCredTwo.SetSensorTemp(context, float(set_temp))
    if not ok:
        print("Error while setting temperature.")
        exit()
except ValueError:
    print("Not a valid temperature")

ok = sdk.Update(context)
print("Starting to cool...")
if not ok:
    print("Error while updating.")
    exit()

res, mb, fe, pw, sensortemp, peltier, heatsink = sdk.FliCredTwo.GetAllTemp(context);

temp_tolerance = 0.3 #get close temp but don't print infinitely

while np.abs(sensortemp - set_temp) >= temp_tolerance:
    res, mb, fe, pw, sensortemp, peltier, heatsink = sdk.FliCredTwo.GetAllTemp(context)
    print("Sensor Temp: {:.2f}C".format(sensortemp),'\n','-------------')
    time.sleep(5)

res, mb, fe, pw, sensortemp, peltier, heatsink = sdk.FliCredTwo.GetAllTemp(context)
print("Finished Setting Temperature",'\n',"Final Temp: {:.2f}C".format(sensortemp))

# Control the fps
fps = 0

if sdk.IsSerialCamera(context):
    res, fps = sdk.FliSerialCamera.GetFps(context)
elif sdk.IsCblueSfnc(context):
    res, fps = sdk.FliCblueSfnc.GetAcquisitionFrameRate(context)
print("Current camera FPS: " + str(fps))


val_fps = input("FPS to set? ")
if val_fps.isnumeric():
    if sdk.IsSerialCamera(context):
        sdk.FliSerialCamera.SetFps(context, float(val_fps))
    elif sdk.IsCblueSfnc(context):
        sdk.FliCblueSfnc.SetAcquisitionFrameRate(context, float(val_fps))


if sdk.IsCredTwo(context) or sdk.IsCredThree(context):
    res, response = sdk.FliSerialCamera.SendCommand(context, "mintint raw")
    minTint = float(response)

    res, response = sdk.FliSerialCamera.SendCommand(context, "maxtint raw")
    maxTint = float(response)

    res, response = sdk.FliSerialCamera.SendCommand(context, "tint raw")

    print("Current camera tint: " + str(float(response)*1000) + "ms")

    set_tint = input("Tint to set? (between " + str(minTint*1000) + "ms and " + str(maxTint*1000)+ "ms) ")
    sdk.FliCredTwo.SetTint(context, float(float(set_tint)/1000))
    ok = sdk.Update(context)
    if not ok:
        print("error setting tint")
        exit()

    res, response = sdk.FliCredTwo.GetTint(context)
    tint = response*1000
    print("Current camera tint: " +str(tint) +"ms")


res = sdk.FliCredTwo.SetConversionGain(context,'low')
if not res:
    print('error setting gain mode')
sdk.Update(context)

val = input("Take how many images?")
val=int(val)
#if not val.isnumeric():
#    val = 10

print("Taking Images...")

sdk.EnableGrabN(context, val+1)
sdk.Update(context)

sdk.Start(context)
time.sleep(val*tint/1000)
counter = 0
max_iter = 10
while sdk.IsGrabNFinished(context) is False:
    if counter >= max_iter:
        break
    time.sleep(1)
    counter += 1

print("Is grab finished? " + str(sdk.IsGrabNFinished(context)))

frame_list = []
# Choose file-saving directory
# foldername = r"Z:\\Lab_Data\\Mueller_Matrix_Polarimeter\\L_Plate_Characterization\\Darks\\"
foldername = "D:\\desktop_drrp_data\\test_12_17_2024\\darks\\"
for i in range(val+1):
    image16b = copy.deepcopy(sdk.GetRawImageAsNumpyArray(context, i))

#     Only saves second frame onward. First image is a throwaway
    if i > 0:
        frame_list.append(image16b)

frame_list = np.array(frame_list)

# Save and name the files 
hdu_new = fits.PrimaryHDU(frame_list)
# filename = "Dark_"+str(val_fps)+"_"+str(set_tint)
filename = str(set_temp)+"C_tint_"+str(set_tint)+"_FPS_"+str(val_fps)
hdu_new.writeto(foldername+filename+".fits", overwrite = True)

hdu = fits.open(foldername+filename+'.fits', mode='update')
header = hdu[0].header
header['COMMENT1'] = "Dark image taken using CRED2 ER camera with cap on."
header['COMMENT2'] = "Camera temperature: "+str(set_temp)+"C. Framerate: "+str(val_fps)+"fps. Exposure time: "+str(set_tint)+"ms."
hdu.flush()
hdu.close()

print("Files saved to " + str(foldername))       
print("Is grab finished? " + str(sdk.IsGrabNFinished(context)))

sdk.Stop(context)


