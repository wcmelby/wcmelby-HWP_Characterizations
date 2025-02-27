import FliSdk_V2 as sdk
from astropy.io import fits
import numpy as np
import time
from pylablib.devices import Thorlabs
import copy
import os

# Double rotate and take image for DRRP method. Make sure the second motor rotates at 5x the rate of the first
# This data saves to the L_Plate_Characterization desktop folder
# This file also adds comments to the fits file header
# I recommend rotating in 4 degree increments to ensure that enough data is taken

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
val = int(val)

# Now that the camera is setup, prepare for rotating while taking picrues
# Most secure way is to ensure connection with the motor through the Kinesis app before running code

stage1 = Thorlabs.KinesisMotor(Thorlabs.list_kinesis_devices()[0][0],scale='stage')
stage2 = Thorlabs.KinesisMotor(Thorlabs.list_kinesis_devices()[1][0], scale='stage')
print("Connected to K10CR1 devices")

print("Homing devices...")
stage1.move_to(0)
stage1.wait_move()
stage1._setup_homing()
home1 = stage1.home(sync=True)

stage2.move_to(0)
stage2.wait_move()
stage2._setup_homing()
home2 = stage2.home(sync=True)
print('Homing complete')

position1 = stage1.get_position()
position2 = stage2.get_position()

print('Current positions are ' + str(position1) + ' and ' + str(position2) + ' degrees')

# Query the user what angle range and what increments
tot_angle = input("Total angle to rotate (degrees)?")
increment = input("Increment angle to change?")

steps = int(tot_angle)/int(increment)

print("Taking images...")

for i in range(int(steps)+1):
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
    # Now begin loop for the images
    #foldername = r"C:\\Users\\EPL User\\Desktop\\L_Plate_Characterization\\Calibration\\Calibration_Raw\\Cal_1750_Filter\\"
    foldername = r"Z:\\Lab_Data\\Mueller_Matrix_Polarimeter\\L_Plate_Characterization\\Raw_Data\\L_1550_Filter\\Stacked_1550_Filter\\"

    for j in range(val+1):
        image16b = copy.deepcopy(sdk.GetRawImageAsNumpyArray(context, j))
        time.sleep(1.3*tint/1000)

        if j > 0:
            frame_list.append(image16b)
    
    frame_list = np.array(frame_list) 
    hdu_new = fits.PrimaryHDU(frame_list)
    position1 = stage1.get_position()
    position2 = stage2.get_position()
    print('Position 1 is ' + str(position1) + ' and position 2 is ' + str(position2))
    filename = "DRRP_L_1550nm_stacked_"+str(val_fps)+"_"+str(set_tint)+"_"+str(position1)          # Remember to update the filename for each wavelenght
    hdu_new.writeto(foldername+ filename+".fits", overwrite = True)

# Add comments to the fits file header
    hdu = fits.open(foldername+filename+'.fits', mode='update')
    header = hdu[0].header
    header['COMMENT1'] = "Raw image taken using CRED2 ER performing DRRP measurements. Another wavelength filter at 1750 nm with 500 nm FWHM was placed just before the lens."
    header['COMMENT2'] = "QWP1 position: "+str(position1)+" degrees. Camera temperature: "+str(set_temp)+"C. Framerate: "+str(val_fps)+"fps. Exposure time: "+str(set_tint)+"ms. "
    hdu.flush()
    hdu.close()
    print("Files saved to " + str(foldername))

# Move the half waveplates to the next position to take more images
    stage1.move_by(int(increment))
    stage1.wait_move()
    stage2.move_by(5*int(increment))
    stage2.wait_move()
    sdk.Stop(context)


print("Raw images taken. Exiting SDK...")
sdk.Exit(context)


# try to subtract dark frame
reduce = input("Subtract dark frame from images? (1 for yes, 0 for no)")
reduce = float(reduce)
if reduce == 1:
    print("Subtracting dark frames")
else:
    quit()

print("Reducing images...")

# UPDATE THESE PARTS FOR EACH WAVELENGTH
dark_file = 'Z:\\Lab_Data\\Mueller_Matrix_Polarimeter\\L_Plate_Characterization\\Darks\\Dark_600_0.1.fits'
image_file = 'DRRP_'
new_directory = r"Z:\\Lab_Data\\Mueller_Matrix_Polarimeter\\L_Plate_Characterization\\Reduced_Data\\Reduced_L_1550\\Stacked_1550_Reduced\\"

# Open the dark image and extract pixel values
fits.open(dark_file)
dark = fits.getdata(dark_file)
dark_median = np.median(dark, axis=0)

# Search through the desired raw data folder
for file in os.listdir(foldername):
    if file.startswith(image_file):                                # Call specific files starting with the desired name
        with fits.open(os.path.join(foldername, file)) as hdul:
            img_data = hdul[0].data
            img_median = np.median(img_data, axis=0)
            reduced_data = img_median - dark_median

        # Save the newly reduced image to a reduced data folder
        new_filename = f"Reduced_{file}"
        new_filepath = os.path.join(new_directory, new_filename)
        fits.writeto(new_filepath, reduced_data, overwrite=True)

# Add comments to the newly reduced imagess
for file in os.listdir(new_directory):
    if file.startswith('Reduced_'):
        hdu = fits.open(new_directory+file, mode='update')
        header = hdu[0].header
        header['COMMENT1'] = "Reduced image taken using CRED2 ER performing DRRP measurements of the L-band plate."
        header['COMMENT2'] = "Camera temperature: "+str(set_temp)+"C. Framerate: "+str(val_fps)+"fps. Exposure time: "+str(set_tint)+"ms."
        hdu.flush()
        hdu.close()

print("Images have been reduced. Process finished.")

