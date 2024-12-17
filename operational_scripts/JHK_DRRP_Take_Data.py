import FliSdk_V2 as sdk  # As of now only Rayleigh has the First Light SDK installed. The camera is First Light CRED2 ER
from astropy.io import fits
import numpy as np
import time
from pylablib.devices import Thorlabs  # Library for controlling the Thorlabs rotation stages (holding the quarter-wave plates)
import copy
import os
from NKTcontrols.controls import compact, select, driver, get_status

# data taking script for DRRP, JHK plate
# save data to the desktop for now until exoserver is reconnected
# TODO: put additional information in fits header
# TODO: create folders automatically while saving data. Maybe only save the reduced data?
# TODO: loop script through multiple wavelengths automatically
# TODO: take images at multiple wavelengths before rotating
# TODO: try connecting rotation stages through serial numbers
# TODO: test if multiple stages can be moved simultaneously without interrupting

# Script that rotates both quarter-wave plates and takes images for the DRRP method. 
# Make sure the second motor rotates at 5x the rate of the first. I recommend rotating the first motor between 0-180 degrees in 4 degree increments.
# This script reduces the images taken by subtracting a dark frame from each. Make sure the corresponding dark frame exists before taking this data. 
# The images are stored as fits files with comments for the camera conditions in the file header. 
# Comments in all caps indicate parts that should be updated for each data set (wavelength, filename, folderpath)

compact1 = compact()
select1 = select()
driver1 = driver()

# Setup the laser
compact1.reset_interlock()
compact1.emission_on()

overall_power = input("Overall laser power to set as a percentage? ")
compact1.overall_power(overall_power) # set overall power as %
channel_power = input("Channel power to set as a percentage? ")
driver1.RF_power_on()
# Check if all other channels are off, then continue

wavelengths = [1400, 1500, 1600, 1700, 1800, 1900, 2000] # wavelengths in nm

# Setting camera context
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

# Now that the camera is set up, prepare for taking data
# Most secure way is to ensure connection with the motor through the Kinesis app before running code

stage1 = Thorlabs.KinesisMotor(Thorlabs.list_kinesis_devices()[0][0],scale='stage')
stage2 = Thorlabs.KinesisMotor(Thorlabs.list_kinesis_devices()[1][0], scale='stage')
print("Connected to K10CR1 devices")

# Query the user what angle range and what increments
tot_angle = input("Total angle to rotate (degrees)?")
increment = input("Increment angle to change?")

steps = int(tot_angle)/int(increment)

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


for i in range(int(steps)+1):
    # Move the quarter waveplates to the next position to take more images
    angle = i * int(increment)
    angle = int(angle)
    # stage1.move_by(int(increment))
    stage1.move_to(angle)
    stage1.wait_move()
    # stage2.move_by(5*int(increment))
    stage2.move_to(5*angle)
    stage2.wait_move()
    position1 = stage1.get_position()
    position2 = stage2.get_position()
    print('Current positions are ' + str(position1) + ' and ' + str(position2) + ' degrees')
    # sdk.Stop(context)


    # for i in range(int(steps)+1):
    for wavelength in wavelengths:
        driver1.set_channel(1, wavelength, channel_power) # set channel 1 to given wavelength at channel power

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
        # foldername = r"C:\\Users\\EPL User\\Desktop\\desktop_drrp_data\\calibration\\calibration_raw\\Cal_1750_Filter\\"
        # foldername = r"Z:\\Lab_Data\\Mueller_Matrix_Polarimeter\\L_Plate_Characterization\\SuperK_Select_Data\\Raw_Data\\L_1600_Raw\\"   # UPDATE FOR EACH FOLDER
        foldername_base = r"C:\\Users\\EPL User\\Desktop\\desktop_drrp_data\\calibration\\calibration_raw\\"   # Base folder
        foldername = os.path.join(foldername_base, f"Cal_{wavelength}_Raw\\")

        if not os.path.exists(foldername):
            os.makedirs(foldername) 
        
        for j in range(val+1):
            image16b = copy.deepcopy(sdk.GetRawImageAsNumpyArray(context, j))
            time.sleep(1.3*tint/1000)

            if j > 0:
                frame_list.append(image16b)
        
        frame_list = np.array(frame_list) 
        hdu_new = fits.PrimaryHDU(frame_list)
        # position1 = stage1.get_position()
        # position2 = stage2.get_position()
        # print('Position 1 is ' + str(position1) + ' and position 2 is ' + str(position2))
        filename = f"DRRP_L_{wavelength}nm_"+str(val_fps)+"_"+str(set_tint)+"_"+str(position1)          # UPDATE FOR EACH WAVELENGTH
        hdu_new.writeto(foldername+ filename+".fits", overwrite = True)

        # Add comments to the fits file header
        hdu = fits.open(foldername+filename+'.fits', mode='update')
        header = hdu[0].header
        header['COMMENT1'] = "Raw image taken using CRED2 ER performing DRRP measurements."
        header['COMMENT2'] = "QWP1 position: "+str(position1)+" degrees. Camera temperature: "+str(set_temp)+"C. Framerate: "+str(val_fps)+"fps. Exposure time: "+str(set_tint)+"ms. "
        hdu.flush()
        hdu.close()
        print("Files saved to " + str(foldername))


print("Raw images taken. Exiting SDK...")
sdk.Exit(context)


print("Reducing images...")
for wavelength in wavelengths:
    # UPDATE THESE PARTS FOR EACH WAVELENGTH
    # dark_file = 'Z:\\Lab_Data\\Mueller_Matrix_Polarimeter\\L_Plate_Characterization\\Darks\\Dark_600_0.1.fits'
    dark_file = r"C:\\Users\\EPL User\\Desktop\\darks\\Dark_1600_0.1.fits" # REPLACE WITH THE CORRECT DARK FILE
    image_file = 'DRRP_'
    # new_directory = r"Z:\\Lab_Data\\Mueller_Matrix_Polarimeter\\L_Plate_Characterization\\SuperK_Select_Data\\Reduced_Data\\Reduced_L_1600\\"
    new_directory_base = r"C:\\Users\\EPL User\\Desktop\\desktop_drrp_data\\calibration\\calibration_reduced\\"   # Base folder
    new_directory = os.path.join(foldername_base, f"Cal_{wavelength}_Reduced\\")

    if not os.path.exists(new_directory):
        os.makedirs(new_directory) 

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


# sdk.Exit(context)
