#Note that this is basically the same as DRRP_Functions.py in mypythonlibrary, but this version goes to the new folder

import numpy as np
from numpy.linalg import inv
from astropy.io import fits
import os
import re
from scipy.optimize import curve_fit

# These are the functions I will be using to run Dual Rotating Retarder Polarimeter measurements and analysis
#test commit
test = 2
# Mueller matrix for a linear polarizer, with angle a between transmission axis and horizontal (radians)
def linear_polarizer(a):
    M01 = np.cos(2*a)
    M02 = np.sin(2*a)
    M10 = np.cos(2*a)
    M11 = np.cos(2*a)**2
    M12 = np.cos(2*a)*np.sin(2*a)
    M20 = np.sin(2*a)
    M21 = np.cos(2*a)*np.sin(2*a)
    M22 = np.sin(2*a)**2

    return 0.5*np.array([[1, M01, M02, 0], 
                         [M10, M11, M12, 0], 
                         [M20, M21, M22, 0], 
                         [0, 0, 0, 0]])


# Mueller matrix for a linear retarder (waveplate). Angle of fast axis a, retardance r in radians
def linear_retarder(a, r):
    M11 = np.cos(2*a)**2 + np.cos(r)*np.sin(2*a)**2
    M12 = np.cos(2*a)*np.sin(2*a)*(1-np.cos(r))
    M13 = -np.sin(2*a)*np.sin(r)
    M21 = M12
    M22 = np.sin(2*a)**2 + np.cos(2*a)**2*np.cos(r)
    M23 = np.cos(2*a)*np.sin(r)
    M31 = -M13
    M32 = -M23
    M33 = np.cos(r)

    return np.array([[1, 0, 0, 0], 
                     [0, M11, M12, M13], 
                     [0, M21, M22, M23], 
                     [0, M31, M32, M33]])


# Sorting function for extracting filenames based on last number in the filename (the angle of rotation)
def extract_number(filename):
    match = re.findall(r'\d+(?:\.\d+)?', filename)
    if match:
        return float(match[-1])


    # Function to subtract dark frames from raw frames. The new reduced images are saved to a different folder
def dark_subtraction(image_file, dark_file, old_directory, new_directory):
    # Open the dark image and extract pixel values
    fits.open(dark_file)
    dark = fits.getdata(dark_file)
    dark_median = np.median(dark, axis=0)

    # Search through the desired raw data folder
    for filename in os.listdir(old_directory):
        if filename.startswith(image_file):                                # Call specific files starting with the desired name
            with fits.open(os.path.join(old_directory, filename)) as hdul:
                img_data = hdul[0].data
                img_median = np.median(img_data, axis=0)
                reduced_data = img_median - dark_median

            # Save the newly reduced image to a reduced data folder
            new_filename = f"Reduced_{filename}"
            new_filepath = os.path.join(new_directory, new_filename)
            fits.writeto(new_filepath, reduced_data, overwrite=True)


# Get intensity values from each spot in the reduced images. reduced_filename should just be the start of the name (leave out the last number, the angle). 
def extract_intensities(reduced_filename, reduced_folder, lcenter, rcenter, maxradius, cutoff=5000):
    I_left = np.array([])
    I_right = np.array([])
    bad_indices = np.array([])
    longtheta = np.linspace(0, np.pi, 46)

    for filename in sorted(os.listdir(reduced_folder), key = extract_number):
        if filename.startswith(reduced_filename):
            with fits.open(os.path.join(reduced_folder, filename)) as hdul:
                reduced_img_data = hdul[0].data
                ys, xs, = np.indices(reduced_img_data.shape)
                lradius = np.sqrt((ys-lcenter[0])**2+(xs-lcenter[1])**2)
                rradius = np.sqrt((ys-rcenter[0])**2+(xs-rcenter[1])**2)

                lbackground_mask = (lradius > 20) & (lradius < 26)
                rbackground_mask = (rradius > 20) & (rradius < 26)   # Index the background around each spot, take the median value

                background_lmedian = np.median(reduced_img_data[lbackground_mask])
                background_rmedian = np.median(reduced_img_data[rbackground_mask])

                lflux = np.sum(reduced_img_data[lradius < maxradius] - background_lmedian)   # Now take the flux with the background mask subtracted
                rflux = np.sum(reduced_img_data[rradius < maxradius] - background_rmedian)
                I_left = np.append(I_left, lflux)
                I_right = np.append(I_right, rflux)

                if lflux+rflux < cutoff:
                    print("Warning: low flux detected, check the image " + filename + ", index: " + str(sorted(os.listdir(reduced_folder), key = extract_number).index(filename)))
                    bad_indices = np.append(bad_indices, sorted(os.listdir(reduced_folder), key = extract_number).index(filename))
                else:
                    continue 

    # Makes the array a list of integers that can be used to index the other array
    bad_indices = bad_indices.astype(int)
    # Deletes the bad indices from the data
    I_left = np.delete(I_left, bad_indices)
    I_right = np.delete(I_right, bad_indices)
    new_angles = np.delete(longtheta, bad_indices)

    return I_left, I_right, new_angles, bad_indices


# Gives the condition number of eventual Mueller matrix (made by Jaren)
def condition_number(matrix):
    minv = np.linalg.pinv(matrix)

    # Compute maximum norm
    norm = np.linalg.norm(matrix, ord=np.inf)
    ninv = np.linalg.norm(minv, ord=np.inf)

    return norm*ninv


# Function to compute the Mueller matrix of a sample based on DRRP intensity measurements and calibration parameters
# def calibrated_full_mueller_polarimetry(thetas, a1, a2, w1, w2, r1, r2, I_meas=1, LPA_angle=0, return_condition_number=False, M_in=None):
#     nmeas = len(thetas)
#     Wmat = np.zeros([nmeas, 16])
#     Pmat = np.zeros([nmeas])
#     th = thetas

#     for i in range(nmeas):
#         # Mueller Matrix of generator (linear polarizer and a quarter wave plate)
#         Mg = linear_retarder(th[i]+w1, np.pi/2+r1) @ linear_polarizer(0+a1)

#         # Mueller Matrix of analyzer (one channel of the Wollaston prism is treated as a linear polarizer. The right spot is horizontal (0) and the left spot is vertical(pi/2))
#         Ma = linear_polarizer(LPA_angle+a2) @ linear_retarder(th[i]*5+w2, np.pi/2+r2)

#         # Data reduction matrix. Taking the 0 index ensures that intensity is the output
#         Wmat[i,:] = np.kron(Ma[0,:], Mg[:,0])

#         # M_in is some example Mueller matrix. Providing this input will test theoretical Mueller matrix. Otherwise, the raw data is used
#         if M_in is not None:
#             Pmat[i] = (Ma[0,:] @ M_in @ Mg[:,0]) * I_meas
#         else:
#             Pmat[i] = I_meas[i]

#     # Compute Mueller matrix using Moore-Penrose pseudo invervse
#     M = np.linalg.pinv(Wmat) @ Pmat
#     M = np.reshape(M,[4,4])

#     if return_condition_number == True:
#         return M, condition_number(Wmat)
#     else:
#         return M

def q_calibrated_full_mueller_polarimetry(thetas, a1, w1, w2, r1, r2, I_minus, I_plus, M_in=None):
    nmeas = len(thetas)  # Number of measurements
    Wmat1 = np.zeros([nmeas, 16])
    Pmat1 = np.zeros([nmeas])
    Wmat2 = np.zeros([nmeas, 16])
    Pmat2 = np.zeros([nmeas])
    th = thetas
    Q = I_plus - I_minus   # Difference in intensities measured by the detector. Plus should be the right spot, minus the left spot
    I_total = I_plus + I_minus

    for i in range(nmeas):
        # Mueller Matrix of generator (linear polarizer and a quarter wave plate)
        Mg = linear_retarder(th[i]+w1, np.pi/2+r1) @ linear_polarizer(0+a1)

        # Mueller Matrix of analyzer (one channel of the Wollaston prism is treated as a linear polarizer. The right spot is horizontal (0) and the left spot is vertical(pi/2))
        Ma = linear_retarder(th[i]*5+w2, np.pi/2+r2)

        # Data reduction matrix. Taking the 0 index ensures that intensity is the output
        Wmat1[i,:] = np.kron((Ma)[0,:], Mg[:,0]) # for the top row, using intensities
        Wmat2[i,:] = np.kron((Ma)[1,:], Mg[:,0]) # for the bottom 3 rows, using Q

        # M_in is some example Mueller matrix. Providing this input will test theoretical Mueller matrix. Otherwise, the raw data is used
        if M_in is not None:
            Pmat1[i] = (Ma[0,:] @ M_in @ Mg[:,0])
            Pmat2[i] = (Ma[1,:] @ M_in @ Mg[:,0])
        else:
            Pmat1[i] = I_total[i]  #Pmat is a vector of measurements (either I or Q)
            Pmat2[i] = Q[i] 

    # Compute Mueller matrix using Moore-Penrose pseudo invervse
    M1 = np.linalg.pinv(Wmat1) @ Pmat1
    M1 = np.reshape(M1, [4,4])

    M2 = np.linalg.pinv(Wmat2) @ Pmat2
    M2 = np.reshape(M2, [4,4])

    M = np.zeros([4,4])
    M[0,:] = M1[0,:]
    M[1:4,:] = M2[1:4,:]

    return M

# Define the identity matrix and other matrices which are useful for the Mueller calculus
M_identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
A = np.array([1, 0, 0, 0])
B = np.array([[1], [0], [0], [0]])
C = np.array([0, 1, 0, 0])


# # This is the full Mueller matrix equation for our setup. The output is a list, useful for curve fitting. Variables with 1 refer to the generator, 2 refers to analyzer. 
# def calibration_function(t, a1, a2, w1, w2, r1, r2):
#     prediction = [None]*len(t)
#     for i in range(len(t)):
#         prediction[i] = float(A @ linear_polarizer(a2) @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M_identity @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
#     return prediction


# # Calibration function designed for data from the left spot, which is the vertial alignment. This changes the angle of the analyzing LP
# def vertical_calibration_function(t, a1, a2, w1, w2, r1, r2):
#     prediction = [None]*len(t)
#     for i in range(len(t)):
#         prediction[i] = float(A @ linear_polarizer(a2+np.pi/2) @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M_identity @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
#     return prediction

def q_calibration_function(t, a1, w1, w2, r1, r2):
    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(C @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M_identity @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Basically the same as above, but with an optional input matrix to simulate data
def output_simulation_function(t, a1, a2, w1, w2, r1, r2, LPA_angle=0, M_in=None):
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(A @ linear_polarizer(LPA_angle+a2) @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Calculate the root-mean-square error of the calibration matrix by comparing wiht the identity matrix
def RMS_calculator(calibration_matrix):
    differences = []
    for i in range(0, 3):
        for j in range(0, 3):
            differences.append(calibration_matrix[i, j]-M_identity[i, j])

    differences_squared = [x**2 for x in differences]
    RMS = np.sqrt(sum(differences_squared))/16
    return RMS


# After testing, each of the above functions works individually. Now combine them into one function to rule them all
# Finds the mueller matrix derived from each channel separately, then averages the two retardances found this way
# First three inputs must come from the calibration data, last three inputs correspond to the HWP sample
# def ultimate_polarimetry(cal_angles, cal_left_intensity, cal_right_intensity, sample_angles, sample_left_intensity, sample_right_intensity):
#     initial_guess = [0, 0, 0, 0, 0, 0]
#     parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

#     # Find parameters from calibration of the left spot
#     lnormalized_intensity = cal_left_intensity/(2*max(cal_left_intensity))
#     lpopt, lpcov = curve_fit(vertical_calibration_function, cal_angles, lnormalized_intensity, p0=initial_guess, bounds=parameter_bounds)
#     print(lpopt, "Left parameters for a1, a2, w1, w2, r1, and r2. 1 for generator, 2 for analyzer")
#     #print(np.sqrt(np.diag(lpcov)))

#     # Find parameters from calibration of the right spot
#     rnormalized_intensity = cal_right_intensity/(2*max(cal_right_intensity))
#     rpopt, rpcov = curve_fit(calibration_function, cal_angles, rnormalized_intensity, p0=initial_guess, bounds=parameter_bounds)
#     print(rpopt, "Right parameters for a1, a2, w1, w2, r1, and r2. 1 for generator, 2 for analyzer")
#     #print(np.sqrt(np.diag(rpcov)))

#     # Optional print the calibration matrices (should be close to identity) to see how well the parameters compensate
#     MlCal = calibrated_full_mueller_polarimetry(cal_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5], cal_left_intensity, LPA_angle=np.pi/2)
#     print(MlCal/MlCal.max(), ' Left calibration')
#     MrCal = calibrated_full_mueller_polarimetry(cal_angles, rpopt[0], rpopt[1], rpopt[2], rpopt[3], rpopt[4], rpopt[5], cal_right_intensity)
#     print(MrCal/MrCal.max(), ' Right calibration')

#     # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample
#     Ml = calibrated_full_mueller_polarimetry(sample_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5], sample_left_intensity, LPA_angle=np.pi/2)
#     Ml = Ml/Ml.max()

#     Mr = calibrated_full_mueller_polarimetry(sample_angles, rpopt[0], rpopt[1], rpopt[2], rpopt[3], rpopt[4], rpopt[5], sample_right_intensity)
#     Mr = Mr/Mr.max()

#     np.set_printoptions(suppress=True)

#     # Extract retardance from the last entry of the mueller matrix, which should just be cos(phi)
#     lretardance = np.arccos(Ml[3,3])/(2*np.pi)
#     rretardance = np.arccos(Mr[3,3])/(2*np.pi)
#     print(lretardance, ' This is the retardance found from the left spot')
#     print(rretardance, ' This is the retardance found from the right spot')

#     avg_retardance = (lretardance+rretardance)/2

#     return Ml, Mr, avg_retardance

def q_ultimate_polarimetry(cal_angles, cal_left_intensity, cal_right_intensity, sample_angles, sample_left_intensity, sample_right_intensity):
    QCal = cal_right_intensity - cal_left_intensity # Plus should be the right spot, minus the left spot
    initial_guess = [0, 0, 0, 0, 0]
    parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

    # Find parameters from calibration of the left spot
    normalized_QCal = QCal/(max(QCal))
    popt, pcov = curve_fit(q_calibration_function, cal_angles, normalized_QCal, p0=initial_guess, bounds=parameter_bounds)
    print(popt, "Fit parameters for a1, w1, w2, r1, and r2. 1 for generator, 2 for analyzer")

    # The calibration matrix (should be close to identity) to see how well the parameters compensate
    MCal = q_calibrated_full_mueller_polarimetry(cal_angles, popt[0], popt[1], popt[2], popt[3], popt[4], cal_left_intensity, cal_right_intensity)
    MCal = MCal/np.max(MCal)
    RMS_error = RMS_calculator(MCal)
    #print(MCal, " This is the calibration Mueller Matrix.")

    # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample
    MSample = q_calibrated_full_mueller_polarimetry(sample_angles, popt[0], popt[1], popt[2], popt[3], popt[4], sample_left_intensity, sample_right_intensity)
    MSample = MSample/np.max(MSample)

    np.set_printoptions(suppress=True) # Suppresses scientific notation, keeps decimal format

    # Extract retardance from the last entry of the mueller matrix, which should just be cos(phi)
    retardance = np.arccos(MSample[3,3])/(2*np.pi)
    print(retardance, ' This is the retardance found from the data after calibration.')

    return MSample, retardance, MCal, RMS_error

