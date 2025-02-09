# Functions used to run Dual Rotating Retarder Polarimeter (DRRP) measurements and analysis
# This is the new version of DRRP_Functions.py that uses Q measurements instead of I, and some additional features

import numpy as np
from numpy.linalg import inv
from astropy.io import fits
import os
import re
from scipy.optimize import curve_fit
# import csv
# import random
# import matplotlib.pyplot as plt
# import textwrap

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
    """reduced_filename: a string indicating the first part of the file name that is the same for each file, like 'Reduced_DRRP_'.
    reduced_folder: a string indicating the folder where these files are located. 
    lcenter and rcenter: the coordinates [y, x] for the location of each beam on the detector. 
    maxradius: an integer number of pixels to define as the radius of the beam.
    Cutof: the value of the sum of pixels in the spot. 
    If the measured value is less than this cutoff threshold, there is likely an error with that image and it will raise a warning.
    
    Outputs:
    I_left and I_right: pixel sum of each spot, the intensity.
    new_angles: the QWP angles with usable data. Angles where I<cutoff are excluded from this list.
    bad_indices: list of the bad angles that were excluded. """

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

                lbackground_mask = (lradius > maxradius+5) & (lradius < maxradius+10)
                rbackground_mask = (rradius > maxradius+5) & (rradius < maxradius+10)   # Index the background around each spot, take the median value

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
    # Deletes the bad indices (caused by camera glitch or some other complication) from the data
    I_left = np.delete(I_left, bad_indices)
    I_right = np.delete(I_right, bad_indices)
    new_angles = np.delete(longtheta, bad_indices)

    return I_left, I_right, new_angles, bad_indices


# Gives the condition number of eventual Mueller matrix 
def condition_number(matrix):
    minv = np.linalg.pinv(matrix)

    # Compute maximum norm
    norm = np.linalg.norm(matrix, ord=np.inf)
    ninv = np.linalg.norm(minv, ord=np.inf)

    return norm*ninv


# Function that makes the Mueller matrix using the calibration parameters a1, w1, w2, r2, and r2. Set these to 0 for an uncalibrated matrix
# def q_calibrated_full_mueller_polarimetry(thetas, a1, w1, w2, r1, r2, I_minus, I_plus, M_in=None):
#     nmeas = len(thetas)  # Number of measurements
#     Wmat1 = np.zeros([nmeas, 16])
#     Pmat1 = np.zeros([nmeas])
#     Wmat2 = np.zeros([nmeas, 16])
#     Pmat2 = np.zeros([nmeas])
#     th = thetas
#     unnormalized_Q = I_plus - I_minus   # Difference in intensities measured by the detector. Plus should be the right spot, minus the left spot
#     unnormalized_I_total = I_plus + I_minus
#     Q = unnormalized_Q/np.max(unnormalized_I_total)
#     I_total = unnormalized_I_total/np.max(unnormalized_I_total)

#     # Both Q and I should be normalized by the total INPUT flux, but we don't know this value. The closest we can guess is the maximum of the measured intensity
#     # This assumes the input flux is constant over time. Could be improved with a beam splitter that lets us monitor the input flux over time

#     for i in range(nmeas):
#         # Mueller Matrix of generator (linear polarizer and a quarter wave plate)
#         Mg = linear_retarder(th[i]+w1, np.pi/2+r1) @ linear_polarizer(0+a1)

#         # Mueller Matrix of analyzer (one channel of the Wollaston prism is treated as a linear polarizer. The right spot is horizontal (0) and the left spot is vertical(pi/2))
#         Ma = linear_retarder(th[i]*5+w2, np.pi/2+r2)

#         # Data reduction matrix. Taking the 0 index ensures that intensity is the output
#         Wmat1[i,:] = np.kron((Ma)[0,:], Mg[:,0]) # for the top row, using intensities
#         Wmat2[i,:] = np.kron((Ma)[1,:], Mg[:,0]) # for the bottom 3 rows, using Q

#         # M_in is some example Mueller matrix. Providing this input will test theoretical Mueller matrix. Otherwise, the raw data is used
#         if M_in is not None:
#             Pmat1[i] = (Ma[0,:] @ M_in @ Mg[:,0])
#             Pmat2[i] = (Ma[1,:] @ M_in @ Mg[:,0])
#         else:
#             Pmat1[i] = I_total[i]  #Pmat is a vector of measurements (either I or Q)
#             Pmat2[i] = Q[i] 

#     # Compute Mueller matrix using Moore-Penrose pseudo invervse
#     M1 = np.linalg.pinv(Wmat1) @ Pmat1
#     M1 = np.reshape(M1, [4,4])

#     M2 = np.linalg.pinv(Wmat2) @ Pmat2
#     M2 = np.reshape(M2, [4,4])

#     M = np.zeros([4,4])
#     M[0,:] = M1[0,:]
#     M[1:4,:] = M2[1:4,:]

#     return M

def q_calibrated_full_mueller_polarimetry(thetas, a1, w1, w2, r1, r2, I_vert, I_hor, M_in=None):
    """Full Mueller polarimetry using measurements of Q and calibration parameters. 
    Gives a calibrated Mueller matrix with the parameters, or set a1, w1, w2, r1, and r2 to zero for an uncalibrated matrix.
    Parameters
    ----------
    thetas : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    I_hor : array
        measured intensity of the horizontal polarization spot from the Wollaston prism
    I_vert : array
        measured intensity of the vertical polarization spot from the Wollaston prism
    Returns
    -------
    M : array
        4x4 Mueller matrix for the measured sample. """
    
    nmeas = len(thetas)  # Number of measurements
    Wmat1 = np.zeros([nmeas, 16])
    Pmat1 = np.zeros([nmeas])
    Wmat2 = np.zeros([nmeas, 16])
    Pmat2 = np.zeros([nmeas])
    th = thetas
    unnormalized_Q = I_hor - I_vert   # Difference in intensities measured by the detector
    unnormalized_I_total = I_vert + I_hor
    Q = unnormalized_Q/np.max(unnormalized_I_total)
    I_total = unnormalized_I_total/np.max(unnormalized_I_total)
    # Both Q and I should be normalized by the total INPUT flux, but we don't know this value. The closest we can guess is the maximum of the measured intensity
    # This assumes the input flux is constant over time. Could be improved with a beam splitter that lets us monitor the input flux over time

    for i in range(nmeas):
        # Mueller Matrix of generator (linear polarizer and a quarter wave plate)
        Mg = linear_retarder(th[i]+w1, np.pi/2+r1) @ linear_polarizer(0+a1)

        # Mueller Matrix of analyzer (one channel of the Wollaston prism is treated as a linear polarizer)
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


# In order, the calibration parameters are LP1 angle, QWP1 axis angle, QWP2 axis angle, QWP1 retardance, QWP2 retrdance
def q_calibration_function(t, a1, w1, w2, r1, r2):
    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(C @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M_identity @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Basically the same as above, but with an optional input matrix to simulate data
# def q_output_simulation_function(t, a1, w1, w2, r1, r2, M_in=None):
#     if M_in is None:
#         M = M_identity
#     else:
#         M = M_in

#     prediction = [None]*len(t)
#     for i in range(len(t)):
#         prediction[i] = float(C @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
#     return prediction


# Function that is useful for generating intensity values for a given sample matrix and offset parameters
# def I_output_simulation_function(t, a1, w1, w2, r1, r2, M_in=None):
#     if M_in is None:
#         M = M_identity
#     else:
#         M = M_in

#     prediction = [None]*len(t)
#     for i in range(len(t)):
#         prediction[i] = float(A  @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
#     return prediction


# Calculate the root-mean-square error of the calibration matrix by comparing with the identity matrix
def RMS_calculator(calibration_matrix):
    differences = []
    for i in range(0, 4):
        for j in range(0, 4):
            differences.append(calibration_matrix[i, j]-M_identity[i, j])

    differences_squared = [x**2 for x in differences]
    RMS = np.sqrt(sum(differences_squared)/16)
    return RMS


# Calculate the retardance error by standard error propogation using RMS in the matrix elements from calibration
def propagated_error(M_R, RMS):
    # return RMS/np.sqrt(1-(np.trace(M_R)/2-1)**2) # These two equations are equivalent
    x = np.trace(M_R)
    return 2*RMS/np.sqrt(4*x-x**2) # Value in radians


# The function that gives everything you want to know at once
# def q_ultimate_polarimetry(cal_angles, cal_left_intensity, cal_right_intensity, sample_angles, sample_left_intensity, sample_right_intensity):
#     """
#     Function that does polarimetric analysis. 
#     Calibration data is a measurement of air, sample data is a measurement of an optical sample. 
#     Inputs:
#     cal_angles: list of angles used in calibration.
#     cal_left_intensity and cal_right_intensity: list of intensity values for each spot at the given calibration angles. 
#     sample_angles: list of angles used for the sample.
#     sample_left_intensity and sample_right_intensity: list of intensity values for each spot at the given sample angles. 
    
#     Outputs:
#     MSample: 4x4 Mueller matrix of the sample.
#     retardance: retardance of the sample in waves.
#     MCal: measured calibration matrix (should resemble a 4x4 identity matrix).
#     RMS_Error: root-mean-squared error of the calibration matrix.
#     Retardance_Error: error propagated to the retardance value.
#     """
    
#     ICal = cal_right_intensity + cal_left_intensity  # Plus should be the right spot, minus is the left spot
#     QCal = cal_right_intensity - cal_left_intensity 
#     initial_guess = [0, 0, 0, 0, 0]
#     parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

#     # Find parameters from calibration 
#     normalized_QCal = QCal/(max(ICal)) # This should be normalized by the input intensity, but we don't know that so use the max of the measured intensity instead as an approximation
#     popt, pcov = curve_fit(q_calibration_function, cal_angles, normalized_QCal, p0=initial_guess, bounds=parameter_bounds)
#     # print(popt, "Fit parameters for a1, w1, w2, r1, and r2. 1 for generator, 2 for analyzer")

#     # The calibration matrix (should be close to identity) to see how well the parameters compensate
#     MCal = q_calibrated_full_mueller_polarimetry(cal_angles, popt[0], popt[1], popt[2], popt[3], popt[4], cal_left_intensity, cal_right_intensity)
#     MCal = MCal/np.max(np.abs(MCal))
#     RMS_Error = RMS_calculator(MCal)
#     #print(MCal, " This is the calibration Mueller Matrix.")

#     # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample
#     MSample = q_calibrated_full_mueller_polarimetry(sample_angles, popt[0], popt[1], popt[2], popt[3], popt[4], sample_left_intensity, sample_right_intensity)
#     MSample = MSample/np.max(np.abs(MSample))

#     np.set_printoptions(suppress=True) # Suppresses scientific notation, keeps decimal format

#     # Use the polar decomposition of the retarder matrix (see below)
#     r_decomposed_MSample = decompose_retarder(MSample)     
#     retardance = np.arccos(np.trace(decompose_retarder(r_decomposed_MSample))/2 - 1)/(2*np.pi)

#     Retardance_Error = propagated_error(r_decomposed_MSample, RMS_Error)
    
#     return MSample, retardance, MCal, RMS_Error, Retardance_Error, popt



# Functions from Jaren's katsu code on Polar decomposition. Inspired by Lu & Chipman 1996 https://doi.org/10.1364/JOSAA.13.001106

def broadcast_outer(a,b):
    """broadcasted outer product of two A,B,...,N vectors. Used for polarimetric data reduction
    where out is a A,B,...,N,N matrix. While in principle this does not require vectors of different length, it is not tested
    to produce anything other than square matrices.
    Parameters
    ----------
    a : numpy.ndarray
        A,B,...,N vector 1
    b : numpy.ndarray
        A,B,...,N vector 2
    Returns
    -------
    numpy.ndarray
        outer product matrix
    """
    return np.einsum('...i,...j->...ij',a,b)

# def _empty_mueller(shape):
#     """Returns an empty array to populate with Mueller matrix elements.
#     Parameters
#     ----------
#     shape : list
#         shape to prepend to the mueller matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
#         where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.
#     Returns
#     -------
#     numpy.ndarray
#         The zero array of specified shape
#     Notes
#     -----
#     The structure of this function was taken from prysm.x.polarization, which was written by Jaren Ashcraft
#     """
#     if shape is None:
#         shape = (4, 4)
#     else:
#         shape = (*shape, 4, 4)
#     return np.zeros(shape)

def _empty_mueller(shape):
    """Returns an empty array to populate with Mueller matrix elements.

    Parameters
    ----------
    shape : list
        shape to prepend to the mueller matrix array. shape = [32,32] returns
        an array of shape [32,32,4,4] where the matrix is assumed to be in the
        last indices. Defaults to None, which returns a 4x4 array.

    Returns
    -------
    numpy.ndarray
        The zero array of specified shape

    Notes
    -----
    The structure of this function was taken from prysm.x.polarization,
    which was written by Jaren Ashcraft
    """

    if shape is None:
        shape = (4, 4)

    else:
        shape = (*shape, 4, 4)

    return np.zeros(shape)


# def decompose_diattenuator(M):
#     """Decompose M into a diattenuator using the Polar decomposition from Lu & Chipman 1996 https://doi.org/10.1364/JOSAA.13.001106
#     Parameters
#     ----------
#     M : numpy.ndarray
#         Mueller Matrix to decompose
#     Returns
#     -------
#     numpy.ndarray
#         Diattenuator component of mueller matrix
#     """
#     # First, determine the diattenuator
#     T = M[..., 0, 0]
#     if M.ndim > 2:
#         diattenuation_vector = M[..., 0, 1:] / T[..., np.newaxis]
#     else:
#         diattenuation_vector = M[..., 0, 1:] / T

#     D = np.sqrt(np.sum(diattenuation_vector * diattenuation_vector, axis=-1))
#     mD = np.sqrt(1 - D**2)

#     if M.ndim > 2:
#         diattenutation_norm = diattenuation_vector / D[..., np.newaxis]
#     else:
#         diattenutation_norm = diattenuation_vector / D

#     # DD = diattenutation_norm @ np.swapaxes(diattenutation_norm,-2,-1)
#     DD = broadcast_outer(diattenutation_norm, diattenutation_norm)

#     # create diattenuator
#     I = np.identity(3)

#     if M.ndim > 2:
#         I = np.broadcast_to(I, [*M.shape[:-2], 3, 3])
#         mD = mD[..., np.newaxis, np.newaxis]

#     inner_diattenuator = mD * I + (1 - mD) * DD # Eq. 19 Lu & Chipman

#     Md = _empty_mueller(M.shape[:-2])

#     # Eq 18 Lu & Chipman
#     Md[..., 0, 0] = 1.
#     Md[..., 0, 1:] = diattenuation_vector
#     Md[..., 1:, 0] = diattenuation_vector
#     Md[..., 1:, 1:] = inner_diattenuator

#     if M.ndim > 2:
#         Md = Md * T[..., np.newaxis, np.newaxis]
#     else:
#         Md = Md * T
    
#     Md = Md/np.max(np.abs(Md))   # remember to normalize the matrix

#     return Md

def decompose_diattenuator(M, normalize=False):
    """Decompose M into a diattenuator using the Polar decomposition

    from Lu & Chipman 1996 https://doi.org/10.1364/JOSAA.13.001106

    Parameters
    ----------
    M : numpy.ndarray
        Mueller Matrix to decompose

    Returns
    -------
    numpy.ndarray
        Diattenuator component of mueller matrix
    """

    # First, determine the diattenuator
    T = M[..., 0, 0]

    if np.__name__ == "jax.numpy":
        if M.ndim > 2:
            diattenuation_vector = M[..., 0, 1:] / (T.at[..., np.newaxis])
        else:
            diattenuation_vector = M[..., 0, 1:] / (T)

        # D = np.sqrt(np.sum(np.matmul(diattenuation_vector, diattenuation_vector), axis=-1))
        D = np.sqrt(np.sum(diattenuation_vector * diattenuation_vector, axis=-1))
        mD = np.sqrt(1 - D ** 2)

        if M.ndim > 2:
            diattenutation_norm = diattenuation_vector / (D.at[..., np.newaxis])
        else:
            diattenutation_norm = diattenuation_vector / (D)

        DD = broadcast_outer(diattenutation_norm, diattenutation_norm)

        # create diattenuator
        I = np.identity(3)

        if M.ndim > 2:
            I = np.broadcast_to(I, [*M.shape[:-2], 3, 3])
            mD = mD[..., np.newaxis, np.newaxis] 

        inner_diattenuator = mD * I + (1 - mD) * DD  # Eq. 19 Lu & Chipman

        Md = _empty_mueller(M.shape[:-2])

        # Eq 18 Lu & Chipman
        Md = Md.at[..., 0, 0].set(1.)
        Md = Md.at[..., 0, 1:].set(diattenuation_vector)
        Md = Md.at[..., 1:, 0].set(diattenuation_vector)
        Md = Md.at[..., 1:, 1:].set(inner_diattenuator)

        if M.ndim > 2:
            Md = Md * T[..., np.newaxis, np.newaxis]
        else:
            Md = Md * T

        if normalize:
            return Md/np.max(np.abs(Md))
        else:
            return Md

    else:
        if M.ndim > 2:
            diattenuation_vector = M[..., 0, 1:] / T[..., np.newaxis]
        else:
            diattenuation_vector = M[..., 0, 1:] / T
        
        D = np.sqrt(np.sum(diattenuation_vector * diattenuation_vector, axis=-1))
        mD = np.sqrt(1 - D**2)

        if M.ndim > 2:
            diattenutation_norm = diattenuation_vector / D[..., np.newaxis]
        else:
            diattenutation_norm = diattenuation_vector / D

        DD = broadcast_outer(diattenutation_norm, diattenutation_norm)

        # create diattenuator
        I = np.identity(3)

        if M.ndim > 2:
            I = np.broadcast_to(I, [*M.shape[:-2], 3, 3])
            mD = mD[..., np.newaxis, np.newaxis]

        inner_diattenuator = mD * I + (1 - mD) * DD  # Eq. 19 Lu & Chipman

        Md = _empty_mueller(M.shape[:-2])

        # Eq 18 Lu & Chipman
        Md[..., 0, 0] = 1.
        Md[..., 0, 1:] = diattenuation_vector
        Md[..., 1:, 0] = diattenuation_vector
        Md[..., 1:, 1:] = inner_diattenuator

        if M.ndim > 2:
            Md = Md * T[..., np.newaxis, np.newaxis]
        else:
            Md = Md * T

        if normalize:
            return Md/np.max(np.abs(Md))
        else:
            return Md


# def decompose_retarder(M, return_all=False):
#     """Decompose M into a retarder using the Polar decomposition from Lu & Chipman 1996 https://doi.org/10.1364/JOSAA.13.001106
#     Note: this doesn't work if the diattenuation can be described by a pure polarizer,
#     because the matrix is singular and therefore non-invertible
#     Parameters
#     ----------
#     M : numpy.ndarray
#         Mueller Matrix to decompose
#     return_all : bool
#         Whether to return the retarder and diattenuator vs just the retarder.
#         Defaults to False, which returns both
#     Returns
#     -------
#     numpy.ndarray
#         Retarder component of mueller matrix
#     """
#     Md = decompose_diattenuator(M)
    
#     # Then, derive the retarder
#     Mr = M @ np.linalg.inv(Md)
#     Mr = Mr/np.max(np.abs(Mr))   # remember to normalize the matrix

#     if return_all:
#         return Mr, Md 
#     else:
#         return Mr

def decompose_retarder(M, return_all=False, normalize=False):
    """Decompose M into a retarder using the Polar decomposition

    from Lu & Chipman 1996 https://doi.org/10.1364/JOSAA.13.001106

    Note: this doesn't work if the diattenuation can be described by a pure
    polarizer, because the matrix is singular and therefore non-invertible

    Parameters
    ----------
    M : numpy.ndarray
        Mueller Matrix to decompose
    return_all : bool
        Whether to return the retarder and diattenuator vs just the retarder.
        Defaults to False, which returns both

    Returns 
    -------
    numpy.ndarray
        Retarder component of mueller matrix
    """

    if normalize:
        Md = decompose_diattenuator(M, normalize=True)
    else:
        Md = decompose_diattenuator(M)

    # Then, derive the retarder
    Mr = M @ np.linalg.inv(Md)

    if normalize:
        Mr = Mr/np.max(np.abs(Mr)) 
    else:
        Mr = Mr

    if return_all:
        return Mr, Md
    else:
        return Mr
    

def q_output_simulation_function(t, a1, w1, w2, r1, r2, M_in=None):
    """Function that models the Mueller calculus for the DRRP system and is used to calculate the calibration parameters.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values."""
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(C @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Function that is useful for generating intensity values for a given sample matrix and offset parameters
def I_total_output_simulation_function(t, a1, w1, w2, r1, r2, M_in=None):
    """Function to generate TOTAL intensity values measured with a given Mueller matrix and offset parameters.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values."""
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(A  @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Basically the same as above, but with an optional input matrix to simulate data
def single_output_simulation_function(t, a1, a2, w1, w2, r1, r2, LPA_angle=0, M_in=None):
    """
    Function to generate intensity values for one polarization at a time. Default is horizontal, with LPA=0. For vertical, set LPA=pi/2.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    a2 : float
        calibration parameter for the offset angle of the second linear polarizer (could be just one channel of the Wollaston prism)
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    LPA_angle : float
        angle of the analyzing linear polarizer. Default is 0 for horizontal. Set to pi/2 for vertical.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values.    """
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(A @ linear_polarizer(LPA_angle+a2) @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


def q_ultimate_polarimetry(cal_angles, cal_vert_intensity, cal_hor_intensity, sample_angles, sample_vert_intensity, sample_hor_intensity):
    """
    Function that calculates the Mueller matrix of a sample and other relevant information.
    cal_angles and sample_angles could be the same, or could be different.
    Parameters
    ----------
    cal_angles : array
        angles of the first quarter wave plate for calibration
    cal_vert_intensity : array
        measured intensity of the vertical polarization spot from the Wollaston prism for calibration
    cal_hor_intensity : array
        measured intensity of the horizontal polarization spot from the Wollaston prism for calibration
    sample_angles : array
        angles of the first quarter wave plate when taking data with the sample
    sample_vert_intensity : array
        measured intensity of the vertical polarization spot from the Wollaston prism when taking data with the sample
    sample_hor_intensity : array
        measured intensity of the horizontal polarization spot from the Wollaston prism when taking data with the sample
    Returns
    -------
    M_Sample : array
        4x4 Mueller matrix for the sample
    retardance : float
        extracted retardance of the sample in waves
    M_Cal : array
        4x4 Mueller matrix for the calibration (should resemble the identity matrix)
    RMS_Error : float
        root mean square error of the calibration matrix
    Retardance_Error : float
        error of the retardance value, assuming the RMS error from the calibration matrix is the same for all elements of the sample matrix.
    """
    ICal = cal_hor_intensity + cal_vert_intensity
    QCal = cal_hor_intensity - cal_vert_intensity 
    initial_guess = [0, 0, 0, 0, 0]
    parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

    # Find parameters from calibration 
    normalized_QCal = QCal/(max(ICal)) # This should be normalized by the input intensity, but we don't know that so use the max of the measured intensity instead as an approximation
    # popt, pcov = curve_fit(q_calibration_function, cal_angles, normalized_QCal, p0=initial_guess, bounds=parameter_bounds)
    popt, pcov = curve_fit(q_output_simulation_function, cal_angles, normalized_QCal, p0=initial_guess, bounds=parameter_bounds)
    # print(popt, "Fit parameters for a1, w1, w2, r1, and r2. 1 for generator, 2 for analyzer")

    # The calibration matrix (should be close to identity) to see how well the parameters compensate
    MCal = q_calibrated_full_mueller_polarimetry(cal_angles, popt[0], popt[1], popt[2], popt[3], popt[4], cal_vert_intensity, cal_hor_intensity)
    MCal = MCal/np.max(np.abs(MCal))
    RMS_Error = RMS_calculator(MCal)

    # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample
    MSample = q_calibrated_full_mueller_polarimetry(sample_angles, popt[0], popt[1], popt[2], popt[3], popt[4], sample_vert_intensity, sample_hor_intensity)
    MSample = MSample/np.max(np.abs(MSample))

    np.set_printoptions(suppress=True) # Suppresses scientific notation, keeps decimal format

    # Use the polar decomposition of the retarder matrix 
    r_decomposed_MSample = decompose_retarder(MSample, normalize=True)
    # retardance = np.arccos(np.trace(normalized_decompose_retarder(r_decomposed_MSample))/2 - 1)/(2*np.pi) # Value in waves
    retardance = np.arccos(np.trace(r_decomposed_MSample)/2 - 1)/(2*np.pi) # Value in waves

    Retardance_Error = propagated_error(r_decomposed_MSample, RMS_Error)
    
    return MSample, retardance, MCal, RMS_Error, Retardance_Error, popt



# imports from Old DRRP_Functions, specializing in I reduction (and averaging over the two beams)

def I_calibrated_full_mueller_polarimetry(thetas, a1, a2, w1, w2, r1, r2, I_meas=1, LPA_angle=0, return_condition_number=False, M_in=None):
    """
    Function that takes measurements and calibration parameters to construct a Mueller matrix using I reduction.
    """
    
    nmeas = len(thetas)
    Wmat = np.zeros([nmeas, 16])
    Pmat = np.zeros([nmeas])
    th = thetas

    for i in range(nmeas):
        # Mueller Matrix of generator (linear polarizer and a quarter wave plate)
        Mg = linear_retarder(th[i]+w1, np.pi/2+r1) @ linear_polarizer(0+a1)

        # Mueller Matrix of analyzer (one channel of the Wollaston prism is treated as a linear polarizer. The right spot is horizontal (0) and the left spot is vertical(pi/2))
        Ma = linear_polarizer(LPA_angle+a2) @ linear_retarder(th[i]*5+w2, np.pi/2+r2)

        # Data reduction matrix. Taking the 0 index ensures that intensity is the output
        Wmat[i,:] = np.kron(Ma[0,:], Mg[:,0])

        # M_in is some example Mueller matrix. Providing this input will test theoretical Mueller matrix. Otherwise, the raw data is used
        if M_in is not None:
            Pmat[i] = (Ma[0,:] @ M_in @ Mg[:,0]) * I_meas
        else:
            Pmat[i] = I_meas[i]

    # Compute Mueller matrix using Moore-Penrose pseudo invervse
    M = np.linalg.pinv(Wmat) @ Pmat
    M = np.reshape(M,[4,4])

    if return_condition_number == True:
        return M, condition_number(Wmat)
    else:
        return M
    

# def I_horizontal_calibration_function(t, a1, a2, w1, w2, r1, r2):
#     """
#     Function to simulate the intensity output of the horizontal channel. Used for curve fitting the calibration parameters.
#     """

#     prediction = [None]*len(t)
#     for i in range(len(t)):
#         prediction[i] = float(A @ linear_polarizer(a2) @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M_identity @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
#     return prediction


# def I_vertical_calibration_function(t, a1, a2, w1, w2, r1, r2):
#     """
#     Function to simulate the intensity output of the vertical channel. Used for curve fitting the calibration parameters.
#     """

#     prediction = [None]*len(t)
#     for i in range(len(t)):
#         prediction[i] = float(A @ linear_polarizer(a2+np.pi/2) @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M_identity @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
#     return prediction

def I_vertical_calibration_function(t, a1, a2, w1, w2, r1, r2):
    """
    Simulate the horizontal channel intensity.
    
    Parameters:
      xdata : tuple
          A tuple containing (t, I_meas), where
              t      : array of angles
              I_total_meas : array of measured intensities, summing both vertical and horizontal channels. Should have the same length as t.
      a1, a2, w1, w2, r1, r2 : float
          Calibration parameters.
    
    Returns:
      Array of predicted output intensities. Values range from 0 to 0.5 (at most 1/2 the input intensity, because the first element is a linear polarizer)
    """

    # Preallocate an array for predictions.
    prediction = np.empty_like(t, dtype=float)
    
    for i in range(len(t)):
        # Assume linear_polarizer and linear_retarder are defined elsewhere.
        prediction[i] = float(
            A @ linear_polarizer(a2+np.pi/2) @ linear_retarder(5 * t[i] + w2, np.pi/2 + r2) @ 
            M_identity @ linear_retarder(t[i] + w1, np.pi/2 + r1) @ linear_polarizer(a1) @ B 
        )
    
    # Multiply the simulated intensity with the measured input intensity at each angle.
    return prediction


def I_horizontal_calibration_function(t, a1, a2, w1, w2, r1, r2):
    """
    Simulate the horizontal channel intensity.
    
    Parameters:
      xdata : tuple
          A tuple containing (t, I_meas), where
              t      : array of angles
              I_total_meas : array of measured intensities, summing both vertical and horizontal channels. Should have the same length as t.
      a1, a2, w1, w2, r1, r2 : float
          Calibration parameters.
    
    Returns:
      Array of predicted output intensities. Values range from 0 to 0.5 (at most 1/2 the input intensity, because the first element is a linear polarizer)
    """
    
    # Preallocate an array for predictions.
    prediction = np.empty_like(t, dtype=float)
    
    for i in range(len(t)):
        # Assume linear_polarizer and linear_retarder are defined elsewhere.
        prediction[i] = float(
            A @ linear_polarizer(a2) @ linear_retarder(5 * t[i] + w2, np.pi/2 + r2) @ 
            M_identity @ linear_retarder(t[i] + w1, np.pi/2 + r1) @ linear_polarizer(a1) @ B 
        )
    
    # Multiply the simulated intensity with the measured input intensity at each angle.
    return prediction


# def I_ultimate_polarimetry(cal_angles, cal_left_intensity, cal_right_intensity, sample_angles, sample_left_intensity, sample_right_intensity):
#     """
#     Function to take input measurements and construct mueller matrices for the left and right beams using I reduction.
#     popt gives best fit parameters for a1, a2, w1, w2, r1, and r2. 1 for generator, 2 for analyzer components. 
#     """

#     initial_guess = [0, 0, 0, 0, 0, 0]
#     parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

#     # Find parameters from calibration of the left spot
#     lnormalized_intensity = cal_left_intensity/(2*max(cal_left_intensity)) # original. Divide by twice the max value to estimate the overall input intensity (split between the two channels)
#     lpopt, lpcov = curve_fit(I_vertical_calibration_function, cal_angles, lnormalized_intensity, p0=initial_guess, bounds=parameter_bounds)

#     # Find parameters from calibration of the right spot
#     rnormalized_intensity = cal_right_intensity/(2*max(cal_right_intensity)) # original. 
#     rpopt, rpcov = curve_fit(I_horizontal_calibration_function, cal_angles, rnormalized_intensity, p0=initial_guess, bounds=parameter_bounds)

#     # Optional print the calibration matrices (should be close to identity) to see how well the parameters compensate
#     MlCal = I_calibrated_full_mueller_polarimetry(cal_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5], cal_left_intensity, LPA_angle=np.pi/2)
#     MlCal = MlCal/np.max(np.abs(MlCal))
#     MrCal = I_calibrated_full_mueller_polarimetry(cal_angles, rpopt[0], rpopt[1], rpopt[2], rpopt[3], rpopt[4], rpopt[5], cal_right_intensity)
#     MrCal = MrCal/np.max(np.abs(MrCal))

#     # Calculate RMS error of each calibration matrix
#     lRMS_Error = RMS_calculator(MlCal)
#     rRMS_Error = RMS_calculator(MrCal)

#     # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample for left and right beams (Ml and Mr)
#     Ml = I_calibrated_full_mueller_polarimetry(sample_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5], sample_left_intensity, LPA_angle=np.pi/2)
#     Ml = Ml/np.max(np.abs(Ml))

#     Mr = I_calibrated_full_mueller_polarimetry(sample_angles, rpopt[0], rpopt[1], rpopt[2], rpopt[3], rpopt[4], rpopt[5], sample_right_intensity)
#     Mr = Mr/np.max(np.abs(Mr))

#     np.set_printoptions(suppress=True)

#     # Use the polar decomposition of the retarder matrix 
#     l_retarder_decomposed_MSample = decompose_retarder(Ml, normalize=True)
#     r_retarder_decomposed_MSample = decompose_retarder(Mr, normalize=True)

#     # retardance = np.arccos(np.trace(normalized_decompose_retarder(r_decomposed_MSample))/2 - 1)/(2*np.pi) # Value in waves
#     lretardance = np.arccos(np.trace(r_retarder_decomposed_MSample)/2 - 1)/(2*np.pi) # Value in waves
#     rretardance = np.arccos(np.trace(r_retarder_decomposed_MSample)/2 - 1)/(2*np.pi)

#     lRetardance_Error = propagated_error(l_retarder_decomposed_MSample, lRMS_Error)
#     rRetardance_Error = propagated_error(r_retarder_decomposed_MSample, rRMS_Error)

#     # Extract retardance from the last entry of the mueller matrix, which should just be cos(phi)
#     # lretardance = np.arccos(Ml[3,3])/(2*np.pi)
#     # rretardance = np.arccos(Mr[3,3])/(2*np.pi)

#     # avg_retardance = (lretardance+rretardance)/2

#     return Ml, Mr, lretardance, rretardance, lRetardance_Error, rRetardance_Error, MlCal, MrCal, lpopt, rpopt


def I_ultimate_polarimetry(cal_angles, cal_left_intensity, cal_right_intensity, sample_angles, sample_left_intensity, sample_right_intensity):
    """
    New version!!! Now the calibration function takes intensity values that are normalized by the total intensity at each angle.
    
    Function to take input measurements and construct mueller matrices for the left and right beams using I reduction.
    popt gives best fit parameters for a1, a2, w1, w2, r1, and r2. 1 for generator, 2 for analyzer components. 
    """

    initial_guess = [0, 0, 0, 0, 0, 0]
    parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

    Cal_I_total_meas = cal_left_intensity + cal_right_intensity
    lnormalized_intensity = cal_left_intensity/(2*Cal_I_total_meas) # normalize values from 0 to 0.5 to match the output of the simulation function
    rnormalized_intensity = cal_right_intensity/(2*Cal_I_total_meas)

    # Find parameters from calibration of the left spot
    lpopt, lpcov = curve_fit(I_vertical_calibration_function, cal_angles, lnormalized_intensity, p0=initial_guess, bounds=parameter_bounds)

    # plt.plot(cal_angles, lnormalized_intensity, label='l measured (normalized)', marker='o')
    # plt.plot(cal_angles, I_vertical_calibration_function(cal_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5]), label='l predicted', marker='o')
    # plt.legend()

    # Find parameters from calibration of the right spot
    rpopt, rpcov = curve_fit(I_horizontal_calibration_function, cal_angles, rnormalized_intensity, p0=initial_guess, bounds=parameter_bounds)

    # plt.plot(cal_angles, rnormalized_intensity, label='r measured (normalized)', marker='o')
    # plt.plot(cal_angles, I_horizontal_calibration_function(cal_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5]), label='r predicted', marker='o')
    # plt.legend()

    # Optional print the calibration matrices (should be close to identity) to see how well the parameters compensate
    MlCal = I_calibrated_full_mueller_polarimetry(cal_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5], cal_left_intensity, LPA_angle=np.pi/2) # intensities don't need to be normalized here
    MlCal = MlCal/np.max(np.abs(MlCal))
    MrCal = I_calibrated_full_mueller_polarimetry(cal_angles, rpopt[0], rpopt[1], rpopt[2], rpopt[3], rpopt[4], rpopt[5], cal_right_intensity)
    MrCal = MrCal/np.max(np.abs(MrCal))

    # Calculate RMS error of each calibration matrix
    lRMS_Error = RMS_calculator(MlCal)
    rRMS_Error = RMS_calculator(MrCal)

    # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample for left and right beams (Ml and Mr)
    Ml = I_calibrated_full_mueller_polarimetry(sample_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5], sample_left_intensity, LPA_angle=np.pi/2) # intensities don't need to be normalized
    Ml = Ml/np.max(np.abs(Ml))

    Mr = I_calibrated_full_mueller_polarimetry(sample_angles, rpopt[0], rpopt[1], rpopt[2], rpopt[3], rpopt[4], rpopt[5], sample_right_intensity)
    Mr = Mr/np.max(np.abs(Mr))

    np.set_printoptions(suppress=True)

    # Use the polar decomposition of the retarder matrix 
    l_retarder_decomposed_MSample = decompose_retarder(Ml, normalize=True)
    r_retarder_decomposed_MSample = decompose_retarder(Mr, normalize=True)

    # retardance = np.arccos(np.trace(normalized_decompose_retarder(r_decomposed_MSample))/2 - 1)/(2*np.pi) # Value in waves
    lretardance = np.arccos(np.trace(r_retarder_decomposed_MSample)/2 - 1)/(2*np.pi) # Value in waves
    rretardance = np.arccos(np.trace(r_retarder_decomposed_MSample)/2 - 1)/(2*np.pi)

    lRetardance_Error = propagated_error(l_retarder_decomposed_MSample, lRMS_Error)
    rRetardance_Error = propagated_error(r_retarder_decomposed_MSample, rRMS_Error)

    # Extract retardance from the last entry of the mueller matrix, which should just be cos(phi)
    # lretardance = np.arccos(Ml[3,3])/(2*np.pi)
    # rretardance = np.arccos(Mr[3,3])/(2*np.pi)

    # avg_retardance = (lretardance+rretardance)/2

    return Ml, Mr, lretardance, rretardance, lRetardance_Error, rRetardance_Error, MlCal, MrCal, lRMS_Error, rRMS_Error, lpopt, rpopt
