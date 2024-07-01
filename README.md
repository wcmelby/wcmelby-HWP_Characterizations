This repository features the code needed to take images and analyze data for a Dual-Rotating Retarder Polarimeter (DRRP). This method uses measurements of the Q Stokes parameter for polarized light to construct the entire Mueller matrix for a sample. This matrix contains information about polarization properties such as the retardance which can be found by this code. In addition, the DRRP system can be precisely calibrated using measurements of air to give an estimate of measurement error. This method uses best fit parameters to match the calibration measurements with a Mueller matrix model for imperfect components. 

The data used here pertains to measurements of an L' band half-wave plate. 

To construct our DRRP you will need at least:<br>
1x SuperK SELECT tunable filter laser light source (or similar)<br>
1x First Light CRED2 ER camera<br>
2x Thorlabs rotation stages<br>
1x linear polarizer<br>
2x achromatic quarter-wave plates<br>
1x Wollaston prism<br>

Important files to look at:
L_SuperK_DRRP_Take_Data.py is the latest version of the data-taking script that controls the camera and rotation stages. 
New_DRRP_Functions.py is the latest version of the functions used to analyze the data using measurements of Q. 
Example_Waveplate_Analysis.ipynb is a simple example of how data is extracted from the images and used to find the retardance and other information. 

Special thanks to Jaren Ashcraft for inspiration with the code. 