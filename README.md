## Analysis for the publication: "Significance of Gamma-ray Detector Efficiency Calibration Bias on Nuclear Astrophysics and Nuclear Data Evaluations"

### Written by: Eric F. Matthews, Department of Nuclear Engineering, University of California - Berkeley

This repo contains the reproducable workflow that was used to obtain data for the impending publication "Significance of Gamma-ray Detector Efficiency Calibration Bias on Nuclear Astrophysics and Nuclear Data Evaluations". This code and Jupyter notebook are presented so that readers may review and scrutinize the dataset and analysis for this publication. 

The majority of this analysis was conducted and annotated in the Jupyter notebook `Analysis.ipynb`. Execute the following steps in order to access this notebook: 

* Install JupyterLab (https://jupyter.org/install). 

* `cd` into the directory where you have cloned this repo. 

* Enter the command `jupyter notebook`. Assuming you have installed JupyterLab correctly, a tab with this directory will appear in your default web browser. 

* Navigate to this newly opened tab and click on `Analysis.ipynb` this will open another tab that displays the analysis notebook.

* Open this new tab to view the notebook. 

* If you wish to rerun the analysis code that is presented in the notebook, navigate to the `Kernel` tab and select `Restart and Run All`. 
 * Note that the first time you run this notebook there will be an approximately 30-60 minute run time. This is because significant computational power will be required to re-perform the gamma spectroscopy fitting and Monte-Carlo uncertainty analysis. Subsequent runs require approximately 2 minutes due to variable serialization that is explained in the notebook. 


Please view `license.txt` to view the licensing and restrictions for this data and code. 
