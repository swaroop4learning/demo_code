Steps to setup environments to run code and reproduce results
================================================================

It is suggested to maintain 2 different environments to run the code, one for the scan component and the other to run the application.

The below steps are for setting up the environment using Anaconda. Please refer to the below link for more details:

https://docs.anaconda.com/anaconda/install/index.html


Steps to Run the code
=====================
1. Launch Terminal and ensure you are in the complai_examples directory

Optional Steps to RUN/RERUN SCAN
=================================
Run these steps to generate trust scores for a given model through complai Scan.
Feel free to skip to next section for viewing already generated complai scan results.

0. Create and switch to new Anaconda Environment:
"conda create -n <NEW SCAN ENVIRONMENT NAME> python=3.8"
"conda activate <NEW SCAN ENVIRONMENT NAME>"

1. Install complai_scan using pip, using below code.
"pip install <PATH TO INSTALLABLE>/complai_scan-0.1.1-mlflow-integration.tar.gz"
Ignore warnings and error messages if any while package is getting installed in your virtual environment.
This may take a while to install based on your hardware setup. Kindly wait with patience

2. Install protobuf specific version as tensorflow internally uses protobuf and this step is to make sure tensorflow dependencies are handled
"pip install protobuf==3.19.4"

3. To run the scan, move to the respective directory.
For e.g. : "cd lung_cancer_lr"

4. Run the code using the below command.
"python run_binary.py"

5. Similarly repeat step3 and step4 for remaining use cases

**Note**: 
1. Each scan initiation may take from 5 to 15 minutes dependending upon the hardware availability.
2. lung_cancer_lr can be interpreted as Logistic Regression model trained on Lung Cancer dataset. Remaining use cases also follow similar naming convention.

Steps to LAUNCH APP
===================
Follow these steps for launching the complAI webapp and to view scanned results of ML models.

0. Create and switch to new Anaconda Environment:
conda deactivate if you are already inside an existing virtual environment or open new terminal and execute below commands for creating new virtual environment
"conda create -n <NEW APP ENVIRONMENT NAME> python=3.8"
"conda activate <NEW APP ENVIRONMENT NAME>"

1. Inside complai_examples directory go into the complai_ui directory using the below command.
"cd complai_ui"

2. Install app dependencies (Preferably in a new env), using the below command
"pip install -r requirements.txt"

3. Install protobuf specific version as tensorflow internally uses protobuf and this step is to make sure tensorflow dependencies are handled
"pip install protobuf==3.19.4"

4. Run the below command to launch the streamlit app. Best viewed in Chrome Browser
"streamlit run app.py"

Note: If you are on Dark Theme then please go to settings of the streamlit app and select Theme as Light. Changing Theme is very easy as explained in this link
https://blog.streamlit.io/content/images/2021/08/Theming-1--No-Padding---1-.gif

Steps to Configure YAML
=======================

1. This project requires the user to configure 2 yaml files. 
For e.g. :"lung_cancer_lr_config.yaml" and "lung_cancer_lr_policy.yaml"

 Please refer to the yaml files for further information

Troubleshooting steps
=======================
1. If the scan fails in between it could be there are some hidden files in your project folder structure.
Most common one in macOS is presence of hidden files ".DS_Store".
Go to complai directory or sub-directory and run the command to delete such hidden files "rm -f .DS_Store"
2. It is recommended to close all heavy memory based applications before running the scan as complai scan needs best available system resources.
3. In case you see any errors at the time of running code or installing packages, try removing cache and install again. To remove pip cache please use the following command "pip3 cache purge"
4. This setup is not recommended for PC devices with Apple silcon M1 chip.