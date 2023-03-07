# Instructions to use this package for the Smear Negative Tuberculosis paper

Note: there are no data or personally identifiable information in this package
Note: Except for TabPFN, all of the models used are not deep learning models.
TabPFN is a pre-trained 12-layer transformer model.

## Install the required packages
- pip install -r requirements.txt

## Learn the model with 10x10 cross-validation
- Obtain TB.csv and put it in the input-data directory
- Create an output folder
- Edit the variables at the beginning of andre16.py
- python andre16.py
- Observe the output on the screen (also captured in a log file) and in the output folder

## Analyze the results with deep ROC analysis
- Edit the variables at the beginning of AnalyzeDeepROCFolds.py
- python AnalyzeDeepROCFolds.py
- Observe the output on the screen (also captured in a log file) and in the output folder
