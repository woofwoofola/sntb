# Instructions to use this package for the Smear Negative Tuberculosis paper

Note: there is no data or personally identifiable information in this package

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
