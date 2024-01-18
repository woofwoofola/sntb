# Instructions to use this package for the Smear Negative Tuberculosis paper

Note: there are no data or personally identifiable information in this package.  
Note: Except for TabPFN, all of the models used are not deep learning models.
TabPFN is a pre-trained 12-layer transformer model intended for continuous variables.

## Install the required packages
- pip install -r requirements.txt

## Learn the model
- Obtain TB.csv and put it in the input-data directory
- Create an output folder
- Edit the variables at the beginning of andre16.py
  - To choose what variables to use as inputs
  - To choose what baseline to use for net reclassification measures
  - To choose whether or not the input continuous variables are converted to categoricals, or not 
  - To choose the kind of validation: apparent validation, cross-validation, train/test splits or bootstrapping
  - To choose which models to use
  - To choose whether or not to use stratification
  - To choose how many folds and repetitions 
  - To choose what groups in ROC to use for group-wise measures
  - To choose what kind of hyperparameter optimization: random, bayesian with a tree of parzen estimators, or bayesian with adaptive tree...
  - To choose how many hyperparameter iterations (search points) 
  - To choose what measure to optmize: AUC, concordant partial AUC normalized in the 3rd group (cpAUCn.3), etc
- python3 andre16.py
- Observe the output on the screen (also captured in a log file) and in the output folder

## Analyze the results for cross validation or train/test splits
- Edit the variables at the beginning of AnalyzeDeepROCFolds.py
- python AnalyzeDeepROCFolds.py
- Observe the output on the screen (also captured in a log file) and in the output folder

## Analyze the results for apparent validation
- Edit the variables at the beginning of analyzeApparentValidation.py
  - To choose what results to analyze (and if they are in resultsv2 form)
  - To choose what baseline to use/load for NRI analysis
- python3 analyzeApparentValidation.py
- Observe the output on the screen (also captured in a log file) and in the output folder
