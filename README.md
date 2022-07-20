# Application of HAPI for Neural Networks in Python
HAPI-NN allows interfacing of HAPI with TensorFlow and PyTorch to rapidly create deep neural network models for predicting and forecasting.

See Test Examples for toy examples that test different functionalties within HAPI-NN.

Real Data Examples are still work-in-progress.


Features
 - A quick conversion from PySpedas plot data to HAPI-Data (as of now, metadata is not included)
 - Trainer
   - Can take data from different HAPI sources and so long as the time column is exactly the same (including time gaps) can be combined.
   - Columns within HAPI data can be selected for input or output or both for the model.
   - Columns with 1D-vectors are supported and elements within the vectors can be subselected as input or output for the model.
   - Preprocessing helper functions that help handle time gaps by either ignoring them or treating each split of data separately.
   - Data proportions for splitting train, validation, and tests set is easily specifiable. (Test splits are still work-in-progress)
   - Can be used to train PyTorch or TensorFlow models with a train method.
 - Tester
   - Has similar capabilities to the Trainer, but does not handle training or time gaps.
   - Predicts all possible predictions from data given some stride.
   - Plotting of forecasts and predictions.
   
TODO
 - Create real example
   - Outlier detector
   - Real-data predictor
 - Improve/Fix Train/Val/Test Split
   - Val split should have no overlap with train like test split is currently
   - Splits should be made from at least several different random locations in the time series
