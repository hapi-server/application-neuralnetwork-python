"""
Author: Travis Hammond
©️ 2022 The Johns Hopkins University Applied Physics Laboratory LLC.
"""

import warnings
from time import time
from os import listdir
from datetime import datetime
from os.path import dirname, basename
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt

from hapiclient import hapitime2datetime

from hapi_nn.util import extract_format_structured_data
from . import config


class HAPINNTester:
    def __init__(self, in_steps, out_steps,
                 preprocess_func=None, preprocess_y_func=None):
        """Initalizes PyTorch or Tensorflow Modules as well
           as other parameters.

        Args:
            in_steps: An integer, which represents the number of
                      data points from the time series data to
                      include as input into a model
            preprocess_func: A function that is called on
                             the inputs to the model, which
                             should be used to normalize the data,
                             fix invalid values, and other processing
            preprocess_y_func: A function that is called on the
                               outputs of the model, which
                               should be used to denormalize the data
                               etc.
        """
        global torch, nn, tf, TensorDataset, DataLoader
        if config.MODEL_ENGINE == 'TORCH':
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        elif config.MODEL_ENGINE == 'TENSORFLOW':
            import tensorflow as tf
        if not isinstance(in_steps, int):
            raise TypeError('in_steps must be an int.')
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.data = None
        self.y_data = None
        self.processed_data = None
        self.preprocess_func = (
            lambda x: x) if preprocess_func is None else preprocess_func
        self.preprocess_y_func = (
            lambda x: x) if preprocess_y_func is None else preprocess_y_func
        self.time_interval = None

    def set_hapidatas(self, datas, xyparameters=None):
        """Gives the Trainer the HAPI data and checks
           that the datas can be used for training.

        Args:
            datas: A list or tuple of hapi data, which
                   has same data columns and same intervals.
                   The datas should not have same columns besides
                   for time. Note, datas cannot have time gaps.
            xyparameters: A list or tuple of lists which contain
                          the column names that indicate the
                          wanted columns in x input and y input,
                          respectively.
        Returns:
            A number that represents the time interval between data points.
        """
        if (xyparameters is not None
                and not isinstance(xyparameters, (list, tuple))):
            raise ValueError(
                'xyparameters should be None, list type, or tuple type.'
            )
        datas = datas if isinstance(datas, (list, tuple)) else [datas]
        self.processed_data = None
        self.processed_test_data = None

        # Calculate time interval and split on inconsistencies for
        # gaps in the data
        time = datas[0]['Time']
        time_data = hapitime2datetime(time)
        time_deltas = np.vectorize(lambda x: x.total_seconds())(
            time_data[1:] - time_data[:-1]
        )
        self.time_interval = np.median(time_deltas)
        split_ndxs = np.nonzero(time_deltas != self.time_interval)[0] + 1
        if len(split_ndxs) > 0:
            warnings.warn('Time gaps exist in the data.')

        # Make sure time columns are same for all data and then
        # remove the time column
        for ndx in range(len(datas)):
            if all(time != datas[ndx]['Time']):
                raise NotImplementedError(
                    'Time columns must be the same.'
                )
            datas[ndx] = datas[ndx][list(datas[ndx].dtype.names[1:])]

        # Combine all columns based on X and Y requested parameters
        if xyparameters is None:
            self.data = extract_format_structured_data(
                datas, datas.dtype.names)
        elif xyparameters[0] == xyparameters[1]:
            self.data = extract_format_structured_data(datas, xyparameters[0])
        else:
            self.data = extract_format_structured_data(datas, xyparameters[0])
            self.y_data = extract_format_structured_data(
                datas, xyparameters[1]
            )
        return self.time_interval

    def prepare_data(self):
        """Prepares the data for testing by
           processing it with preprocessing and
           reformatting.
        """
        self.processed_data = rf.structured_to_unstructured(
            self.preprocess_func(self.data)
        ).astype(np.float32)
        if self.y_data is None:
            self.y_data = self.data
            self.processed_y_data = self.preprocess_y_func(self.processed_data)
        else:
            self.processed_y_data = self.preprocess_y_func(
                rf.structured_to_unstructured(
                    self.preprocess_func(self.y_data)
                ).astype(np.float32))

    def test(self, model, stride=None):
        """Tests the model by giving it all inputs
           gathered with the provided stride.
           Useful for then plotting the outputs and
           comparing them to the expected.

        Args:
            model: A PyTorch or TensorFlow model
            stride: An integer, which defaults to out_steps
        Returns:
            A list of predictions.
        """
        if self.data is None:
            raise ValueError('Data must first be set.')
        if self.processed_data is None:
            raise ValueError('Data was not prepared for testing.')
        if stride is None:
            stride = self.out_steps

        preds = []

        if config.MODEL_ENGINE == 'TORCH':
            # OPTIMIZE
            for ndx in range(self.in_steps, len(self.processed_data), stride):
                pred = model(torch.Tensor(np.expand_dims(
                    self.processed_data[ndx - self.in_steps:ndx], axis=0
                )))
                if isinstance(pred, (list, tuple)):
                    pred = [x.detach().numpy()[0] for x in pred]
                else:
                    pred = pred.detach().numpy()[0]
                preds.append(self.preprocess_y_func(pred))
        else:
            # OPTIMIZE
            for ndx in range(self.in_steps, len(self.processed_data), stride):
                pred = model(np.expand_dims(
                    self.processed_data[ndx - self.in_steps:ndx], axis=0
                ))
                if isinstance(pred, (list, tuple)):
                    pred = [x.numpy()[0] for x in pred]
                else:
                    pred = pred.numpy()[0]
                preds.append(self.preprocess_y_func(pred))
        return preds

    def forecast_plot(
            self,
            preds,
            pred_ndx,
            column_name,
            stride=None,
            return_data=False):
        """Creates a line plot to compare the ground truth and predicted
           values of a specific output column for one of the predicted outputs.
           Used when Trainer had lag=True.

        Args:
            preds: A list of predictions from test method.
            pred_ndx: An index for indexing the preds. If -1, displays
                      results for all preds.
            column_name: A string which is a name of a output column.
            stride: A integer, which is the stride used in test method to
                    produce the preds.
            return_data: A boolean, which if True will return data for plotting
                         instead of plotting itself.
        Returns:
            None or if return_data is True, returns a dict of forecast and
            truth which both are tuple of (Time, Values)
        """
        if stride is None:
            stride = self.out_steps

        if pred_ndx != -1:
            ndx = stride * pred_ndx + self.in_steps
            col_ndx = self.y_data.dtype.names.index(column_name)
            forecast = preds[pred_ndx][:, col_ndx]
            truth = self.processed_y_data[ndx - self.in_steps:ndx +
                                          self.out_steps][:, col_ndx]
            forecast_time = np.arange(
                ndx, len(forecast) + ndx
            ) * self.time_interval
            truth_time = np.arange(
                ndx - self.in_steps, ndx + self.out_steps
            ) * self.time_interval
        else:
            if stride != self.out_steps:
                raise NotImplementedError('stride must match out_steps')
            col_ndx = self.y_data.dtype.names.index(column_name)
            forecast = np.concatenate(preds)[:, col_ndx]
            truth = self.processed_y_data[:, col_ndx]
            forecast_time = np.arange(
                self.in_steps, len(forecast) + self.in_steps
            ) * self.time_interval
            truth_time = np.arange(len(truth)) * self.time_interval
        forecast_time = forecast_time.astype('timedelta64[s]')
        truth_time = truth_time.astype('timedelta64[s]')

        if return_data:
            return {'forecast': (forecast_time, forecast),
                    'truth': (truth_time, truth)}
        else:
            plt.title('Test Forecast')
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Value')
            plt.plot(forecast_time, forecast, color='C0')
            plt.plot(truth_time, truth, color='C1')
            plt.legend(['forecast', 'truth'])

    def plot(
            self,
            preds,
            pred_ndx,
            column_name,
            stride=None,
            return_data=False):
        """Creates a line plot to compare the ground truth and predicted
           values of a specific output column for one of the predicted outputs.
           Used when Trainer had lag=False.

        Args:
            preds: A list of predictions from test method.
            pred_ndx: An index for indexing the preds. If -1, displays
                      results for all preds.
            column_name: A string which is a name of a output column.
            stride: A integer, which is the stride used in test method to
                    produce the preds.
            return_data: A boolean, which if True will return data for plotting
                         instead of plotting itself.
        Returns:
            None or if return_data is True, returns a dict of prediction
            and truth which both are tuple of (Time, Values)
        """
        if self.out_steps > self.in_steps:
            raise Exception('Use forecast_plot instead. out_steps > in_steps.')

        if stride is None:
            stride = self.out_steps

        if pred_ndx != -1:
            ndx = stride * pred_ndx + self.in_steps
            col_ndx = self.y_data.dtype.names.index(column_name)
            prediction = preds[pred_ndx][:, col_ndx]
            truth = self.processed_y_data[ndx - self.in_steps:ndx][:, col_ndx]
            prediction_time = np.arange(
                ndx - self.in_steps, len(prediction) + (ndx - self.in_steps)
            ) * self.time_interval
            truth_time = np.arange(
                ndx - self.in_steps, ndx
            ) * self.time_interval
        else:
            if stride != self.out_steps:
                raise NotImplementedError('stride must match out_steps')
            ndx = stride + self.in_steps
            col_ndx = self.y_data.dtype.names.index(column_name)
            prediction = np.concatenate(preds)[:, col_ndx]
            truth = self.processed_y_data[:, col_ndx]
            prediction_time = np.arange(len(prediction)) * self.time_interval
            truth_time = np.arange(len(truth)) * self.time_interval

        if return_data:
            return {'prediction': (prediction_time, prediction),
                    'truth': (truth_time, truth)}
        else:
            plt.title('Test Prediction')
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Value')
            plt.plot(prediction_time, prediction, color='C0')
            plt.plot(truth_time, truth, color='C1')
            plt.legend(['prediction', 'truth'])
