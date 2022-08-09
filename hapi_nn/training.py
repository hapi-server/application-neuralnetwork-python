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

from .util import extract_format_structured_data
from . import config


class HAPINNTrainer:
    """A class for preparing HAPI data and putting it in
       a form that can be used to train time series neural networks.
    """

    def __init__(self, data_split, in_steps, out_steps,
                 preprocess_func=None, preprocess_y_func=None, lag=True):
        """Initalizes PyTorch or Tensorflow Modules as well
           as other parameters.

        Args:
            data_split: A list or tuple of three values that sum to 1,
                        where each value represents that percentage of
                        the data split among train, validation, and test
                        sets, respectively. Note, train and validation
                        can have overlap, while test has no overlap.
                        Also note, test proportions are before windowing,
                        so the proportions after windowing for test can
                        vary by several percent. Furthermore, the precision
                        of the test proportion is limited to .05
                        (Ex. .17 ~> .15 or .20). Lastly, the larger the data
                        or the smaller the in_steps and out_steps,
                        the less data will be lost due to splitting.
                        Splitting of the data makes the windowing
                        function generate less data points.
                        Recommended Split: [.7, .2, .1]
            in_steps: An integer, which represents the number of
                      data points from the time series data to
                      include as input into a model
            out_steps: An integer, which represents the number of
                       data points the model should output
            preprocess_func: A function that is called on the
                             different input data splits, which
                             should be used to normalize the data,
                             fix invalid values, and other processing
                             before the data is more heavely formatted
                             for training.
            preprocess_y_func: A function that is called on the
                               different output data splits, which
                               should be used to normalize the data,
                               fix invalid values, and other processing
                               before the data is more heavely formatted
                               for training.
            lag: A boolean, which determines if the input should lag one
                 timestep behind the expected output. Defaults to True.
                 True implies the model is used for forecasting.
        """
        global torch, nn, tf, TensorDataset, DataLoader
        if config.MODEL_ENGINE == 'TORCH':
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        elif config.MODEL_ENGINE == 'TENSORFLOW':
            import tensorflow as tf

        if abs(sum(data_split) - 1) > 1e-9:
            raise ValueError(
                'data_split values should be equal to 1.'
            )
        if len(data_split) != 3:
            raise ValueError('data_split should have 3 values.')
        if data_split[2] >= .5:
            raise ValueError('Test split for the data should be less than .5')
        self.data_split = data_split
        if not isinstance(in_steps, int):
            raise TypeError('in_steps must be an int.')
        if not isinstance(out_steps, int):
            raise TypeError('out_steps must be an int.')
        if not lag and out_steps > in_steps:
            raise ValueError('out_steps must be less than '
                             'in_steps if not lagging data')
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.data = None
        self.y_data = None
        self.split_ndxs = None
        self.processed_data = None
        self.processed_train_data = None
        self.processed_val_data = None
        self.processed_test_data = None
        self.preprocess_func = preprocess_func
        self.preprocess_y_func = preprocess_y_func
        self.lag = lag
        self.time_interval = None

    def set_hapidatas(self, datas, xyparameters=None):
        """Gives the Trainer the HAPI data and checks
           that the datas can be used for training.

        Args:
            datas: A list or tuple of hapi data, which
                   has same data columns and same intervals.
                   The datas should not have same columns besides
                   for time. datas can have time gaps, however,
                   the gaps must be the same across all datas.
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
        self.processed_train_data = None
        self.processed_val_data = None
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

        # Split Data on time gaps
        self.data = np.split(self.data, split_ndxs)
        if self.y_data is not None:
            self.y_data = np.split(self.y_data, split_ndxs)

        # Filter out too small of gaps and send warning if too small
        if self.lag:
            min_steps = self.in_steps + self.out_steps
        else:
            min_steps = max(self.in_steps, self.out_steps)
        data = []
        data_ndxs = []
        for ndx, split in enumerate(self.data):
            split_size = int(len(split) / 20)
            if split_size >= min_steps:
                data.append(split)
                data_ndxs.append(ndx)
            else:
                warnings.warn(f'Removed data gab at index {ndx}. '
                              f'Length of gab ({len(split)}) was too small. '
                              f'Split size ({split_size}) is less than '
                              f'minimum step size ({min_steps}).')
        if len(self.data) > len(data):
            warnings.warn('Data points with time gaps that caused '
                          'too small of splits where removed. Removed '
                          f'{len(self.data) - len(data)} out of '
                          f'{len(self.data)} gaps.')
            self.data = data
            if self.y_data is not None:
                self.y_data = [self.y_data[ndx] for ndx in data_ndxs]
        avg_size = int(sum([len(x) for x in data]) / len(data) * .10)
        if avg_size < self.in_steps + self.out_steps:
            warnings.warn('in_steps and out_steps sum to a value greater than '
                          f'10% of the average data gap size ({avg_size}). '
                          'This may reduce the precision of the test split '
                          'and lead to increased data lost from splits.')
        return self.time_interval

    def prepare_data(self):
        """Prepares the data for training by
           processing it and partitioning it.

        Returns:
            A list of the actual resulting partition proportions is returned.
        """
        self.process_data()
        return self.partition_data()

    def save_prepared_data(self, path):
        """Saves the data that is prepared for training.
           The data will be saved as npy files and each file
           represents a value in the processed_data dict.

        Args:
            path: A string that has the path and prefix of the
                  files that will be saved.
        """
        if self.processed_data is None:
            raise ValueError(
                'Data must first be prepared (processed and then partitioned).'
            )
        if 'train_x' not in self.processed_data:
            raise ValueError('Data must first be partitioned.')
        for key in self.processed_data:
            np.save(path + '_' + key, self.processed_data[key])

    def load_prepared_data(self, path):
        """Loads the data that was prepared for training.

        Args:
            path: A string that has the path and prefix of the
                  files that will be loaded.
        """
        self.processed_data = {}
        for di in listdir(dirname(path)):
            if basename(path) + '_' in di:
                self.processed_data[di.split('_', 1)[1].split('.', 1)[
                    0]] = np.load(di)

    def get_prepared_data(self):
        """Returns the processed data that is prepared for training.

        Returns:
            The processed data if it is prepared and partitioned else
            it will return None
        """
        if self.processed_data is not None or 'train_x' in self.processed_data:
            return self.processed_data
        return None

    def _process_data(self, data, y_data):
        """Processes data and optionally y_data.
           Used internally by process_data method.
        """
        if self.preprocess_func is None:
            preprocessed_data_all = data
        else:
            preprocessed_data_all = self.preprocess_func(data)
        if self.preprocess_y_func is None:
            preprocessed_y_data_all = y_data
        else:
            preprocessed_y_data_all = self.preprocess_y_func(y_data)

        x_datas = []
        y_datas = []
        for ndx in range(len(preprocessed_data_all)):
            preprocessed_data = rf.structured_to_unstructured(
                preprocessed_data_all[ndx]
            ).astype(np.float32)

            if preprocessed_y_data_all is not None:
                preprocessed_y_data = rf.structured_to_unstructured(
                    preprocessed_y_data_all[ndx]
                ).astype(np.float32)

            # Group Data into In and Out Blocks
            x_data = np.swapaxes(np.lib.stride_tricks.sliding_window_view(
                preprocessed_data, self.in_steps, axis=0
            ), 1, 2)
            y_data = np.swapaxes(
                np.lib.stride_tricks.sliding_window_view(
                    preprocessed_y_data, self.out_steps, axis=0), 1, 2
            )
            min_len = min(y_data.shape[0], x_data.shape[0])
            if self.lag:
                # CHECK lengths
                y_data = y_data[self.in_steps:min_len]
                x_data = x_data[:min_len - self.in_steps]
            else:
                y_data = y_data[:min_len]
                x_data = x_data[:min_len]
            x_datas.append(x_data)
            y_datas.append(y_data)
        return {'x': np.vstack(x_datas), 'y': np.vstack(y_datas)}

    def process_data(self):
        """Processes the data that was set with set_hapidatas method.
           This method also preforms part of the test partitioning.
        """
        if self.data is None:
            raise ValueError('Data must first be set.')

        # Min steps needed to get one output from windowing
        if self.lag:
            min_steps = self.in_steps + self.out_steps
        else:
            min_steps = max(self.in_steps, self.out_steps)

        # Split data into segments for later partitioning.
        # The data split has noise in the edges to avoid the
        # chance of a bias in splitting.
        # NEED TO IMPROVE SPLIT ALGO:
        # Tests should be pulled from several different places
        # Currently, only test splits come from test proportion divided by 5%
        # So 10% test split would result in two test splits
        data = []
        y_data = []
        for ndx in range(len(self.data)):
            # Split Data into Sections
            split_stride = int(len(self.data[ndx]) / 20)
            num_splits = int(len(self.data[ndx]) / split_stride)
            ends = np.random.randint(split_stride - min_steps / num_splits,
                                     split_stride + min_steps / num_splits,
                                     num_splits)
            cndx = 0
            for end in ends:
                if len(self.data[ndx]) - cndx < min_steps:
                    break
                data.append(self.data[ndx][cndx:cndx + end])
                if self.y_data is not None:
                    y_data.append(self.y_data[ndx][cndx:cndx + end])
                cndx += end
            else:
                if len(self.data[ndx]) - cndx >= min_steps:
                    data.append(self.data[ndx][cndx:])
                    if self.y_data is not None:
                        y_data.append(self.y_data[ndx][cndx:])

        # Sample Sections for Test Data
        ndxs = np.random.choice(
            np.arange(
                len(data) -
                1),
            size=round(
                len(data) *
                self.data_split[2]),
            replace=False)
        ndxs2 = np.random.choice(
            np.delete(np.arange(len(data) - 1), ndxs, axis=0),
            size=round(
                len(data) *
                self.data_split[1]),
            replace=False)
        
        data = np.array(data)
        test_data = data[ndxs]
        val_data = data[ndxs2]
        remerge_data = []
        last_ndx = 0
        for ndx in sorted(np.append(ndxs, ndxs2)):
            if last_ndx != ndx:
                remerge_data.append(np.concatenate(data[last_ndx:ndx]))
            last_ndx = ndx + 1
        if ndx + 1 < len(data):
            remerge_data.append(np.concatenate(data[ndx + 1:]))
        data = np.array(remerge_data)

        if self.y_data is not None:
            y_data = np.array(y_data)
            y_test_data = y_data[ndxs]
            y_val_data = y_data[ndxs2]
            remerge_data = []
            last_ndx = 0
            for ndx in sorted(np.append(ndxs, ndxs2)):
                if last_ndx != ndx:
                    remerge_data.append(np.concatenate(y_data[last_ndx:ndx]))
                last_ndx = ndx + 1
            if ndx + 1 < len(y_data):
                remerge_data.append(np.concatenate(y_data[ndx + 1:]))
            y_data = np.array(remerge_data)

        # Process the data
        if self.y_data is None:
            self.processed_train_data = self._process_data(data, data.copy())
            self.processed_val_data = self._process_data(
                val_data, val_data.copy())
            self.processed_test_data = self._process_data(
                test_data, test_data.copy())
        else:
            self.processed_train_data = self._process_data(data, y_data)
            self.processed_val_data = self._process_data(
                val_data, y_val_data
            )
            self.processed_test_data = self._process_data(
                test_data, y_test_data
            )

    def partition_data(self):
        """Partitions the data that was processed by process_data method.

        Returns:
            A list of the actual resulting partition proportions is returned.
        """
        if self.processed_train_data is None:
            raise ValueError('Train data must first be processed.')
        if self.processed_val_data is None:
            raise ValueError('Validation data must first be processed.')
        if self.processed_test_data is None:
            raise ValueError('Test data must first be processed.')
        if 'train_x' in self.processed_train_data:
            raise ValueError(
                'Train data is already partitioned. Reprocess first if need be.')

        # Make final dict for training
        processed_data = {'train_x': self.processed_train_data['x'],
                          'train_y': self.processed_train_data['y'],
                          'val_x': self.processed_val_data['x'],
                          'val_y': self.processed_val_data['y'],
                          'test_x': self.processed_test_data['x'],
                          'test_y': self.processed_test_data['y']}
        self.processed_data = processed_data

        # Calculate split proportions
        tn = len(processed_data['train_x'])
        vd = len(processed_data['val_x'])
        tt = len(processed_data['test_x'])
        sm = tn + vd + tt
        return (tn / sm, vd / sm, tt / sm)

    def torch_train(self, model, loss_func,
                    optimizer, epochs, device, batch_size=None,
                    metric_func=None, verbose=1):
        """Trains and evaluates a torch model.

        Args:
            model: A PyTorch Module.
            loss_func: A torch loss function.
            optimizer: A torch optimizer.
            epochs: An integer, the number of training epochs.
            device: A string, the device to use gpu/cpu etc.
            batch_size: An integer, the size of each batch for training.
            metric_func: A torch loss/metric function.
            verbose: An integer, rates verbosity. 0 None, 1 All.
        Returns:
            A dict of results for train, validation, and test.
        """
        assert config.MODEL_ENGINE == 'TORCH'
        if self.processed_data is None:
            raise ValueError(
                'Data must first be prepared (processed and then partitioned).'
            )
        if 'train_x' not in self.processed_data:
            raise ValueError('Data must first be partitioned.')

        train_loader = DataLoader(
            TensorDataset(torch.Tensor(self.processed_data['train_x']),
                          torch.Tensor(self.processed_data['train_y'])),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.Tensor(self.processed_data['val_x']),
                          torch.Tensor(self.processed_data['val_y'])),
            batch_size=batch_size * 2, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(torch.Tensor(self.processed_data['test_x']),
                          torch.Tensor(self.processed_data['test_y'])),
            batch_size=batch_size * 2, shuffle=False
        )

        model.train()

        for epoch in range(epochs):
            start_time = time()
            if verbose:
                print(f'Epoch {epoch + 1}/{epochs}', end='\r')

            model.train(mode=True)
            epoch_loss = 0
            epoch_metric_loss = 0
            batch_length = len(train_loader)
            for batch_ndx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                if metric_func is not None:
                    epoch_metric_loss += metric_func(output, target).item()
                loss.backward()
                optimizer.step()
                loss = loss.item()
                epoch_loss += loss
                if verbose:
                    string = (f'Epoch: {epoch + 1}/{epochs} - '
                              f'Batch: {batch_ndx + 1}/{batch_length} - '
                              f'Loss: {epoch_loss / (batch_ndx + 1):.6f}')
                    if metric_func is not None:
                        string += (
                            ' - Metric Loss: '
                            f'{epoch_metric_loss / (batch_ndx + 1):.6f}'
                        )
                    string += '\t\t'
                    str_length = len(string)
                    print(string, end='\r')
            if verbose:
                time_taken = time() - start_time
                string = (
                    f'Epoch: {epoch + 1}/{epochs} - '
                    f'Batch: {batch_ndx + 1}/{batch_length} - '
                    f'{time_taken:.1f}s '
                    f'{int(1000 * time_taken / (batch_ndx + 1))}ms/step - '
                    f'Loss: {(epoch_loss / (batch_ndx + 1)):.6f}')
                if metric_func is not None:
                    string += (
                        ' - Metric Loss: '
                        f'{epoch_metric_loss / (batch_ndx + 1):.6f}'
                    )

            model.train(mode=False)
            epoch_loss = 0
            epoch_metric_loss = 0
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_func(output, target).item()
                epoch_loss += loss
                if metric_func is not None:
                    epoch_metric_loss += metric_func(output, target).item()
            if verbose:
                string += (' - Validation Loss: '
                           f'{(epoch_loss / (batch_ndx + 1)):.6f}')
                if metric_func is not None:
                    string += (
                        f' - Validation Metric Loss: '
                        f'{epoch_metric_loss / (batch_ndx + 1):.6f}'
                    )
                print(string + " " * (str_length - len(string)))

        results = {}
        epoch_loss = 0
        epoch_metric_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target).item()
            epoch_loss += loss
            if metric_func is not None:
                epoch_metric_loss += metric_func(output, target).item()
        if metric_func is None:
            results['train'] = (epoch_loss / (batch_ndx + 1))
        else:
            results['train'] = [(epoch_loss / (batch_ndx + 1)),
                                (epoch_metric_loss / (batch_ndx + 1))]
        epoch_loss = 0
        epoch_metric_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target).item()
            epoch_loss += loss
            if metric_func is not None:
                epoch_metric_loss += metric_func(output, target).item()
        if metric_func is None:
            results['val'] = (epoch_loss / (batch_ndx + 1))
        else:
            results['val'] = [(epoch_loss / (batch_ndx + 1)),
                              (epoch_metric_loss / (batch_ndx + 1))]
        epoch_loss = 0
        epoch_metric_loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target).item()
            epoch_loss += loss
            if metric_func is not None:
                epoch_metric_loss += metric_func(output, target).item()
        if metric_func is None:
            results['test'] = (epoch_loss / (batch_ndx + 1))
        else:
            results['test'] = [(epoch_loss / (batch_ndx + 1)),
                               (epoch_metric_loss / (batch_ndx + 1))]
        return results

    def tf_train(self, model, epochs, batch_size=None, **kwargs):
        """Trains and evaluates a tensorflow model.

        Args:
            model: A TensorFLow/Keras Model.
            epochs: An integer, the number of training epochs.
            device: A string, the device to use gpu/cpu etc.
            kwargs: Keyword arguments for Keras fit model method.
        Returns:
            A dict of results for train, validation, and test.
        """
        assert config.MODEL_ENGINE == 'TENSORFLOW'
        if self.processed_data is None:
            raise ValueError(
                'Data must first be prepared (processed and then partitioned).'
            )
        if 'train_x' not in self.processed_data:
            raise ValueError('Data must first be partitioned.')

        model.fit(self.processed_data['train_x'],
                  self.processed_data['train_y'],
                  validation_data=(self.processed_data['val_x'],
                                   self.processed_data['val_y']),
                  epochs=epochs, batch_size=batch_size, **kwargs)

        if batch_size is None:
            batch_size = 256
        verbose = kwargs['verbose'] if 'verbose' in kwargs else 'auto'
        results = {
            'train': model.evaluate(self.processed_data['train_x'],
                                    self.processed_data['train_y'],
                                    batch_size=batch_size * 2,
                                    verbose=verbose),
            'val': model.evaluate(self.processed_data['val_x'],
                                  self.processed_data['val_y'],
                                  batch_size=batch_size * 2,
                                  verbose=verbose),
            'test': model.evaluate(self.processed_data['test_x'],
                                   self.processed_data['test_y'],
                                   batch_size=batch_size * 2,
                                   verbose=verbose),
        }
        return results

    def train(self, model, epochs, batch_size=None,
              loss_func=None, metric_func=None, optimizer=None,
              device=None, verbose=1):
        """Trains and evaluates a tensorflow or torch model.

        Args:
            model: A PyTorch/TensorFlow Model.
            epochs: An integer, the number of training epochs.
            batch_size: An integer, the size of each batch for training.
            loss_func: A torch loss function.
            metric_func: A torch loss/metric function.
            optimizer: A torch optimizer.
            device: A string, the device to use gpu/cpu etc.
            verbose: An integer, rates verbosity. 0 None, 1 All.
        Returns:
            A dict of results for train, validation, and test.
        """
        if config.MODEL_ENGINE == 'TORCH':
            if None in [loss_func, optimizer, device]:
                raise ValueError(
                    'loss_func, optimizer, and device must be '
                    'supplied for using PyTorch for models'
                )
            results = self.torch_train(
                model,
                loss_func,
                optimizer,
                epochs,
                device,
                batch_size=batch_size,
                metric_func=metric_func,
                verbose=verbose
            )
        else:
            results = self.tf_train(
                model, epochs, batch_size=batch_size, verbose=verbose
            )
        return results

    @staticmethod
    def ignore_gaps(func):
        """Wraps a preprocess function to ignore gaps.
           Useful when not accessing neighbor elements.

        Args:
            func: A preprocess function that handles structured data
        Returns:
            A wrapped preprocess function
        """
        def ig_func(data):
            split_ndxs = np.cumsum([len(x) for x in data])[:-1]
            data = np.hstack(data)
            data = func(data)
            # if dtype=object, there can be a problem if array does not need to
            # be an object
            data = np.array(np.split(data, split_ndxs))
            return data
        return ig_func

    @staticmethod
    def on_gaps(func):
        """Wraps a preprocess function to apply itself on every array
           that was split because of gaps.
           Useful when accessing neighbor elements.
        Args:
            func: A preprocess function that handles structured data
        Returns:
            A wrapped preprocess function
        """
        def g_func(data):
            return np.array([func(x) for x in data])
        return g_func
