# Author: Travis Hammond

import numpy as np
import numpy.lib.recfunctions as rf
import sys
from os import listdir
from os.path import dirname, basename
import matplotlib.pyplot as plt

from hapiclient import hapitime2datetime

__version__ = '0.1.0'

MODEL_ENGINE = 'TORCH'
# MODEL_ENGINE = 'TENSORFLOW'


class HAPINNTrainer:
    """A class for preparing HAPI data and putting it in
       a form that can be used to train time series neural networks.
    """

    def __init__(self, data_split, in_steps, out_steps,
                 preprocess_func=None, lag=True):
        """Initalizes PyTorch or Tensorflow Modules as well
           as other parameters.

        Args:
            data_split: A list or tuple of three values that sum to 1,
                        where each value represents that percentage of
                        the data split among train, validation, and test
                        sets, respectively.
            in_steps: An integer, which represents the number of
                      data points from the time series data to
                      include as input into a model
            out_steps: An integer, which represents the number of
                       data points the model should output
            preprocess_func: A function that is called on the
                             different data splits, which
                             should be used to normalize the data,
                             fix invalid values, and other processing
                             before the data is more heavely formatted
                             for training.
            lag: A boolean, which determines if the input should lag one
                 timestep behind the expected output. Defaults to True.
                 True implies the model is used for forecasting.
        """
        global torch, nn, tf, TensorDataset, DataLoader
        if (('torch' not in sys.modules or 'torch.nn' not in sys.modules
                or 'torch.utils.data' in sys.modules)
                and 'tensorflow' not in sys.modules):
            if MODEL_ENGINE == 'TORCH':
                import torch
                import torch.nn as nn
                from torch.utils.data import TensorDataset, DataLoader
            elif MODEL_ENGINE == 'TENSORFLOW':
                import tensorflow as tf

        if abs(sum(data_split) - 1) > 1e-9:
            raise ValueError(
                'data_split values should be equal to 1.'
            )
        if len(data_split) != 3:
            raise ValueError('data_split should have 3 values.')
        self.data_split = data_split
        if not isinstance(in_steps, int):
            raise TypeError('in_steps must be an int.')
        if not isinstance(out_steps, int):
            raise TypeError('out_steps must be an int.')
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.data = None
        self.y_data = None
        self.processed_data = None
        self.processed_test_data = None
        self.preprocess_func = preprocess_func
        self.lag = lag

    def set_hapidatas(self, datas, xyparameters=None):
        """Gives the Trainer the HAPI data and checks
           that the datas can be used for training.

        Args:
            datas: A list or tuple of hapi data, which
                   has same data columns and same intervals.
                   The datas should not have same columns besides
                   for time.
            xyparameters: A list or tuple of lists which contain
                          the column names that indicate the
                          wanted columns in x input and y input,
                          respectively.

        """
        if (xyparameters is not None
                and not isinstance(xyparameters, (list, tuple))):
            raise ValueError(
                'xyparameters should be None, list type, or tuple type.'
            )
        datas = datas if isinstance(datas, (list, tuple)) else [datas]
        self.processed_data = None
        self.processed_test_data = None

        time = datas[0]['Time']
        timedata = hapitime2datetime(time)
        if np.vectorize(lambda x: x.total_seconds())(
                timedata[1:] - timedata[:-1]).std() != 0.0:
            raise NotImplementedError(
                'Date time intervals are not consistent in data.'
            )

        for ndx in range(len(datas)):
            # Make sure time columns are same for all data
            if all(time != datas[ndx]['Time']):
                raise NotImplementedError(
                    'Date time intervals must be the same.'
                )

            # Remove datetime column
            datas[ndx] = datas[ndx][list(datas[ndx].dtype.names[1:])]

        if xyparameters is None:
            self.data = rf.merge_arrays(datas, flatten=True)
        elif xyparameters[0] == xyparameters[1]:
            self.data = rf.merge_arrays(
                [data[[param]] for param in xyparameters[0]
                    for data in datas if param in data.dtype.names],
                flatten=True
            )
        else:
            # from datas combine columns for x and y
            self.data = rf.merge_arrays(
                [data[[param]] for param in xyparameters[0]
                    for data in datas if param in data.dtype.names],
                flatten=True
            )
            self.y_data = rf.merge_arrays(
                [data[[param]] for param in xyparameters[1]
                    for data in datas if param in data.dtype.names],
                flatten=True
            )

    def prepare_data(self):
        """Prepares the data for training by
           processing it and partitioning it.
        """
        self.process_data()
        self.partition_data()

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

    def _process_data(self, data, y_data=None):
        """Processes data and optionally y_data.
           Used internally by process_data method.
        """
        if self.preprocess_func is None:
            preprocessed_data = data
            if y_data is not None:
                preprocessed_y_data = y_data
        else:
            preprocessed_data = self.preprocess_func(data)
            if y_data is not None:
                preprocessed_y_data = self.preprocess_func(y_data)

        preprocessed_data = rf.structured_to_unstructured(
            preprocessed_data
        ).astype(np.float32)

        if y_data is not None:
            preprocessed_y_data = rf.structured_to_unstructured(
                preprocessed_y_data
            ).astype(np.float32)

        # Group Data into In and Out Blocks
        x_data = np.swapaxes(np.lib.stride_tricks.sliding_window_view(
            preprocessed_data, self.in_steps, axis=0
        ), 1, 2)

        if self.in_steps == self.out_steps and y_data is None:
            if self.lag:
                y_data = x_data[1:]
                x_data = x_data[:-1]
            else:
                y_data = x_data.copy()
                x_data = x_data
        else:
            if y_data is None:
                y_data = np.swapaxes(np.lib.stride_tricks.sliding_window_view(
                    preprocessed_data, self.out_steps, axis=0
                ), 1, 2)
            else:
                y_data = np.swapaxes(np.lib.stride_tricks.sliding_window_view(
                    preprocessed_y_data, self.out_steps, axis=0
                ), 1, 2)
                print(y_data.shape)
            min_len = min(y_data.shape[0], x_data.shape[0])
            if self.lag:
                y_data = y_data[1:min_len]
                x_data = x_data[:min_len - 1]
            else:
                y_data = y_data[:min_len]
                x_data = x_data[:min_len]
        return {'x': x_data, 'y': y_data}

    def process_data(self):
        """Processes the data that was set with set_hapidatas method.
        """
        if self.data is None:
            raise ValueError('Data must first be set.')

        data = self.data[:-int(len(self.data) * self.data_split[2])]
        test_data = self.data[-int(len(self.data) * self.data_split[2]):]
        if self.y_data is None:
            self.processed_data = self._process_data(data)
            self.processed_test_data = self._process_data(test_data)
        else:
            y_data = self.y_data[:-int(len(self.y_data) * self.data_split[2])]
            y_test_data = self.y_data[
                -int(len(self.y_data) * self.data_split[2]):
            ]
            self.processed_data = self._process_data(data, y_data)
            self.processed_test_data = self._process_data(
                test_data, y_test_data
            )

    def _partition_data(self, data, split):
        """Partitions data given the percentage of the left split result.
           Used internally by partition_data method.
        """
        if len(data) != 2:
            raise ValueError('Data should only have two dict entries.')
        length = len(data['x'])
        ndxs = np.random.choice(
            np.arange(length),
            size=int(split * length),
            replace=False
        )
        data1 = {'x': data['x'][ndxs], 'y': data['y'][ndxs]}
        data2 = {
            'x': np.delete(data['x'], ndxs, axis=0),
            'y': np.delete(data['y'], ndxs, axis=0)
        }
        return data1, data2

    def partition_data(self):
        """Partitions the data that was processed by process_data method.
        """
        if self.processed_data is None:
            raise ValueError('Data must first be processed.')
        if self.processed_test_data is None:
            raise ValueError('Test data must first be processed.')
        if 'train_x' in self.processed_data:
            raise ValueError(
                'Data is already partitioned. Reprocess first if need be.')

        train_val_split = self._partition_data(
            self.processed_data, self.data_split[0] /
            (self.data_split[0] + self.data_split[1])
        )
        processed_data = {'train_x': train_val_split[0]['x'],
                          'train_y': train_val_split[0]['y'],
                          'val_x': train_val_split[1]['x'],
                          'val_y': train_val_split[1]['y'],
                          'test_x': self.processed_test_data['x'],
                          'test_y': self.processed_test_data['y']}
        self.processed_data = processed_data

    def torch_train(self, model, loss_func,
                    optimizer, epochs, device, batch_size=None,
                    verbose=1):
        """Trains and evaluates a torch model.

        Args:
            model: A PyTorch Module.
            loss_func: A torch loss function.
            optimizer: A torch optimizer.
            epochs: An integer, the number of training epochs.
            device: A string, the device to use gpu/cpu etc.
            batch_size: An integer, the size of each batch for training.
            verbose: An integer, rates verbosity. 0 None, 1 All.
        Returns:
            A dict of results for train, validation, and test.
        """
        assert MODEL_ENGINE == 'TORCH'
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
            if verbose:
                print(f'Epoch {epoch + 1}/{epochs}', end='\r')

            model.train(mode=True)
            epoch_loss = 0
            batch_length = len(train_loader)
            for batch_ndx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                loss = loss.item()
                epoch_loss += loss
                if verbose:
                    string = (f'Epoch: {epoch + 1}/{epochs} - '
                              f'Batch: {batch_ndx + 1}/{batch_length} - '
                              f'Loss: {epoch_loss / (batch_ndx + 1):.6f}\t\t')
                    str_length = len(string)
                    print(string, end='\r')
            if verbose:
                string = (f'Epoch: {epoch + 1}/{epochs} - '
                          f'Batch: {batch_ndx + 1}/{batch_length} - '
                          f'Loss: {(epoch_loss / (batch_ndx + 1)):.6f}')

            model.train(mode=False)
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            if verbose:
                string += ('- Validation Loss: '
                           f'{(epoch_loss / (batch_ndx + 1)):.6f}')
                print(string + " " * (str_length - len(string)))

        results = {}
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        results['train'] = (epoch_loss / (batch_ndx + 1))
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        results['val'] = (epoch_loss / (batch_ndx + 1))
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        results['test'] = (epoch_loss / (batch_ndx + 1))

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
        assert MODEL_ENGINE == 'TENSORFLOW'
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
              loss_func=None, optimizer=None, device=None):
        """Trains and evaluates a tensorflow or torch model.

        Args:
            model: A PyTorch/TensorFlow Model.
            epochs: An integer, the number of training epochs.
            batch_size: An integer, the size of each batch for training.
            loss_func: A torch loss function.
            optimizer: A torch optimizer.
            device: A string, the device to use gpu/cpu etc.
        Returns:
            A dict of results for train, validation, and test.
        """
        if MODEL_ENGINE == 'TORCH':
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
                batch_size=batch_size)
        else:
            results = self.tf_train(model, epochs, batch_size=batch_size)
        return results


class HAPINNTester:
    def __init__(self, in_steps, out_steps,
                 preprocess_func=None, postprocess_func=None):
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
            postprocess_func: A function that is called on the
                              outputs of the model, which
                              should be used to denormalize the data
                              etc.
        """
        global torch, nn, tf, TensorDataset, DataLoader
        if (('torch' not in sys.modules or 'torch.nn' not in sys.modules
                or 'torch.utils.data' in sys.modules)
                and 'tensorflow' not in sys.modules):
            if MODEL_ENGINE == 'TORCH':
                import torch
                import torch.nn as nn
                from torch.utils.data import TensorDataset, DataLoader
            elif MODEL_ENGINE == 'TENSORFLOW':
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
        self.postprocess_func = (
            lambda x: x) if postprocess_func is None else postprocess_func

    def set_hapidatas(self, datas, xyparameters=None):
        """Gives the Trainer the HAPI data and checks
           that the datas can be used for training.

        Args:
            datas: A list or tuple of hapi data, which
                   has same data columns and same intervals.
                   The datas should not have same columns besides
                   for time.
            xyparameters: A list or tuple of lists which contain
                          the column names that indicate the
                          wanted columns in x input and y input,
                          respectively.
        """
        if (xyparameters is not None
                and not isinstance(xyparameters, (list, tuple))):
            raise ValueError(
                'xyparameters should be None, list type, or tuple type.'
            )
        datas = datas if isinstance(datas, (list, tuple)) else [datas]
        self.processed_data = None
        self.processed_test_data = None

        time = datas[0]['Time']
        timedata = hapitime2datetime(time)
        if np.vectorize(lambda x: x.total_seconds())(
                timedata[1:] - timedata[:-1]).std() != 0.0:
            raise NotImplementedError(
                'Date time intervals are not consistent in data.'
            )

        for ndx in range(len(datas)):
            # Make sure time columns are same for all data
            if all(time != datas[ndx]['Time']):
                raise NotImplementedError(
                    'Date time intervals must be the same.'
                )

            # Remove datetime column
            datas[ndx] = datas[ndx][list(datas[ndx].dtype.names[1:])]

        if xyparameters is None:
            self.data = rf.merge_arrays(datas, flatten=True)
        elif xyparameters[0] == xyparameters[1]:
            self.data = rf.merge_arrays(
                [data[[param]] for param in xyparameters[0]
                    for data in datas if param in data.dtype.names],
                flatten=True
            )
            self.y_data = self.data
        else:
            # from datas combine columns for x and y
            self.data = rf.merge_arrays(
                [data[[param]] for param in xyparameters[0]
                    for data in datas if param in data.dtype.names],
                flatten=True
            )
            self.y_data = rf.merge_arrays(
                [data[[param]] for param in xyparameters[1]
                    for data in datas if param in data.dtype.names],
                flatten=True
            )

    def prepare_data(self):
        """Prepares the data for testing by
           processing it with preprocessing and
           reformatting.
        """
        self.processed_data = rf.structured_to_unstructured(
            self.preprocess_func(self.data)
        ).astype(np.float32)
        if self.y_data is None:
            self.processed_y_data = self.processed_data
        else:
            self.processed_y_data = rf.structured_to_unstructured(
                self.preprocess_func(self.y_data)
            ).astype(np.float32)

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

        if MODEL_ENGINE == 'TORCH':
            # OPTIMIZE
            for ndx in range(self.in_steps, len(self.processed_data), stride):
                pred = model(torch.Tensor(np.expand_dims(
                    self.processed_data[ndx - self.in_steps:ndx], axis=0
                ))).detach().numpy()[0]
                preds.append(self.postprocess_func(pred))
        else:
            # OPTIMIZE
            for ndx in range(self.in_steps, len(self.processed_data), stride):
                pred = model(np.expand_dims(
                    self.processed_data[ndx - self.in_steps:ndx], axis=0
                )).numpy()[0]
                preds.append(self.postprocess_func(pred))
        return preds

    def forecast_plot(self, preds, pred_ndx, column_name, stride=None):
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
        """
        if stride is None:
            stride = self.out_steps

        if pred_ndx != -1:
            ndx = stride * pred_ndx + self.in_steps
            col_ndx = self.y_data.dtype.names.index(column_name)
            plt.plot(np.append(
                self.y_data[ndx - self.in_steps:ndx][column_name],
                preds[pred_ndx][:, col_ndx]
            ))
            plt.plot(self.y_data[ndx - self.in_steps:ndx +
                     self.out_steps][column_name])
        else:
            if stride != self.out_steps:
                raise NotImplementedError('stride must match out_steps')

            col_ndx = self.y_data.dtype.names.index(column_name)
            forecasts = np.append(
                self.y_data[:self.in_steps][column_name], np.concatenate(preds)
            )

            plt.plot(forecasts)
            plt.plot(self.y_data[column_name])

    def plot(self, preds, pred_ndx, column_name, stride=None):
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
        """
        if self.out_steps > self.in_steps:
            raise Exception('Use forecast_plot instead. out_steps > in_steps.')

        if stride is None:
            stride = self.out_steps

        if pred_ndx != -1:
            ndx = stride * pred_ndx + self.in_steps
            col_ndx = self.y_data.dtype.names.index(column_name)
            plt.plot(np.append(self.y_data[ndx - self.out_steps:ndx -
                     self.in_steps][column_name], preds[pred_ndx][:, col_ndx]))
            plt.plot(self.y_data[ndx - self.in_steps:ndx][column_name])
        else:
            if stride != self.out_steps:
                raise NotImplementedError('stride must match out_steps')
            ndx = stride + self.in_steps
            col_ndx = self.y_data.dtype.names.index(column_name)
            preds = np.concatenate(preds)
            plt.plot(preds)
            plt.plot(self.y_data[column_name])
