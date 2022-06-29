# Author: Travis Hammond

import warnings
from time import time
from os import listdir
from os.path import dirname, basename
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt

from hapiclient import hapitime2datetime

MODEL_ENGINE = 'TORCH'
# MODEL_ENGINE = 'TENSORFLOW'


def extract_format_structured_data(data, parameters):
    new_data = []
    for dat in data:
        for param in parameters:
            if param in dat.dtype.names:
                dp = dat[param]
                if len(dp.shape) > 1:
                    for ndx in range(dp.shape[1]):
                        new_data.append(np.array(
                            dp[:, ndx], dtype=[(param + f'_{ndx}', dp.dtype)]
                        ))
                else:
                    new_data.append(dat[[param]])
            elif '_' in param:
                base_param, ndx = param.rsplit('_', 1)
                ndx = int(ndx)
                dp = dat[base_param]
                if base_param in dat.dtype.names and len(
                        dat.dtype[base_param].shape) > 0:
                    new_data.append(np.array(
                        dp[:, ndx], dtype=[(base_param + f'_{ndx}', dp.dtype)]
                    ))
    return rf.merge_arrays(new_data, flatten=True)


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
                        sets, respectively. Note, train and validation
                        can have overlap, while test has no overlap.
                        Also note, test proportions are before windowing,
                        so the proportions after windowing for test can
                        vary by several percent. Furthermore, the precision
                        of the test proportion is limited to tenths
                        (Ex. .15 ~> .1 or .2). Lastly, the larger the data
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
        self.processed_test_data = None
        self.preprocess_func = preprocess_func
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
            split_size = int(len(split) / 10)
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

    def _process_data(self, data, y_data=None):
        """Processes data and optionally y_data.
           Used internally by process_data method.
        """
        if self.preprocess_func is None:
            preprocessed_data_all = data
            if y_data is None:
                preprocessed_y_data_all = None
            else:
                preprocessed_y_data_all = y_data
        else:
            preprocessed_data_all = self.preprocess_func(data)
            if y_data is None:
                preprocessed_y_data_all = None
            else:
                preprocessed_y_data_all = self.preprocess_func(y_data)

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

            if (self.in_steps == self.out_steps
                    and preprocessed_y_data_all is None):
                if self.lag:
                    # CHECK lengths
                    y_data = x_data[self.in_steps:]
                    x_data = x_data[:-self.in_steps]
                else:
                    y_data = x_data.copy()
                    x_data = x_data
            else:
                if preprocessed_y_data_all is None:
                    y_data = np.swapaxes(
                        np.lib.stride_tricks.sliding_window_view(
                            preprocessed_data, self.out_steps, axis=0), 1, 2
                    )
                else:
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
        data = []
        y_data = []
        for ndx in range(len(self.data)):
            # Split Data into Sections
            split_stride = int(len(self.data[ndx]) / 10)
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
        data = np.array(data)
        test_data = data[ndxs]
        data = np.delete(data, ndxs, axis=0)
        if self.y_data is not None:
            y_data = np.array(y_data)
            y_test_data = y_data[ndxs]
            y_data = np.delete(y_data, ndxs, axis=0)

        # Process the data
        if self.y_data is None:
            self.processed_data = self._process_data(data)
            self.processed_test_data = self._process_data(test_data)
        else:
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

        # Randomly sample to split into two data dicts
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

        Returns:
            A list of the actual resulting partition proportions is returned.
        """
        if self.processed_data is None:
            raise ValueError('Data must first be processed.')
        if self.processed_test_data is None:
            raise ValueError('Test data must first be processed.')
        if 'train_x' in self.processed_data:
            raise ValueError(
                'Data is already partitioned. Reprocess first if need be.')

        # Split data and make final dict for training
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
                batch_size=batch_size,
                metric_func=metric_func,
                verbose=verbose
            )
        else:
            results = self.tf_train(
                model, epochs, batch_size=batch_size, verbose=verbose
            )
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
            forecast = np.append(
                self.y_data[ndx - self.in_steps:ndx][column_name],
                preds[pred_ndx][:, col_ndx]
            )
            truth = self.y_data[ndx - self.in_steps:ndx +
                                self.out_steps][column_name]
        else:
            if stride != self.out_steps:
                raise NotImplementedError('stride must match out_steps')
            col_ndx = self.y_data.dtype.names.index(column_name)
            forecasts = np.append(
                self.y_data[:self.in_steps][column_name],
                np.concatenate(preds)[:, col_ndx]
            )
            truth = self.y_data[column_name]
        forecast_time = np.arange(len(forecasts)) * self.time_interval
        truth_time = np.arange(len(truth)) * self.time_interval

        if return_data:
            return {'forecast': (forecast_time, forecast),
                    'truth': (truth_time, truth)}
        else:
            plt.title('Test Forecast')
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Value')
            plt.plot(forecast_time, forecast)
            plt.plot(truth_time, truth)
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
            prediction = np.append(
                self.y_data[ndx - self.out_steps:
                            ndx - self.in_steps][column_name],
                preds[pred_ndx][:, col_ndx]
            )
            truth = self.y_data[ndx - self.in_steps:ndx][column_name]
        else:
            if stride != self.out_steps:
                raise NotImplementedError('stride must match out_steps')
            ndx = stride + self.in_steps
            col_ndx = self.y_data.dtype.names.index(column_name)
            prediction = np.concatenate(preds)[:, col_ndx]
            truth = self.y_data[column_name]
        prediction_time = (np.arange(len(prediction)) *
                           self.time_interval).astype('timedelta64[s]')
        truth_time = (np.arange(len(truth)) *
                      self.time_interval).astype('timedelta64[s]')

        if return_data:
            return {'prediction': (prediction_time, prediction),
                    'truth': (truth_time, truth)}
        else:
            plt.title('Test Prediction')
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Value')
            plt.plot(prediction_time, prediction)
            plt.plot(truth_time, truth)
            plt.legend(['prediction', 'truth'])
