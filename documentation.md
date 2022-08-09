<a id="hapi_nn"></a>

# hapi\_nn

Author: Travis Hammond

©️ 2022 The Johns Hopkins University Applied Physics Laboratory LLC.

<a id="hapi_nn.training"></a>

# hapi\_nn.training

Author: Travis Hammond

©️ 2022 The Johns Hopkins University Applied Physics Laboratory LLC.

<a id="hapi_nn.training.HAPINNTrainer"></a>

## HAPINNTrainer Objects

```python
class HAPINNTrainer()
```

A class for preparing HAPI data and putting it in
a form that can be used to train time series neural networks.

<a id="hapi_nn.training.HAPINNTrainer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(data_split,
             in_steps,
             out_steps,
             preprocess_func=None,
             preprocess_y_func=None,
             lag=True)
```

Initalizes PyTorch or Tensorflow Modules as well
as other parameters.

**Arguments**:

- `data_split` - A list or tuple of three values that sum to 1,
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
- `in_steps` - An integer, which represents the number of
  data points from the time series data to
  include as input into a model
- `out_steps` - An integer, which represents the number of
  data points the model should output
- `preprocess_func` - A function that is called on the
  different input data splits, which
  should be used to normalize the data,
  fix invalid values, and other processing
  before the data is more heavely formatted
  for training.
- `preprocess_y_func` - A function that is called on the
  different output data splits, which
  should be used to normalize the data,
  fix invalid values, and other processing
  before the data is more heavely formatted
  for training.
- `lag` - A boolean, which determines if the input should lag one
  timestep behind the expected output. Defaults to True.
  True implies the model is used for forecasting.

<a id="hapi_nn.training.HAPINNTrainer.set_hapidatas"></a>

#### set\_hapidatas

```python
def set_hapidatas(datas, xyparameters=None)
```

Gives the Trainer the HAPI data and checks
that the datas can be used for training.

**Arguments**:

- `datas` - A list or tuple of hapi data, which
  has same data columns and same intervals.
  The datas should not have same columns besides
  for time. datas can have time gaps, however,
  the gaps must be the same across all datas.
- `xyparameters` - A list or tuple of lists which contain
  the column names that indicate the
  wanted columns in x input and y input,
  respectively.

**Returns**:

  A number that represents the time interval between data points.

<a id="hapi_nn.training.HAPINNTrainer.prepare_data"></a>

#### prepare\_data

```python
def prepare_data()
```

Prepares the data for training by
processing it and partitioning it.

**Returns**:

  A list of the actual resulting partition proportions is returned.

<a id="hapi_nn.training.HAPINNTrainer.save_prepared_data"></a>

#### save\_prepared\_data

```python
def save_prepared_data(path)
```

Saves the data that is prepared for training.
The data will be saved as npy files and each file
represents a value in the processed_data dict.

**Arguments**:

- `path` - A string that has the path and prefix of the
  files that will be saved.

<a id="hapi_nn.training.HAPINNTrainer.load_prepared_data"></a>

#### load\_prepared\_data

```python
def load_prepared_data(path)
```

Loads the data that was prepared for training.

**Arguments**:

- `path` - A string that has the path and prefix of the
  files that will be loaded.

<a id="hapi_nn.training.HAPINNTrainer.get_prepared_data"></a>

#### get\_prepared\_data

```python
def get_prepared_data()
```

Returns the processed data that is prepared for training.

**Returns**:

  The processed data if it is prepared and partitioned else
  it will return None

<a id="hapi_nn.training.HAPINNTrainer.process_data"></a>

#### process\_data

```python
def process_data()
```

Processes the data that was set with set_hapidatas method.
This method also preforms part of the test partitioning.

<a id="hapi_nn.training.HAPINNTrainer.partition_data"></a>

#### partition\_data

```python
def partition_data()
```

Partitions the data that was processed by process_data method.

**Returns**:

  A list of the actual resulting partition proportions is returned.

<a id="hapi_nn.training.HAPINNTrainer.torch_train"></a>

#### torch\_train

```python
def torch_train(model,
                loss_func,
                optimizer,
                epochs,
                device,
                batch_size=None,
                metric_func=None,
                verbose=1)
```

Trains and evaluates a torch model.

**Arguments**:

- `model` - A PyTorch Module.
- `loss_func` - A torch loss function.
- `optimizer` - A torch optimizer.
- `epochs` - An integer, the number of training epochs.
- `device` - A string, the device to use gpu/cpu etc.
- `batch_size` - An integer, the size of each batch for training.
- `metric_func` - A torch loss/metric function.
- `verbose` - An integer, rates verbosity. 0 None, 1 All.

**Returns**:

  A dict of results for train, validation, and test.

<a id="hapi_nn.training.HAPINNTrainer.tf_train"></a>

#### tf\_train

```python
def tf_train(model, epochs, batch_size=None, **kwargs)
```

Trains and evaluates a tensorflow model.

**Arguments**:

- `model` - A TensorFLow/Keras Model.
- `epochs` - An integer, the number of training epochs.
- `device` - A string, the device to use gpu/cpu etc.
- `kwargs` - Keyword arguments for Keras fit model method.

**Returns**:

  A dict of results for train, validation, and test.

<a id="hapi_nn.training.HAPINNTrainer.train"></a>

#### train

```python
def train(model,
          epochs,
          batch_size=None,
          loss_func=None,
          metric_func=None,
          optimizer=None,
          device=None,
          verbose=1)
```

Trains and evaluates a tensorflow or torch model.

**Arguments**:

- `model` - A PyTorch/TensorFlow Model.
- `epochs` - An integer, the number of training epochs.
- `batch_size` - An integer, the size of each batch for training.
- `loss_func` - A torch loss function.
- `metric_func` - A torch loss/metric function.
- `optimizer` - A torch optimizer.
- `device` - A string, the device to use gpu/cpu etc.
- `verbose` - An integer, rates verbosity. 0 None, 1 All.

**Returns**:

  A dict of results for train, validation, and test.

<a id="hapi_nn.training.HAPINNTrainer.ignore_gaps"></a>

#### ignore\_gaps

```python
@staticmethod
def ignore_gaps(func)
```

Wraps a preprocess function to ignore gaps.
Useful when not accessing neighbor elements.

**Arguments**:

- `func` - A preprocess function that handles structured data

**Returns**:

  A wrapped preprocess function

<a id="hapi_nn.training.HAPINNTrainer.on_gaps"></a>

#### on\_gaps

```python
@staticmethod
def on_gaps(func)
```

Wraps a preprocess function to apply itself on every array
that was split because of gaps.
Useful when accessing neighbor elements.

**Arguments**:

- `func` - A preprocess function that handles structured data

**Returns**:

  A wrapped preprocess function

<a id="hapi_nn.testing"></a>

# hapi\_nn.testing

Author: Travis Hammond

©️ 2022 The Johns Hopkins University Applied Physics Laboratory LLC.

<a id="hapi_nn.testing.HAPINNTester"></a>

## HAPINNTester Objects

```python
class HAPINNTester()
```

<a id="hapi_nn.testing.HAPINNTester.__init__"></a>

#### \_\_init\_\_

```python
def __init__(in_steps,
             out_steps,
             preprocess_func=None,
             preprocess_y_func=None)
```

Initalizes PyTorch or Tensorflow Modules as well
as other parameters.

**Arguments**:

- `in_steps` - An integer, which represents the number of
  data points from the time series data to
  include as input into a model
- `preprocess_func` - A function that is called on
  the inputs to the model, which
  should be used to normalize the data,
  fix invalid values, and other processing
- `preprocess_y_func` - A function that is called on the
  outputs of the model, which
  should be used to denormalize the data
  etc.

<a id="hapi_nn.testing.HAPINNTester.set_hapidatas"></a>

#### set\_hapidatas

```python
def set_hapidatas(datas, xyparameters=None)
```

Gives the Trainer the HAPI data and checks
that the datas can be used for training.

**Arguments**:

- `datas` - A list or tuple of hapi data, which
  has same data columns and same intervals.
  The datas should not have same columns besides
  for time. Note, datas cannot have time gaps.
- `xyparameters` - A list or tuple of lists which contain
  the column names that indicate the
  wanted columns in x input and y input,
  respectively.

**Returns**:

  A number that represents the time interval between data points.

<a id="hapi_nn.testing.HAPINNTester.prepare_data"></a>

#### prepare\_data

```python
def prepare_data()
```

Prepares the data for testing by
processing it with preprocessing and
reformatting.

<a id="hapi_nn.testing.HAPINNTester.test"></a>

#### test

```python
def test(model, stride=None)
```

Tests the model by giving it all inputs
gathered with the provided stride.
Useful for then plotting the outputs and
comparing them to the expected.

**Arguments**:

- `model` - A PyTorch or TensorFlow model
- `stride` - An integer, which defaults to out_steps

**Returns**:

  A list of predictions.

<a id="hapi_nn.testing.HAPINNTester.forecast_plot"></a>

#### forecast\_plot

```python
def forecast_plot(preds,
                  pred_ndx,
                  column_name,
                  stride=None,
                  return_data=False)
```

Creates a line plot to compare the ground truth and predicted
values of a specific output column for one of the predicted outputs.
Used when Trainer had lag=True.

**Arguments**:

- `preds` - A list of predictions from test method.
- `pred_ndx` - An index for indexing the preds. If -1, displays
  results for all preds.
- `column_name` - A string which is a name of a output column.
- `stride` - A integer, which is the stride used in test method to
  produce the preds.
- `return_data` - A boolean, which if True will return data for plotting
  instead of plotting itself.

**Returns**:

  None or if return_data is True, returns a dict of forecast and
  truth which both are tuple of (Time, Values)

<a id="hapi_nn.testing.HAPINNTester.plot"></a>

#### plot

```python
def plot(preds, pred_ndx, column_name, stride=None, return_data=False)
```

Creates a line plot to compare the ground truth and predicted
values of a specific output column for one of the predicted outputs.
Used when Trainer had lag=False.

**Arguments**:

- `preds` - A list of predictions from test method.
- `pred_ndx` - An index for indexing the preds. If -1, displays
  results for all preds.
- `column_name` - A string which is a name of a output column.
- `stride` - A integer, which is the stride used in test method to
  produce the preds.
- `return_data` - A boolean, which if True will return data for plotting
  instead of plotting itself.

**Returns**:

  None or if return_data is True, returns a dict of prediction
  and truth which both are tuple of (Time, Values)

<a id="hapi_nn.config"></a>

# hapi\_nn.config

Author: Travis Hammond

©️ 2022 The Johns Hopkins University Applied Physics Laboratory LLC.

<a id="hapi_nn.util"></a>

# hapi\_nn.util

Author: Travis Hammond

©️ 2022 The Johns Hopkins University Applied Physics Laboratory LLC.

<a id="hapi_nn.util.pyspedas_plotdata_to_hapidata"></a>

#### pyspedas\_plotdata\_to\_hapidata

```python
def pyspedas_plotdata_to_hapidata(pyspedas_plotdata)
```

Converts a PySpedas variable to HAPI Data

**Arguments**:

- `pyspedas_plotdata` - A PySpedas Variable with time.

**Returns**:

  A structured numpy array following HAPI data format.

<a id="hapi_nn.util.extract_format_structured_data"></a>

#### extract\_format\_structured\_data

```python
def extract_format_structured_data(data, parameters)
```

Extracts elements/columns out of structured data.
A helper function.

**Arguments**:

- `data` - A numpy structured array.
- `parameters` - A list of strings, which specific the columns and
  subelements.

**Returns**:

  A new structured array with only the specified columns and subelements.

