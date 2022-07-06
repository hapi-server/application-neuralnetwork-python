# Sin Wave Test Example
This example tests the code by training on inputing and outputing the same points along a sin wave.

## Setup
Import HAPI and other packages


```python
from hapiclient import hapi, hapitime2datetime
from datetime import datetime
from hapiplot import hapiplot
from hapi_nn import HAPINNTrainer, HAPINNTester
import hapi_nn
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
warnings.simplefilter('always')
```

### HAPI Setup
Set HAPI related parameters


```python
hapi_nn.MODEL_ENGINE = 'TORCH'

server = 'http://hapi-server.org/servers/TestData2.0/hapi'
dataset = 'dataset1'
start = '1970-01-01T00:00:00Z'
stop = '1970-01-02T00:00:00Z'

start2 = '1970-01-02T01:00:00Z'
stop2 = '1970-01-02T02:00:00Z'

parameters = 'vector'
options = {'logging': True, 'usecache': True, 'cachedir': './hapicache'}
```

## HAPI-NN Testing
Begin testing HAPI-NN


```python
in_steps = 512
out_steps = 512
```

### Create Trainer and Tester as well as load data with HAPI

Create Trainer


```python
splits = (.5, .05, .45)  # Train, Validation, Test

means = {}
stds = {}
def func(data):
    global means, stds
    for name in data.dtype.names:
        means[name] = data[name].mean()
        stds[name] = data[name].std()
        data[name] = (data[name] - means[name]) / stds[name]
    return data
    
func = HAPINNTrainer.ignore_gaps(func)

# Just testing on_gaps on top of ignore_gaps
# Not needed
func2 = HAPINNTrainer.on_gaps(lambda x: x)
def func3(data):
    data = func(data)
    data = func2(data)
    return data

trainer = HAPINNTrainer(
    splits, in_steps, out_steps,
    preprocess_func=func3,
    preprocess_y_func=func3,
    lag=False
)
```

Load data for Training

Model input will come from vector in HAPI dataset and will be the first element in the column. The output comes from the second and third element in the column.
The input is not lagged behind the output, so we are not forecasting the outputs based on the inputs.


```python
data, meta = hapi(server, dataset, parameters, start, stop, **options)
trainer.set_hapidatas([data], xyparameters=[['vector_0'], ['vector_1', 'vector_2']])
```

    hapi(): Running hapi.py version 0.2.4
    hapi(): file directory = ./hapicache/hapi-server.org_servers_TestData2.0_hapi
    hapi(): Reading dataset1_vector_19700101T000000_19700102T000000.pkl
    hapi(): Reading dataset1_vector_19700101T000000_19700102T000000.npy 


    /home/jovyan/HAPI_NN/hapi_nn.py:209: UserWarning: Time gaps exist in the data.
      warnings.warn('Time gaps exist in the data.')
    /home/jovyan/HAPI_NN/hapi_nn.py:250: UserWarning: Removed data gab at index 0. Length of gab (10) was too small. Split size (0) is less than minimum step size (512).
      warnings.warn(f'Removed data gab at index {ndx}. '
    /home/jovyan/HAPI_NN/hapi_nn.py:255: UserWarning: Data points with time gaps that caused too small of splits where removed. Removed 1 out of 2 gaps.
      warnings.warn('Data points with time gaps that caused '





    1.0



Prepare the downloaded data for training


```python
trainer.prepare_data()
```

    /home/jovyan/HAPI_NN/hapi_nn.py:419: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(data)
    /home/jovyan/HAPI_NN/hapi_nn.py:430: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(remerge_data)
    /home/jovyan/HAPI_NN/hapi_nn.py:433: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      y_data = np.array(y_data)
    /home/jovyan/HAPI_NN/hapi_nn.py:443: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      y_data = np.array(remerge_data)
    /home/jovyan/HAPI_NN/hapi_nn.py:764: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(np.split(data, split_ndxs))
    /home/jovyan/HAPI_NN/hapi_nn.py:779: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      return np.array([func(x) for x in data])
    /home/jovyan/HAPI_NN/hapi_nn.py:764: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(np.split(data, split_ndxs))
    /home/jovyan/HAPI_NN/hapi_nn.py:779: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      return np.array([func(x) for x in data])
    /home/jovyan/HAPI_NN/hapi_nn.py:764: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(np.split(data, split_ndxs))
    /home/jovyan/HAPI_NN/hapi_nn.py:779: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      return np.array([func(x) for x in data])
    /home/jovyan/HAPI_NN/hapi_nn.py:764: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(np.split(data, split_ndxs))
    /home/jovyan/HAPI_NN/hapi_nn.py:779: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      return np.array([func(x) for x in data])


    2 3
    4 5
    7 8
    9 14
    16 17





    (0.5125960744457855, 0.05126087785047322, 0.43614304770374135)



Test saving and loading


```python
trainer.save_prepared_data('data')
```


```python
trainer.load_prepared_data('./data')
```

Create Tester


```python
# Tester does not have any gaps
def func(data):
    global means, stds
    for name in data.dtype.names:
        data[name] = (data[name] - means[name]) / stds[name]
    return data

# If wanted to undo func for output
def yfunc(data):
    global means, stds
    # currently output is not structured
    data[:, 0] = data[:, 0] *  stds['vector_1'] + means['vector_1']
    data[:, 1] = data[:, 1] *  stds['vector_2'] + means['vector_2']
    return data


tester = HAPINNTester(
    in_steps, out_steps, preprocess_func=func,
    preprocess_y_func=yfunc
)
```

Load data for testing


```python
data, meta = hapi(server, dataset, parameters, start2, stop2, **options)
tester.set_hapidatas([data], xyparameters=[['vector_0'], ['vector_1', 'vector_2']])
```

    hapi(): Running hapi.py version 0.2.4
    hapi(): file directory = ./hapicache/hapi-server.org_servers_TestData2.0_hapi
    hapi(): Reading dataset1_vector_19700102T010000_19700102T020000.pkl
    hapi(): Reading dataset1_vector_19700102T010000_19700102T020000.npy 





    1.0



Prepare data for testing


```python
tester.prepare_data()
```

### Create Models and Train

Import either the modules for PyTorch or TensorFlow


```python
if hapi_nn.MODEL_ENGINE == 'TORCH':
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
else:
    import tensorflow as tf
    from tensorflow import keras
```

Create PyTorch Module or TensorFlow Model


```python
if hapi_nn.MODEL_ENGINE == 'TORCH':
    class Conv1dSamePadding(nn.Conv1d):
        def forward(self, x):
            pad = max(
                (math.ceil(x.size()[-1] / self.stride[0]) - 1) *
                self.stride[0] + (self.kernel_size[0] - 1) + 1 - x.size()[-1], 0
            )
            if pad > 1:
                x = nn.functional.pad(x, (pad // 2, 1 + pad // 2))
            return  self._conv_forward(x, self.weight, self.bias)

    class Conv1dTransposeSamePadding(nn.ConvTranspose1d):
        def forward(self, x):
            pad = max(
                (math.ceil(x.size()[-1] / self.stride[0]) - 1) *
                self.stride[0] + (self.kernel_size[0] - 1) + 1 - x.size()[-1], 0
            )
            x = nn.ConvTranspose1d.forward(self, x)
            return x[:, :, pad // 2:-(1 + pad // 2)]
        
    class S2S(nn.Module):
        def __init__(self, units_in, units_out, hidden_units):
            super().__init__()
            self.units_in = units_in
            self.units_out = units_out
            self.hidden_units = hidden_units

            self.conv1 = Conv1dSamePadding(units_in, hidden_units, 5, stride=2)
            self.conv2 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv3 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv4 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv5 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv6 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt1 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt2 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt3 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt4 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt5 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt6 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv = nn.Conv1d(hidden_units, units_out, 1, stride=1)

        def forward(self, x):
            batch_size = x.shape[0]

            x = torch.swapaxes(x, 1, 2)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.convt1(x)
            x = self.convt2(x)
            x = self.convt3(x)
            x = self.convt4(x)
            x = self.convt5(x)
            x = self.convt6(x)
            x = self.conv(x)
            x = torch.swapaxes(x, 1, 2)

            return x
        
    model = S2S(1, 2, 16)
    loss_function = nn.MSELoss()
    metric_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = 'cpu'
    print(model)
else:
    #x0 = keras.layers.Input(shape=(in_steps, 1))
    #x = keras.layers.LSTM(16)(x0)
    #x = keras.layers.RepeatVector(out_steps)(x)
    #x = keras.layers.LSTM(16, return_sequences=True)(x)
    #x = keras.layers.Conv1D(1, 1, strides=1)(x)
    
    x0 = keras.layers.Input(shape=(in_steps, 1))
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x0)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(2, 1, strides=1, padding='same')(x)

    model = keras.models.Model(inputs=x0, outputs=x)
    model.summary()

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    device = None
    loss_function = None
    optimizer = None
    metric_function = None
```

    S2S(
      (conv1): Conv1dSamePadding(1, 16, kernel_size=(5,), stride=(2,))
      (conv2): Conv1dSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (conv3): Conv1dSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (conv4): Conv1dSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (conv5): Conv1dSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (conv6): Conv1dSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (convt1): Conv1dTransposeSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (convt2): Conv1dTransposeSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (convt3): Conv1dTransposeSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (convt4): Conv1dTransposeSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (convt5): Conv1dTransposeSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (convt6): Conv1dTransposeSamePadding(16, 16, kernel_size=(5,), stride=(2,))
      (conv): Conv1d(16, 2, kernel_size=(1,), stride=(1,))
    )


### Train the model


```python
epochs = 2
batch_size = 32

trainer.train(model, epochs, batch_size=batch_size, loss_func=loss_function,
              metric_func=metric_function, optimizer=optimizer, device=device)
```

    Epoch: 1/2 - Batch: 1261/1261 - 24.4s 19ms/step - Loss: 0.033535 - Metric Loss: 0.065474 - Validation Loss: 0.000018 - Validation Metric Loss: 0.000540
    Epoch: 2/2 - Batch: 1261/1261 - 24.5s 19ms/step - Loss: 0.000308 - Metric Loss: 0.011578 - Validation Loss: 0.000002 - Validation Metric Loss: 0.000238





    {'train': [4.076269602746567e-05, 0.004687965188388145],
     'val': [2.06429439668472e-06, 0.0002379226801114882],
     'test': [0.000150579351758803, 0.007208811088179332]}




```python
epochs = 1
batch_size = 32

trainer.train(model, epochs, batch_size=batch_size, loss_func=loss_function,
              optimizer=optimizer, device=device)
```

    2022-06-29 18:56:10.271048: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 116467712 exceeds 10% of free system memory.
    2022-06-29 18:56:10.381409: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 232935424 exceeds 10% of free system memory.


    1778/1778 [==============================] - 27s 14ms/step - loss: 0.0075 - mae: 0.0288 - val_loss: 4.6224e-04 - val_mae: 0.0176
     16/889 [..............................] - ETA: 6s - loss: 4.5615e-04 - mae: 0.0175

    2022-06-29 18:56:37.885233: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 116467712 exceeds 10% of free system memory.
    2022-06-29 18:56:37.971320: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 232935424 exceeds 10% of free system memory.


    889/889 [==============================] - 7s 7ms/step - loss: 4.5925e-04 - mae: 0.0175
    254/254 [==============================] - 2s 7ms/step - loss: 4.6224e-04 - mae: 0.0176
    127/127 [==============================] - 1s 8ms/step - loss: 4.5555e-04 - mae: 0.0174





    {'train': [0.0004592504119500518, 0.017533820122480392],
     'val': [0.00046223547542467713, 0.017617294564843178],
     'test': [0.00045554927783086896, 0.0174103993922472]}



### Test Model

Predict all forecasts over the downloaded testing data using the default stride (out_steps)


```python
predictions = tester.test(model)
```


```python
tester.plot(predictions, 0, 'vector_1')
```


    
![png](Sin%20Wave%20Test%20Example_files/Sin%20Wave%20Test%20Example_36_0.png)
    



```python
tester.plot(predictions, -1, 'vector_1')
```


    
![png](Sin%20Wave%20Test%20Example_files/Sin%20Wave%20Test%20Example_37_0.png)
    



```python
tester.plot(predictions, -1, 'vector_2')
```


    
![png](Sin%20Wave%20Test%20Example_files/Sin%20Wave%20Test%20Example_38_0.png)
    



```python
tester.plot(predictions, -1, 'vector_2', return_data=True)
```




    {'prediction': (array([0.000e+00, 1.000e+00, 2.000e+00, ..., 3.581e+03, 3.582e+03,
             3.583e+03]),
      array([-0.95717376, -0.9783135 , -1.0007764 , ..., -1.0132554 ,
             -1.0266743 , -0.9949336 ], dtype=float32)),
     'truth': (array([0.000e+00, 1.000e+00, 2.000e+00, ..., 3.597e+03, 3.598e+03,
             3.599e+03]),
      array([-1.       , -0.9999863, -0.9999451, ..., -0.9998766, -0.9999451,
             -0.9999863], dtype=float32))}




```python

```
