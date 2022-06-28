# Sin Wave Test Example
This example tests the code by training on inputing and outputing the future points along a sin wave.

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
hapi_nn.MODEL_ENGINE = 'TENSORFLOW'

server = 'http://hapi-server.org/servers/TestData2.0/hapi'
dataset = 'dataset1'
start = '1970-01-01T00:00:00Z'
stop = '1970-01-02T00:00:00Z'

start2 = '1970-01-02T01:00:00Z'
stop2 = '1970-01-02T04:00:00Z'

parameters = 'scalar'
y_parameters = 'scalar'
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
splits = (.7, .2, .1)  # Train, Validation, Test
    
trainer = HAPINNTrainer(
    splits, in_steps, out_steps,
    preprocess_func=None,
    lag=True
)
```

    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/flatbuffers/compat.py:19: DeprecationWarning: the imp module is deprecated in favour of importlib and slated for removal in Python 3.12; see the module's documentation for alternative uses
      import imp
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:23: DeprecationWarning: NEAREST is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.NEAREST or Dither.NONE instead.
      'nearest': pil_image.NEAREST,
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:24: DeprecationWarning: BILINEAR is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BILINEAR instead.
      'bilinear': pil_image.BILINEAR,
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:25: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
      'bicubic': pil_image.BICUBIC,
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:28: DeprecationWarning: HAMMING is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.HAMMING instead.
      if hasattr(pil_image, 'HAMMING'):
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:29: DeprecationWarning: HAMMING is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.HAMMING instead.
      _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:30: DeprecationWarning: BOX is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BOX instead.
      if hasattr(pil_image, 'BOX'):
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:31: DeprecationWarning: BOX is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BOX instead.
      _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:33: DeprecationWarning: LANCZOS is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.LANCZOS instead.
      if hasattr(pil_image, 'LANCZOS'):
    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras_preprocessing/image/utils.py:34: DeprecationWarning: LANCZOS is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.LANCZOS instead.
      _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


Load data for Training


```python
data, meta = hapi(server, dataset, parameters, start, stop, **options)
trainer.set_hapidatas([data], xyparameters=[parameters.split(','), y_parameters.split(',')])
```

    hapi(): Running hapi.py version 0.2.4
    hapi(): file directory = ./hapicache/hapi-server.org_servers_TestData2.0_hapi
    hapi(): Reading dataset1_scalar_19700101T000000_19700102T000000.pkl
    hapi(): Reading dataset1_scalar_19700101T000000_19700102T000000.npy 


    /home/jovyan/hapi_nn.py:122: UserWarning: Time gaps exist in the data.
      warnings.warn('Time gaps exist in the data.')
    /home/jovyan/hapi_nn.py:173: UserWarning: Removed data gab at index 0. Length of gab (10) was too small. Split size (1) is less than minimum step size (1024).
      warnings.warn(f'Removed data gab at index {ndx}. '
    /home/jovyan/hapi_nn.py:178: UserWarning: Data points with time gaps that caused too small of splits where removed. Removed 1 out of 2 gaps.
      warnings.warn('Data points with time gaps that caused '


Prepare the downloaded data for training


```python
trainer.prepare_data()
```

    /home/jovyan/hapi_nn.py:348: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(data)





    (0.6990676296782666, 0.19973736047275115, 0.10119500984898228)



Test saving and loading


```python
trainer.save_prepared_data('data')
```


```python
trainer.load_prepared_data('./data')
```

Create Tester


```python
tester = HAPINNTester(
    in_steps, out_steps, preprocess_func=None
)
```

Load data for testing


```python
data, meta = hapi(server, dataset, parameters, start2, stop2, **options)
tester.set_hapidatas([data], xyparameters=[parameters.split(','), y_parameters.split(',')])
```

    hapi(): Running hapi.py version 0.2.4
    hapi(): file directory = ./hapicache/hapi-server.org_servers_TestData2.0_hapi
    hapi(): Reading dataset1_scalar_19700102T010000_19700102T040000.pkl
    hapi(): Reading dataset1_scalar_19700102T010000_19700102T040000.npy 


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
        def __init__(self, units, hidden_units):
            super().__init__()
            self.units = units
            self.hidden_units = hidden_units

            self.conv1 = Conv1dSamePadding(units, hidden_units, 5, stride=2)
            self.conv2 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv3 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv4 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv5 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv6 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv7 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv8 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv9 = nn.Conv1d(hidden_units, hidden_units, 2, stride=2)
            self.convt1 = nn.ConvTranspose1d(hidden_units, hidden_units, 2, stride=2)
            self.convt2 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt3 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt4 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt5 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt6 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt7 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt8 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.convt9 = Conv1dTransposeSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv = nn.Conv1d(hidden_units, units, 1, stride=1)

        def forward(self, x):
            batch_size = x.shape[0]

            x = torch.swapaxes(x, 1, 2)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.convt1(x)
            x = self.convt2(x)
            x = self.convt3(x)
            x = self.convt4(x)
            x = self.convt5(x)
            x = self.convt6(x)
            x = self.convt7(x)
            x = self.convt8(x)
            x = self.convt9(x)
            x = self.conv(x)
            x = torch.swapaxes(x, 1, 2)

            return x
        
    model = S2S(1, 16)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = 'cpu'
    print(model)
else:
    x0 = keras.layers.Input(shape=(in_steps, 1))
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x0)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 2, strides=2)(x)
    x = keras.layers.Conv1DTranspose(16, 2, strides=2)(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(1, 1, strides=1, padding='same')(x)

    model = keras.models.Model(inputs=x0, outputs=x)
    model.summary()

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    device = None
    loss_function = None
    optimizer = None
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 512, 1)]          0         
                                                                     
     conv1d (Conv1D)             (None, 256, 16)           96        
                                                                     
     conv1d_1 (Conv1D)           (None, 128, 16)           1296      
                                                                     
     conv1d_2 (Conv1D)           (None, 64, 16)            1296      
                                                                     


    2022-06-28 17:54:01.664895: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


     conv1d_3 (Conv1D)           (None, 32, 16)            1296      
                                                                     
     conv1d_4 (Conv1D)           (None, 16, 16)            1296      
                                                                     
     conv1d_5 (Conv1D)           (None, 8, 16)             1296      
                                                                     
     conv1d_6 (Conv1D)           (None, 4, 16)             1296      
                                                                     
     conv1d_7 (Conv1D)           (None, 2, 16)             1296      
                                                                     
     conv1d_8 (Conv1D)           (None, 1, 16)             528       
                                                                     
     conv1d_transpose (Conv1DTra  (None, 2, 16)            528       
     nspose)                                                         
                                                                     
     conv1d_transpose_1 (Conv1DT  (None, 4, 16)            1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_2 (Conv1DT  (None, 8, 16)            1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_3 (Conv1DT  (None, 16, 16)           1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_4 (Conv1DT  (None, 32, 16)           1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_5 (Conv1DT  (None, 64, 16)           1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_6 (Conv1DT  (None, 128, 16)          1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_7 (Conv1DT  (None, 256, 16)          1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_8 (Conv1DT  (None, 512, 16)          1296      
     ranspose)                                                       
                                                                     
     conv1d_9 (Conv1D)           (None, 512, 1)            17        
                                                                     
    =================================================================
    Total params: 20,609
    Trainable params: 20,609
    Non-trainable params: 0
    _________________________________________________________________


    /home/jovyan/users_conda_envs/HAPINN/lib/python3.10/site-packages/keras/optimizer_v2/adam.py:105: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(Adam, self).__init__(name, **kwargs)


### Train the model


```python
epochs = 1
batch_size = 32

trainer.train(model, epochs, batch_size=batch_size, loss_func=loss_function,
        optimizer=optimizer, device=device)
```

    Epoch: 1/1 - Batch: 1663/1663 - Loss: 0.015476 - Validation Loss: 0.000001





    {'train': 1.0465006275614115e-05,
     'val': 1.499082845697245e-06,
     'test': 7.614717822191789e-07}




```python
epochs = 1
batch_size = 32

trainer.train(model, epochs, batch_size=batch_size, loss_func=loss_function,
        optimizer=optimizer, device=device)
```

    2022-06-28 17:54:05.314029: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 109023232 exceeds 10% of free system memory.
    2022-06-28 17:54:05.394877: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 109023232 exceeds 10% of free system memory.


    1664/1664 [==============================] - 22s 13ms/step - loss: 0.0045 - mae: 0.0176 - val_loss: 1.0163e-05 - val_mae: 0.0025


    2022-06-28 17:54:27.741858: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 109023232 exceeds 10% of free system memory.
    2022-06-28 17:54:27.828447: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 109023232 exceeds 10% of free system memory.





    {'train': [1.0166485481022391e-05, 0.0024825772270560265],
     'val': [1.0162631042476278e-05, 0.0024816100485622883],
     'test': [1.01699270089739e-05, 0.002484126714989543]}



### Test Model

Predict all forecasts over the downloaded testing data using the default stride (out_steps)


```python
predictions = tester.test(model)
```


```python
tester.forecast_plot(predictions, 0, 'scalar')
```


    
![png](Sin%20Wave%20Test%20Example_files/Sin%20Wave%20Test%20Example3_35_0.png)
    



```python
tester.forecast_plot(predictions, -1, 'scalar')
```


    
![png](Sin%20Wave%20Test%20Example_files/Sin%20Wave%20Test%20Example3_36_0.png)
    



```python

```
