# Sin Wave Test Example
This example tests the code by training on inputing and outputing the first half of the same points along a sin wave.

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
stop2 = '1970-01-02T02:00:00Z'

parameters = 'vector'
options = {'logging': True, 'usecache': True, 'cachedir': './hapicache'}
```

## HAPI-NN Testing
Begin testing HAPI-NN


```python
in_steps = 512
out_steps = 128
```

### Create Trainer and Tester as well as load data with HAPI

Create Trainer


```python
splits = (.7, .2, .1)  # Train, Validation, Test
    
trainer = HAPINNTrainer(
    splits, in_steps, out_steps,
    preprocess_func=None,
    preprocess_y_func=None,
    lag=False
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

Model input will come from vector in HAPI dataset and will be the first and second element in the column. The output comes from the third element in the column.
The input is not lagged behind the output, so we are not forecasting the outputs based on the inputs.


```python
data, meta = hapi(server, dataset, parameters, start, stop, **options)
trainer.set_hapidatas([data], xyparameters=[['vector_0', 'vector_1'], ['vector_2']])
```

    hapi(): Running hapi.py version 0.2.4
    hapi(): file directory = ./hapicache/hapi-server.org_servers_TestData2.0_hapi
    hapi(): Reading dataset1_vector_19700101T000000_19700102T000000.pkl
    hapi(): Reading dataset1_vector_19700101T000000_19700102T000000.npy 


    /home/jovyan/hapi_nn.py:198: UserWarning: Time gaps exist in the data.
      warnings.warn('Time gaps exist in the data.')
    /home/jovyan/hapi_nn.py:239: UserWarning: Removed data gab at index 0. Length of gab (10) was too small. Split size (0) is less than minimum step size (512).
      warnings.warn(f'Removed data gab at index {ndx}. '
    /home/jovyan/hapi_nn.py:244: UserWarning: Data points with time gaps that caused too small of splits where removed. Removed 1 out of 2 gaps.
      warnings.warn('Data points with time gaps that caused '





    1.0



Prepare the downloaded data for training


```python
trainer.prepare_data()
```

    /home/jovyan/hapi_nn.py:408: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(data)
    /home/jovyan/hapi_nn.py:419: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      data = np.array(remerge_data)
    /home/jovyan/hapi_nn.py:422: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      y_data = np.array(y_data)
    /home/jovyan/hapi_nn.py:432: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      y_data = np.array(remerge_data)


    0 8
    9 13





    (0.7069609305099911, 0.20199224575007457, 0.09104682373993439)



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
    in_steps, out_steps,
    preprocess_func=None,
    preprocess_y_func=None
)
```

Load data for testing


```python
data, meta = hapi(server, dataset, parameters, start2, stop2, **options)
tester.set_hapidatas([data], xyparameters=[['vector_0', 'vector_1'], ['vector_2']])
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
            self.conv7 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
            self.conv8 = Conv1dSamePadding(hidden_units, hidden_units, 5, stride=2)
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
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.convt1(x)
            x = self.convt2(x)
            x = self.convt3(x)
            x = self.convt4(x)
            x = self.convt5(x)
            x = self.convt6(x)
            x = self.conv(x)
            x = torch.swapaxes(x, 1, 2)

            return x
        
    model = S2S(2, 1, 16)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = 'cpu'
    print(model)
else:    
    x0 = keras.layers.Input(shape=(in_steps, 2))
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x0)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding='same')(x)
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
     input_1 (InputLayer)        [(None, 512, 2)]          0         
                                                                     
     conv1d (Conv1D)             (None, 256, 16)           176       
                                                                     
     conv1d_1 (Conv1D)           (None, 128, 16)           1296      
                                                                     
     conv1d_2 (Conv1D)           (None, 64, 16)            1296      
                                                                     
     conv1d_3 (Conv1D)           (None, 32, 16)            1296      
                                                                     
     conv1d_4 (Conv1D)           (None, 16, 16)            1296      
                                                                     
     conv1d_5 (Conv1D)           (None, 8, 16)             1296      
                                                                     
     conv1d_6 (Conv1D)           (None, 4, 16)             1296      
                                                                     
     conv1d_7 (Conv1D)           (None, 2, 16)             1296      
                                                                     
     conv1d_transpose (Conv1DTra  (None, 4, 16)            1296      
     nspose)                                                         
                                                                     
     conv1d_transpose_1 (Conv1DT  (None, 8, 16)            1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_2 (Conv1DT  (None, 16, 16)           1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_3 (Conv1DT  (None, 32, 16)           1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_4 (Conv1DT  (None, 64, 16)           1296      
     ranspose)                                                       
                                                                     
     conv1d_transpose_5 (Conv1DT  (None, 128, 16)          1296      
     ranspose)                                                       
                                                                     


    2022-07-01 21:52:10.141274: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


     conv1d_8 (Conv1D)           (None, 128, 1)            17        
                                                                     
    =================================================================
    Total params: 17,041
    Trainable params: 17,041
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

    Epoch: 1/1 - Batch: 1774/1774 - Loss: 0.008294 - Validation Loss: 0.000001





    {'train': 4.80292351440698e-06,
     'val': 6.897524106538649e-07,
     'test': 3.447913393842738e-07}




```python
epochs = 1
batch_size = 32

trainer.train(model, epochs, batch_size=batch_size, loss_func=loss_function,
        optimizer=optimizer, device=device)
```

    2022-07-01 21:52:13.491808: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 242733056 exceeds 10% of free system memory.


    1852/1852 [==============================] - 15s 8ms/step - loss: 0.0035 - mae: 0.0127 - val_loss: 0.0026 - val_mae: 0.0418
     37/926 [>.............................] - ETA: 3s - loss: 0.0025 - mae: 0.0410

    2022-07-01 21:52:29.234230: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 242733056 exceeds 10% of free system memory.


    926/926 [==============================] - 4s 4ms/step - loss: 0.0026 - mae: 0.0417
    265/265 [==============================] - 1s 4ms/step - loss: 0.0026 - mae: 0.0418
    120/120 [==============================] - 1s 4ms/step - loss: 0.0026 - mae: 0.0420





    {'train': [0.0025684740394353867, 0.041745200753211975],
     'val': [0.0025709427427500486, 0.04180735722184181],
     'test': [0.002583264373242855, 0.042035769671201706]}



### Test Model

Predict all forecasts over the downloaded testing data using the default stride (out_steps)


```python
predictions = tester.test(model)
```


```python
tester.plot(predictions, 0, 'vector_2')
```


    
![png](Sin%20Wave%20Test%20Example2_files/Sin%20Wave%20Test%20Example2_36_0.png)
    



```python
tester.plot(predictions, -1, 'vector_2')
```


    
![png](Sin%20Wave%20Test%20Example2_files/Sin%20Wave%20Test%20Example2_37_0.png)
    



```python
tester.plot(predictions, -1, 'vector_2', return_data=True)
```




    {'prediction': (array([   0,    1,    2, ..., 3197, 3198, 3199], dtype='timedelta64[s]'),
      array([-1.058659 , -1.0569491, -1.0668932, ...,  0.5308854,  0.5248555,
              0.5166418], dtype=float32)),
     'truth': (array([   0,    1,    2, ..., 3597, 3598, 3599], dtype='timedelta64[s]'),
      array([-1.        , -0.9999863 , -0.99994516, ..., -0.9998766 ,
             -0.99994516, -0.9999863 ], dtype=float32))}


