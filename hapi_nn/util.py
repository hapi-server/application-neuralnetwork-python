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


# If not on time axis, push to metdata or force to time
# TODO: RETURN metadata (time range, data name, etc.)
def pyspedas_plotdata_to_hapidata(pyspedas_plotdata):
    """Converts a PySpedas variable to HAPI Data

    Args:
        pyspedas_plotdata: A PySpedas Variable with time.
    Returns:
        A structured numpy array following HAPI data format.
    """
    time = np.array(
        [datetime.fromtimestamp(time).isoformat('T', 'milliseconds') + 'Z'
         for time in pyspedas_plotdata.times], dtype=[('Time', 'S24')]
    )
    columns = [time]
    for ndx, field in enumerate(pyspedas_plotdata._fields[1:]):
        if pyspedas_plotdata[ndx + 1].shape[0] == time.shape[0]:
            if pyspedas_plotdata[ndx + 1].ndim == 1:
                columns.append(
                    pyspedas_plotdata[ndx + 1].astype(
                        [(field, pyspedas_plotdata[ndx + 1].dtype)])
                )
            else:
                shape = pyspedas_plotdata[ndx + 1].shape[1:]
                columns.append(
                    np.array(
                        [(e,)
                         for e in np.reshape(
                             pyspedas_plotdata[ndx + 1],
                             (-1, np.prod(shape)))],
                        [(field, (pyspedas_plotdata[ndx + 1].dtype, shape))])
                )
        else:
            warnings.warn(f'{field} does not have the same time axis')
    return rf.merge_arrays(columns, flatten=True)


def extract_format_structured_data(data, parameters):
    """Extracts elements/columns out of structured data.
       A helper function.

    Args:
        data: A numpy structured array.
        parameters: A list of strings, which specific the columns and
                    subelements.
    Returns:
        A new structured array with only the specified columns and subelements.
    """
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
