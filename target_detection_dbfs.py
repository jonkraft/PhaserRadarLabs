'''
   target_detection_dbfs.py
   Original code from Marshall Bruner, Colorado State University
   https://github.com/brunerm99/ADI_Radar_DSP
   Modified by Jon Kraft to use dBFS values
'''

import numpy as np
from scipy.interpolate import interp1d

def cfar(X_k, num_guard_cells, num_ref_cells, bias=1, cfar_method='average',
    fa_rate=0.2):
    N = X_k.size
    cfar_values = np.ma.masked_all(X_k.shape)
    for center_index in range(num_guard_cells + num_ref_cells, N - (num_guard_cells + num_ref_cells)):
        min_index = center_index - (num_guard_cells + num_ref_cells)
        min_guard = center_index - num_guard_cells 
        max_index = center_index + (num_guard_cells + num_ref_cells) + 1
        max_guard = center_index + num_guard_cells + 1

        lower_nearby = X_k[min_index:min_guard]
        upper_nearby = X_k[max_guard:max_index]

        lower_mean = np.mean(lower_nearby)
        upper_mean = np.mean(upper_nearby)

        if (cfar_method == 'average'):
            mean = np.mean(np.concatenate((lower_nearby, upper_nearby)))
            output = mean + bias
        elif (cfar_method == 'greatest'):
            mean = max(lower_mean, upper_mean)
            output = mean + bias
        elif (cfar_method == 'smallest'):
            mean = min(lower_mean, upper_mean)
            output = mean + bias
        elif (cfar_method == 'false_alarm'):
            refs = np.concatenate((lower_nearby, upper_nearby))
            noise_variance = np.sum(refs**2 / refs.size)
            output = (noise_variance * -2 * np.log(fa_rate))**0.5
        else:
            raise Exception('No CFAR method received')

        cfar_values[center_index] = output

    cfar_values[np.where(cfar_values == np.ma.masked)] = np.min(cfar_values)

    targets_only = np.ma.masked_array(np.copy(X_k))
    targets_only[np.where(abs(X_k) > abs(cfar_values))] = np.ma.masked

    if (cfar_method == 'false_alarm'):
        return cfar_values, targets_only, noise_variance
    else:
        return cfar_values, targets_only
    
    
