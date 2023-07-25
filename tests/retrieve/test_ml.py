import numpy as np

import pyart


def test_detect_ml():
    rhi = pyart.io.read_cfradial(pyart.testing.RHI_ML_FILE,
                                    field_names=['Zh', 'Rhohv'])
    rhi.sweep_mode['data' ]= np.array(['rhi'])
    # Remove

    refl_field = 'reflectivity'
    rhohv_field = 'uncorrected_cross_correlation_ratio'

    ml = pyart.retrieve.detect_ml(rhi, refl_field=refl_field,
                                rhohv_field=rhohv_field,
                                interp_holes=False,
                                max_length_holes=250)

    assert np.nanmean(ml[0].fields['melting_layer_height']['data']) > 2000
    assert np.nanmean(ml[0].fields['melting_layer_height']['data']) < 3000
