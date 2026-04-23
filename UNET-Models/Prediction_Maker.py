import json
import numpy as np
from tensorflow.keras.models import load_model
import os

base_dir = str(os.path.dirname(os.path.abspath(__file__)))

filter_names = ['F606w', 'F625w', 'F775w', 'F814w', 'F850lp']
Model_Names = ['F606w', 'F625w', 'F775w', 'F814w', 'F850lp','MultiChannel','MultiEncoder']

def split_log(x):
    """
    Split log10 transform — same as used during training.

    Maps:
        x > 0  →  +log10(x)
        x = 0  →   0.0
        x < 0  →  -log10(|x|)
    """
    log_x = np.zeros_like(x, dtype=np.float64)
    pos_mask = x > 0
    neg_mask = x < 0
    log_x[pos_mask] =  np.log10(x[pos_mask])
    log_x[neg_mask] = -np.log10(np.abs(x[neg_mask]))
    return log_x


def Single_Prediction(model_path, luminosity, redshift, filter_type):

    # ── Load standardization metrics
    with open(base_dir+f"/results_{filter_type}/norm_stats.json", 'r') as f:
        norm_stats = json.load(f)

    LUMI_MEAN = norm_stats["lumi_mean"]
    LUMI_STD  = norm_stats["lumi_std"]
    MASS_MEAN = norm_stats["mass_mean"]
    MASS_STD  = norm_stats["mass_std"]

    print(f"Loaded norm stats for {norm_stats['filter']}:")
    print(f"  lumi_mean={LUMI_MEAN:.4f}, lumi_std={LUMI_STD:.4f}")
    print(f"  mass_mean={MASS_MEAN:.4f}, mass_std={MASS_STD:.4f}")

    
    luminosity = np.array(luminosity[filter_type])       
    luminosity = luminosity.reshape((-1, 128, 128))
    log_lum    = split_log(luminosity)
    input_normalized_lum = ((log_lum - LUMI_MEAN) / LUMI_STD).astype(np.float32)
    input_normalized_lum = np.expand_dims(input_normalized_lum, -1)  # (N,128,128,1)

    
    redshift = np.array(redshift).astype(np.float32).reshape((-1,1))

    # ── Load model
    model = load_model(model_path, compile=False)

    # ── Prediction
    inputs = {
        "image_input": input_normalized_lum,
        "z_input": redshift
    }

    mass_pred_standard = model.predict(inputs, verbose=1)

    # Unstandardize
    mass_pred_log = mass_pred_standard * MASS_STD + MASS_MEAN

    # ── Inverse log
    mass_pred = np.power(10, mass_pred_log)

    return mass_pred[:,:,:,0]



def Multi_Channel_Prediction(model_path, luminosity, redshift):

    # ── Load normalization stats
    with open(base_dir+"/results_MultiChannel/norm_stats.json", 'r') as f:
        norm_stats = json.load(f)
        
    MASS_MEAN = norm_stats["mass_mean"]
    MASS_STD  = norm_stats["mass_std"]
    
    filter_keys = list(luminosity.keys())
    N_filters  = len(filter_keys)
    N_samples  = luminosity[filter_keys[0]].shape[0]
    
    
    luminosity_stacked = np.empty((N_samples, 128, 128, N_filters), dtype=np.float64)
    for i, f_i in enumerate(filter_keys):
        luminosity_stacked[:,:,:,i] = luminosity[f_i]
    
   
    log_lum_stacked = split_log(luminosity_stacked)
    
    
    for i, f_i in enumerate(filter_keys):
        LUMI_MEAN = norm_stats['filters'][f_i]["lumi_mean"]
        LUMI_STD  = norm_stats['filters'][f_i]["lumi_std"]
        log_lum_stacked[:,:,:,i] = (log_lum_stacked[:,:,:,i] - LUMI_MEAN) / LUMI_STD
    
    log_lum_stacked = log_lum_stacked.astype(np.float32)
    
    
    redshift = np.array(redshift).astype(np.float32).reshape((-1,1))
    
    
    model = load_model(model_path, compile=False)
    
    inputs = {
        "image_input": log_lum_stacked,
        "z_input": redshift
    }
    

    # Deterministic prediction
    mass_pred_std = model.predict(inputs)
    mass_pred = mass_pred_std * MASS_STD + MASS_MEAN
    mass_pred = np.power(10, mass_pred)
    
    return mass_pred[:,:,:,0]

    
    

def MultiEncoder_Mass_Prediction(model_path, luminosity, redshift):

    with open(base_dir+"/results_MultiEncoder/norm_stats.json", 'r') as f:
        norm_stats = json.load(f)

    MASS_MEAN = norm_stats["mass_mean"]
    MASS_STD  = norm_stats["mass_std"]

    filter_keys = list(luminosity.keys())
    N_samples   = luminosity[filter_keys[0]].shape[0]
    N_filters   = len(filter_keys)

    
    inputs_dict = {}
    for filt in filter_keys:
        log_lum = split_log(luminosity[filt])
        LUMI_MEAN = norm_stats["filters"][filt]["lumi_mean"]
        LUMI_STD  = norm_stats["filters"][filt]["lumi_std"]
        inputs_dict[f"input_{filt}"] = ((log_lum - LUMI_MEAN) / LUMI_STD).astype(np.float32)

    
    inputs_dict["z_input"] = np.array(redshift, dtype=np.float32).reshape((-1,1))

    
    model = load_model(model_path, compile=False)



    mass_pred_std = model.predict(inputs_dict)
    mass_pred = mass_pred_std * MASS_STD + MASS_MEAN
    mass_pred = np.power(10, mass_pred)
    return mass_pred[:,:,:,0]
    
    
    
    
    
    
    
    
    
    
    
    
    
