def get_config_model_name(dense_model):
    """
    Returns the model name for the given dense model.
    """
    if dense_model == 'hubert':
        return "hubert-base-ls960"
    else:  # cpc
        return "cpc-big-ll6k"

def get_model_units_seconds(dense_model):
    """
    Returns the duration of one unit in seconds for the given dense model.
    """
    if dense_model == "hubert" or dense_model == 'mfcc':
        return 0.02
    else:  # cpc
        return 0.01

# define the sample rate
SR = 16_000
