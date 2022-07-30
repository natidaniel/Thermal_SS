import segmentation_models_pytorch as smp

def get_model(model_dict):
    '''
    Parameters:
        -   model_dict: a dictionary containing the model parameters from the config file
    '''
    name = model_dict["arch"]
    
    if name == "smp_deeplabv3":
        n_channels = model_dict["n_channels"]
        n_classes = model_dict["n_classes"]

        model = smp.DeepLabV3(encoder_name="resnet34",
                         encoder_depth=5,
                         encoder_weights=None,
                         in_channels=n_channels,
                         classes=n_classes,
                         activation=None,
                         aux_params=None)
    else:
        model = None
    return model