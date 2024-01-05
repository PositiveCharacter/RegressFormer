class Parameter:
    data_path = "wind_power"                     # dataset file path
    sequence_length = 24 * 4 * 4                 # sequence length
    forecast_length = 24 * 4                     # forecast horizon
    label_length = 24 * 4                        # decoder label length
    batch_size = 64                              # batch size
    num_workers = 1                              # number of workers for DataLoader
    shuffle_train = True                         # whether shuffle the train dataloader
    shuffle_valid = False                        # whether shuffle the valid dataloader
    scale = 'standard'                           # scale the input data
    features = 'M'                               # for multivariate model or univariate model
    timeenc = 0                                  # the style of time encoding
    use_time_features = 0                        # whether to use time features or not
    
    # Patch
    patch_len = 32                               # patch length
    stride = 16                                  # stride between patch
    
    # RevIN
    revin = 1                                    # reversible instance normalization
    
    # Model args
    layers_number = 3                            # number of Transformer layers
    heads_number = 16                            # number of Transformer heads
    model_input = 128                            # Transformer model input length
    mlp_dimension = 256                          # Tranformer MLP dimension
    dropout = 0.2                                # Transformer dropout
    head_dropout = 0                             # head dropout
    
    # Optimization args
    train_epochs = 20                            # number of training epochs
    learning_rate = 1e-4                         # learning rate
    
    # model id to keep track of the number of models saved
    model_id = 1                                 # id of the saved model
    model_type = 'based_model'                   # for multivariate model or univariate model
    
    # training
    is_train = 1                                 # training the model