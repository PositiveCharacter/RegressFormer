from parameter import Parameter
from data.dataloader import WeatherDataLoader


def train():
    # get dataloader
    dataloader = WeatherDataLoader(
        data_path=Parameter.data_path,
        batch_size=Parameter.batch_size,
        num_workers=Parameter.num_workers,
        shuffle_train=Parameter.shuffle_train,
        shuffle_valid=Parameter.shuffle_valid,
        sequence_length=Parameter.sequence_length,
        predict_length=Parameter.forecast_length,
        label_length=Parameter.label_length,
        scale=Parameter.scale,
        features=Parameter.features,
        timeenc=Parameter.timeenc,
        use_time_features=Parameter.use_time_features,
    )
    
    # get model
    model = get_model(dls.vars, args)

    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')

    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name, 
                     path=args.save_path )
        ]

    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )
                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)