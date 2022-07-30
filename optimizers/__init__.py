from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

#logger = logging.getLogger("ptsemseg")

optim_dict = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(config):
    if config["training"]["optimizer"] is None:
        return SGD

    else:
        opt_name = config["training"]["optimizer"]["name"]
        if opt_name not in optim_dict:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

    return optim_dict[opt_name]
