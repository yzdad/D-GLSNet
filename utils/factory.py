def get_module(path, name):
    """ get module
    Args:
        path: data path
        name: which data set to load
    Return:
        getattr(mod, name): importlib.import_module and name
    """
    print(f"{path}: {name}")
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)


def get_model(name):
    """
    import models
    Args:
        name : the name of models to load
    """
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)  # 


def model_loader(model='SuperPointNet', **options):
    """
    models loader
    """
    print(f"creating models: {model}")
    net = get_model(model)
    net = net(**options)
    return net


def get_loss(name):
    """
    import models
    Args:
        name : the name of models to load
    """
    mod = __import__('loss.{}'.format(name), fromlist=[''])
    return getattr(mod, name)  # 


def loss_loader(model='DetectorLoss', **options):
    """
    models loader
    """
    print(f"creating loss: {model}")
    loss = get_loss(model)
    loss = loss(options)
    return loss
