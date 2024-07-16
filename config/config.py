import configparser


def read_config(file_name):
    config = configparser.RawConfigParser()
    config.read(file_name)

    config_parms = {}

    config_parms['dataset_dir'] = config.get('I/O parameters', 'dataset_dir')
    config_parms['template_file'] = config.get('I/O parameters', 'template_file')
    config_parms['checkpoint_dir'] = config.get('I/O parameters', 'checkpoint_dir')

    config_parms['n_layers'] = config.getint('model parameters', 'n_layers')
    config_parms['z_length'] = config.getint('model parameters', 'z_length')
    config_parms['down_sampling_factors'] = [int(x) for x in
                                             config.get('model parameters', 'down_sampling_factors').split(',')]
    config_parms['num_features_global'] = [int(x) for x in
                                           config.get('model parameters', 'num_features_global').split(',')]
    config_parms['num_features_local'] = [int(x) for x in
                                          config.get('model parameters', 'num_features_local').split(',')]
    config_parms['batch_norm'] = True if config.getint('model parameters', 'batch_norm') == 1 else False

    config_parms['num_workers'] = config.getint('training parameters', 'num_workers')
    config_parms['lr'] = config.getfloat('training parameters', 'lr')
    config_parms['batch_size'] = config.getint('training parameters', 'batch_size')
    config_parms['weight_decay'] = config.getfloat('training parameters', 'weight_decay')
    config_parms['epoch'] = config.getint('training parameters', 'epoch')
    config_parms["lambda_reg"] = config.getfloat("training parameters", "lambda_reg")

    assert config_parms['n_layers'] == len(
        config_parms['down_sampling_factors']), 'length of down_sampling_factors must equal to n_layers'
    assert config_parms['n_layers'] + 1 == len(
        config_parms['num_features_global']), 'length of num_features must equal to n_layers + 1'

    return config_parms
