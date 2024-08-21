from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

def get_trader_cnn_env_cfg(argv=None):
    # Parse the default configuration
    parser, cfg = parse_sf_args(argv=argv, evaluation=False)
    cfg = parse_full_cfg(parser, argv)
    
    # Apply custom configuration for your environment
    cfg.env = 'gym_examples/Trader-v1'
    cfg.algo = 'APPO'
    cfg.use_rnn = True
    
    cfg.num_workers = 8  # 4
    cfg.num_envs_per_worker = 4 # 2
    cfg.policy_workers_per_policy = 4
    
    
    cfg.with_vtrace = False
    cfg.batch_size = 1024
    cfg.save_every_sec = 60
    cfg.experiment = 'example_gymnasium_trader_cnn'
    
    # Custom parameters for the Trader environment
    #cfg.max_episode_length = 1000  # Adjust as necessary
    cfg.learning_rate = 0.00001
    cfg.num_envs_per_worker = 16
    cfg.rollout = 64
    cfg.recurrence = 64
    cfg.gamma = 0.99
    cfg.max_grad_norm = 4.0 # 4.0
    cfg.exploration_loss_coeff = 0.003  # 0.003
    
    cfg.frame_stack = 4
    cfg.nonlinearity = "elu"
    cfg.rnn_type = "lstm"
    cfg.decoder_mlp_layers = [512, 512]
    
    #cfg.kl_loss_coeff = 0.0  # 0.0 Highly recommended for environments with continuous action  spaces. (default: 0.0)
    #cfg.ppo_clip_ratio = 0.2 # 0.1
    #cfg.ppo_clip_value = 1.0 # 1.0
    #cfg.value_loss_coeff =  0.6 # 0.5
    
    #cfg.shuffle_minibatches = False
    

    return cfg
