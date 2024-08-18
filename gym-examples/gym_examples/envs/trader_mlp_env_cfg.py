from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

def get_trader_env_cfg(argv=None):
    # Parse the default configuration
    parser, cfg = parse_sf_args(argv=argv, evaluation=False)
    cfg = parse_full_cfg(parser, argv)
    
    # Apply custom configuration for your environment
    cfg.env = 'gym_examples/Trader-v0'
    cfg.algo = 'APPO'
    cfg.use_rnn = True
    cfg.num_envs_per_worker = 10
    cfg.policy_workers_per_policy = 2
    cfg.recurrence = -1
    cfg.rollout = 64
    cfg.with_vtrace = False
    cfg.batch_size = 512
    cfg.save_every_sec = 60
    cfg.experiment = 'example_gymnasium_trader'
    
    # Custom parameters for the Trader environment
    cfg.max_episode_length = 1000  # Adjust as necessary
    

    return cfg
