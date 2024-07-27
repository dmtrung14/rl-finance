class TraderCfg:
    class envs:
        num_envs = 3072

    class market:
        # choose from ["tech", "finance", "energy", "sp500"]
        stock_groups = ["tech", "finance"]
        slip = 0.2 # slip for a single purchase
        partial_exchange = True # allow partial exchange of stocks

    class trader:
        balance = 10000
        max_position = 1000 # max USD value of a single stock to hold
        close = 8000 # terminate if portfolio value is below 8000
    
    class rewards:
        class scales:
            pass
        only_positive_rewards = True
    

class TraderCfgPPO:
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        scan_encoder_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    # TODO: REVIEW
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    #TODO: REVIEW
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 5000 # number of policy updates   4500 orginally

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt