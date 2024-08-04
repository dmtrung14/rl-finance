class TraderCfg:
    class env:
        num_envs = 5000
        num_obs = 326
        num_privileged_obs = 374
        num_actions = 12 # number of stocks to buy/sell
        
    class market:
        # choose from ["tech", "finance", "energy", "sp500"]
        stock_groups = ["tech"]
        fee = 0.0 # slip for a single purchase
        partial_exchange = True # allow partial exchange of stocks
        start_date = '2013-01-31'
        end_date = '2023-01-01'

        # # uncomment when infering
        start_date = '2015-01-30'
        end_date = '2024-07-01'

    class trader:
        balance = 10000
        max_position = 2000 # max USD value of a single stock to hold
        close = 5000 # terminate if portfolio value is below 8000 trung: making very low to promote learning?
        max_action = 10 # max number of stocks to buy/sell at once

    class rewards:
        class scales:
            termination = 5.0
            profit = 1.0 * 15
            extreme_position = -1e-2

        only_positive_rewards = True
    

class TraderCfgPPO:
    seed = 0
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        # actor_hidden_dims = [256, 128, 64]
        # critic_hidden_dims = [256, 128, 64]
        # actor_hidden_dims = [256, 256, 256]
        # critic_hidden_dims = [256, 256, 256]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    # TODO: REVIEW
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5e-3 #1e-3
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    #TODO: REVIEW
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 10 # per iteration
        max_iterations = 100000 # number of policy updates 4500 orginally

        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        experiment_name = 'trader'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt