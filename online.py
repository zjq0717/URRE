import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import utils
import TD3_PEX
import TD3_BC
import torch.utils.tensorboard as thboard
import datetime
from torch.utils.tensorboard import SummaryWriter
from metaworld_env.metaworld_env import make_metaworld_env
import imageio
from pathlib import Path
from network import LatentModel_new
from network import LatentModel
mean = 0
std = 1
log_dir = './new_logs/pex/'


# mean = np.array([-0.11002816, 0.15680763, 0.10378725, 0.14687687, 0.07839588, -0.20106335,
#                  -0.08224171, -0.2802395, 4.463403, -0.07580097, -0.09260748, 0.41871706,
#                  -0.41171676, 0.11628567, -0.06000552, -0.09738238, -0.14540626])
# std = np.array([0.10956703, 0.6119863, 0.49235544, 0.44962165, 0.39817896, 0.4823394,
#                 0.30695462, 0.26474255, 1.9024047, 0.939795, 1.625154, 14.427593,
#                 11.996738, 11.985555, 12.159913, 8.127248, 6.419199])


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, z, seed_offset=100, eval_episodes=10,):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)
    z = z[0, :]
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = torchify(state)
        state = torch.cat((state, z), dim=-1)
        while not done:
            state = (np.array(state.cpu().detach()).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            # print(action.shape)
            action = action[:action_dim]
            if len(action.shape) > 1:
                action = action.squeeze()[0]  # evaluate the first actor
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            state = torchify(state)
            state = torch.cat((state, z), dim=-1)

    avg_reward /= eval_episodes
    # d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    normalized_returns = d4rl.get_normalized_score(env_name, avg_reward) * 100.0
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {normalized_returns:.3f}")
    print("---------------------------------------")
    return avg_reward, normalized_returns
def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device='cuda')
    return x

def create_env(env_name):
    if "metaworld" in env_name:
        env_id = int(env_name[-1])
        env = make_metaworld_env(env_id)
    else:
        env = gym.make(env_name)
    return env


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--comment", default="")  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    # parser.add_argument("--env", default="halfcheetah-medium-v0")  # OpenAI gym environment name
    parser.add_argument("--env", default="hopper-expert-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--alpha", default=10)  # PEX policy temperature
    parser.add_argument("--mix_ratio", default=0.9, type=float)  # OpenAI gym environment name
    parser.add_argument('--np_ckpt_path',
                        default='',
                        help='path to the offline checkpoint')
    parser.add_argument("--critic_tau", default=0.01)  # Target network update rate
    parser.add_argument("--critic_target_freq", default=0.01)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--grad_clip", default=1, type=float)  # Range to clip gradient in representation layer
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_actor", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_guidance",
                        default="bullet-walker2d-medium-v0")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--guidance_type", default="TD3_BC")
    parser.add_argument("--load_critic", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--omega", default=1., type=float)  # omega for relaxation methods
    parser.add_argument("--reward_dim", default=10, type=int)
    parser.add_argument("--filter_prob", default=0.05)
    parser.add_argument("--lamda", default=0.05)
    args = parser.parse_args()

    file_name = f"finetune_{args.policy}_{args.env}_{args.comment}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists(f"{log_dir}"):
        os.makedirs(f"{log_dir}")
    # env = gym.make(args.env)
    env = create_env(args.env)
    # env = utils.RandomDimWrapper(env, noisy_dim)
    env = utils.MaxStepWrapper(env)
    # mean, std = utils.GAME_MEAN[args.env], utils.GAME_STD[args.env]
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]



    action_dim = env.action_space.shape[0]
    state_dim_cat = env.observation_space.shape[0] + action_dim
    max_action = float(env.action_space.high[0])

    latent_dim = action_dim
    input_x = state_dim + action_dim
    input_y = 1

    kwargs = {
        "state_dim_oracle": state_dim,
        "state_dim": state_dim_cat,
        "action_dim": action_dim,
        "reward_dim": 1,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha,
        "pcgrad": False,
        "filter_prob": args.filter_prob,
        "lamda": args.lamda

    }

    #
    replay_buffer_offline = utils.ReplayBuffer(state_dim, action_dim, 1, online_buffer=True)
    replay_buffer_offline.convert_D4RL(d4rl.qlearning_dataset(env), 1)



    policy = TD3_BC.TD3_BC(**kwargs)


    # Initialize writer
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_dir = Path(f"{log_dir}_{args.env}_{str(args.mix_ratio)}_{args.seed}_online")
    summary_dir.mkdir(parents=True, exist_ok=True)

    filename_suffix = f"{args.env}_{current_time}"
    writer = thboard.SummaryWriter(str(summary_dir), filename_suffix=filename_suffix)


    if args.load_actor != "" or args.load_critic != "":
        if args.load_actor != "":
            policy.load_actor(f"./models/{args.load_actor}")
        if args.load_critic:
            policy.load_critic(f"./models/{args.load_critic}")
    elif args.load_model != "":
        policy_file = args.load_model
        policy.load(f"./models/{policy_file}")
    else:
        print("Warning: finetune must specify pretrained model!")
    # load model
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim,online_buffer=True)
    mean, std = 0, 1
    IN_replay_buffer = utils.ReplayBuffer(state_dim, action_dim,online_buffer=True)

    model = LatentModel_new(input_x, input_y, latent_dim, hidden_dim=512).to("cuda")
    checkpoint = torch.load(args.np_ckpt_path)
    model.load_state_dict(checkpoint)  # 加载参数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



    state, done = env.reset(), False

    # state = (np.array(state).reshape(1, -1) - mean) / std
    state = np.array(state).reshape(1, -1)

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0



    state1, action, next_state, reward, not_done = replay_buffer_offline.sample(64)
    state_target = torch.cat((state1, action), dim=-1)

    state_OFF, actionOFF, _, _, _ = replay_buffer_offline.sample(5000)

    all_states_OFF = []
    sa_OFF = torch.cat((state_OFF, actionOFF), dim=-1)
    sa_OFF = sa_OFF.cpu()
    all_states_OFF.append(sa_OFF)

    all_states_OFF = np.vstack(all_states_OFF)
    # np.save('./sa-tsne/ant_OFF_sa_tsne1.npy', all_states_OFF)



    pre_r, mean_C, std_C, mean_T, std_T, z = model(state_target, reward, state_target, reward)



    flag = 0
    sorce_at_25_percent =0
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:

            state_z = torch.tensor(state).cuda()

            state_z = torch.cat([state_z, z], dim=-1).cpu()

            action = (
                    policy.select_action(np.array(state_z.detach()))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        next_state = (np.array(next_state).reshape(1, -1) - mean) / std
        done_bool = float(done) if episode_timesteps < env.max_episode_steps else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)
        if t >= args.start_timesteps+1:
            state_tensor = torch.from_numpy(state).float()
            action_tensor = torch.from_numpy(action).float()
            action_tensor = action_tensor.unsqueeze(0)
            ex_sa = torch.cat([state_tensor, action_tensor], 1).cuda()

            sorce = policy.Discriminator(ex_sa)

            if sorce > sorce_at_25_percent:
                IN_replay_buffer.add(state, action, next_state, reward, done_bool)

        # Store data in replay buffer


        state = next_state
        episode_reward += reward

        all_states111 = []
        if t >= args.start_timesteps and flag == 0 :
            print("context set--------------------------------")
            flag = 1
            # Assuming replay_buffer has an attribute 'observations' that stores the data
            # 从 replay_buffer 中获取样本
            state_D, action_D, next_state_D, reward_D, done_bool_D = replay_buffer.sample(20000)
            # print(state_D.size())
            # print(action_D.size())
            state_D_simple = state_D
            action_D_simple = action_D
            # 合并 state 和 action
            sa_D = torch.cat([state_D, action_D], 1).cuda()
            sa_ON = torch.cat([state_D_simple, action_D_simple], 1)
            # 获取每个样本的分数
            sorce = policy.Discriminator(sa_D)

            sa_ON = sa_ON.cpu()
            all_states111.append(sa_ON)

            all_states111 = np.vstack(all_states111)
            # np.save('./sa-tsne/ant_ON_sa_tsne1.npy', all_states111)


            # 获取排序的索引，按照 sorce 从高到低排序
            sorted_indices, suoyin = torch.sort(sorce,dim=0, descending=True)

            # 计算前 25% 的索引数
            num_samples = len(sorted_indices)

            top_25_percent_index = int(num_samples * 0.25)

            # 选择前 25% 的样本
            top_25_percent_indices = suoyin[:top_25_percent_index]
            top_25_percent_indices = top_25_percent_indices.squeeze(1)
            # 从原始数据中选取这些前 25% 的样本

            top_state_D = state_D[top_25_percent_indices]
            top_action_D = action_D[top_25_percent_indices]
            top_next_state_D = next_state_D[top_25_percent_indices]
            top_reward_D = reward_D[top_25_percent_indices]
            top_done_bool_D = done_bool_D[top_25_percent_indices]

            # 将这些前 25% 的样本添加到新的缓冲区 IN_replay_buffer
            # print(top_state_D.size())
            # print(top_action_D.size())

            all_states = []
            state_IN_simple = top_state_D[0:4000, :]
            action_IN_simple = top_action_D[0:4000, :]
            s_a = torch.cat([state_IN_simple, action_IN_simple], -1).cpu()
            all_states.append(s_a)

            all_states = np.vstack(all_states)
            # np.save('./sa-tsne/ant_IN_sa_tsne1.npy', all_states)



            for s, a, ns, r, d in zip(top_state_D, top_action_D, top_next_state_D, top_reward_D, top_done_bool_D):

                s_a = s_a.cpu()
                s = s.cpu().numpy()
                ns = ns.cpu().numpy()
                a = a.cpu().numpy()
                r = r.cpu().numpy()
                d = d.cpu().numpy()



                IN_replay_buffer.add(s, a, ns, r, d)

            sorce_at_25_percent = sorce[suoyin[top_25_percent_index]]
            # 输出排在第 25% 处的 sorce 值
            print("排在第 25% 处的 sorce 值：")
            print(sorce_at_25_percent)


        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            # state_context, action_context, next_state_context, reward_context, not_done_context = IN_replay_buffer.sample(512)
            # state_context = torch.cat((state_context, action_context), dim=-1)
            # state_context = state_context.unsqueeze(0).cuda()
            # reward_context = reward_context.unsqueeze(0).cuda()
            #
            # # state_context1, action_context1, next_state_context1, reward_context1, not_done_context1 = replay_buffer_offline.sample(
            # #     512)
            # #
            # # state_context1 = state_context1.unsqueeze(0).cuda()
            # # reward_context1 = reward_context1.unsqueeze(0).cuda()
            # y_pred, kl, loss, z = model(state_context, reward_context, state_context, reward_context)
            # with torch.no_grad():
            #     z = z.squeeze(0)
            # state_, action_, next_state_, reward_, not_done_ = replay_buffer.sample(512)
            # state_ = torch.cat([state_, z], dim=-1)
            # next_state_ = torch.cat([next_state_, z], dim=-1)
            # critic_loss, actor_loss, info = policy.train_online(state_, action_, next_state_, reward_, not_done_, 512)
            # z = z[0:1, :]
            state_vae, action_vae, next_state_vae, reward_vae, not_done_vae = replay_buffer.sample(256)

            state_target = torch.cat((state_vae, action_vae), dim=-1)


            pre_r, mean_C, std_C, mean_T, std_T, z = model(state_target, reward_vae, state_target, reward_vae)

            z_de = z.expand(256, -1)



            with torch.no_grad():
                z = z.repeat(256, 1)
            state_, action_, next_state_, reward_, not_done_ = replay_buffer.sample(256)
            state_action_de = torch.cat((state_, action_), dim=-1)

            state_ = torch.cat([state_, z], dim=-1)
            next_state_ = torch.cat([next_state_, z], dim=-1)
            z = z[0:1, :]


            critic_loss, actor_loss, info = policy.train_online(state_, action_, next_state_, reward_, not_done_, z_de, model, state_action_de, 256)

        if done:
            state, done = env.reset(), False
            state = (np.array(state).reshape(1, -1) - mean) / std

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if t >= args.start_timesteps and (t + 1) % args.eval_freq == 0:

            avg_reward, normalized_returns = eval_policy(policy, args.env, args.seed, mean, std, z)

            writer.add_scalar("eval/episode_reward", avg_reward, t)
            writer.add_scalar("eval/normalized_returns", normalized_returns, t)
            print(f"timesteps {t} normalized_returns:{normalized_returns} episode_reward: {avg_reward:.3f}")


