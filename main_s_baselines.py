# from stable_baselines3 import PPO
# from stable_baselines3 import A2C
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.a2c.policies import CnnPolicy
# from stable_baselines3.common.policies import ActorCriticCnnPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from nq_environment_baselines import *
from power_switch import *
from stable_baselines3.common.logger import configure

run_terminator_listener()
iteration = 6
env = NQEnv()

model = RecurrentPPO.load(f"ppo_loop_40x40_recurrent_n_step_7states_{iteration}_1360000")
model.ent_coef = 0.14
model.set_env(env)

print("ent_coef value is " + str(model.ent_coef))
print("total timesteps is " + str(model._total_timesteps))
print("num timesteps is " + str(model.num_timesteps))


# env = DummyVecEnv([lambda: NQEnv()])
# env = make_vec_env(NQEnv, n_envs=1)
# env = VecFrameStack(env, n_stack = 5)

# check_env(env, warn=True)
# check_env(env)
# print("check is finished")

# env.reset()
# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="my_tf_board")

# model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, tensorboard_log="my_tf_board", ent_coef=0.1, learning_rate=0.0001)
# model = RecurrentPPO("CnnLstmPolicy", env, verbose=1)

# model = RecurrentPPO("CnnLstmPolicy", env, verbose=1)

# model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, ent_coef=0.15, learning_rate=0.0003)

# model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, n_steps=5120, batch_size=5120, ent_coef=0.15)

# model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, n_steps=5120, batch_size=5120, ent_coef=0.15)

# new_logger = configure(None, ["stdout"])

# model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, ent_coef=0.08, learning_rate=0.0003)
# model.load(f"ppo_loop_40x40_recurrent_pingpong_{iteration}_160000")



# model.set_logger(new_logger)

# env.reset()

# episode = 10
# for episode in range(episode):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         print("Goal reached!", "reward=", reward)
#         break





# Train the agent
TIMESTEPS = 20000

for i in range(1, 900000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=5120)
    model.save(f"ppo_loop_40x40_recurrent_n_step_7states_{iteration+1}_{TIMESTEPS*i}")
    print("check point has been saved")



# for i in range(30):
#     if i == 0:
#         model = PPO.load("nimble_quest_stable_baselines_ppo_20000", tensorboard_log="my_tf_board")
#     else:
#         model = PPO.load("nimble_quest_stable_baselines_ppo_20000_" + str(i), tensorboard_log="my_tf_board")
#     model.set_env(env)
#     model.learn(total_timesteps=20000, log_interval=1, reset_num_timesteps=False)
#     model.save("nimble_quest_stable_baselines_ppo_20000_" + str(i + 1))
#     mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# model = PPO.load("nimble_quest_stable_baselines_ppo_20000_1", tensorboard_log="my_tf_board")
# model.set_env(env)
# model.learn(total_timesteps=30000, log_interval=30000, reset_num_timesteps=False)
# model.save("nimble_quest_stable_baselines_ppo_20000_2")

# model = PPO.load("nimble_quest_stable_baselines_ppo_20000_2", tensorboard_log="my_tf_board")
# model.set_env(env)
# model.learn(total_timesteps=30000, log_interval=30000, reset_num_timesteps=False)
# model.save("nimble_quest_stable_baselines_ppo_20000_3")

# model = PPO.load("nimble_quest_stable_baselines_ppo_20000_3", tensorboard_log="my_tf_board")
# model.set_env(env)
# model.learn(total_timesteps=30000, log_interval=30000, reset_num_timesteps=False)
# model.save("nimble_quest_stable_baselines_ppo_20000_4")

# model = PPO.load("nimble_quest_stable_baselines_ppo_20000_4", tensorboard_log="my_tf_board")
# model.set_env(env)
# model.learn(total_timesteps=30000, log_interval=30000, reset_num_timesteps=False)
# model.save("nimble_quest_stable_baselines_ppo_20000_5")

# obs = env.reset()
# n_steps = 10000
# for step in range(n_steps):
#   action, _ = model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', 'reward=', reward, 'done=', done)
#   if done:
#     # Note that the VecEnv resets automatically
#     # when a done signal is encountered
#     print("Goal reached!", "reward=", reward)
#     break