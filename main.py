from MADDPG import MADDPG
import numpy as np
import torch as th
import visdom
from params import scale_reward
from world import world


class MovingAverager:
    def __init__(self, bufferLength):
        self.buffer = np.zeros(bufferLength)

    def append(self, x):
        self.buffer = np.delete(self.buffer, 0)
        self.buffer = np.append(self.buffer, x)

    def average(self):
        return np.average(self.buffer)

# do not render the scene
e_render = False

nSellers = 1
nAgents = nSellers * 2
totalTime = 60
teamSpirit = 1
world = world(nSellers, totalTime, teamSpirit)
vis = visdom.Visdom(port=8097)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
obsSize = 6
nActions = 4
capacity = 1000000
batchSize = 250

n_episode = 20000
episodes_before_train = 50

win = None
param = None

maddpg = MADDPG(nAgents, obsSize, nActions, batchSize, capacity,
                episodes_before_train)

sellerRewardAverager100 = MovingAverager(100)
buyerRewardAverager100 = MovingAverager(100)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.resetWorld()
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((nAgents,))
    done = False
    while(not done):
        # render every 100 episodes to speed up training
        if i_episode % 100 == 0 and e_render:
            world.render()
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        actionSeller = np.argmax(np.array(action[0]))
        actionBuyer = np.argmax(np.array(action[1]))
        obs_, reward, done, obs_state_ = world.stepWorld([actionSeller], [actionBuyer])

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        next_obs = obs_
        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    print(obs_state_)
    reward_record.append(total_reward)
    sellerRewardAverager100.append(rr[0])
    buyerRewardAverager100.append(rr[1])
    sellerAverageReward100 = sellerRewardAverager100.average()
    buyerAverageReward100 = buyerRewardAverager100.average()

    # reset
    done = True
    next_obs = None


    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on Bots\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % nAgents +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n')

    if win is None:
        win = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array([[sellerAverageReward100, buyerAverageReward100]]),
                       opts=dict(
                           ylabel='Average Reward over 100 eps',
                           xlabel='Episode',
                           title='MADDPG on Bots\n' +
                           'agent=%d' % nAgents +
                           ', sensor_range=0.2\n',
                           legend=['Agent-%d' % i for i in range(nAgents)]))
    else:
        vis.line(X=np.array(
            [np.array(i_episode).repeat(nAgents)]),
                 Y=np.array([[sellerAverageReward100, buyerAverageReward100]]),
                 win=win,
                 update='append')
    if param is None:
        param = vis.line(X=np.arange(i_episode, i_episode+1),
                         Y=np.array([maddpg.var[0]]),
                         opts=dict(
                             ylabel='Var',
                             xlabel='Episode',
                             title='MADDPG: Exploration',
                             legend=['Variance']))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([maddpg.var[0]]),
                 win=param,
                 update='append')

