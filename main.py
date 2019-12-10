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

nSellers = 2
nAgents = nSellers * 2
totalTime = 60
teamSpirit = 1
world = world(nSellers, totalTime, teamSpirit)
vis = visdom.Visdom(port=8097)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
obsSize = 4*nSellers+2
nActions = 4
capacity = 1000000
batchSize = 250

n_episode = 20000
episodes_before_train = 50

save_every_episodes = 1000

win = None
param = None

maddpg = MADDPG(nAgents, obsSize, nActions, batchSize, capacity,
                episodes_before_train)

rewardAveragers = [MovingAverager(100)] * nAgents
averageReward100 = [0] * nAgents

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
        actions = maddpg.select_action(obs).data.cpu()
        obs_, reward, done = world.stepWorld(actions)

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        next_obs = obs_
        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, actions, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    print(obs_)

  # reset
    done = True
    next_obs = None


    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on Bots\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % nAgents +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n')


    if maddpg.episode_done > maddpg.episodes_before_train:
        reward_record.append(total_reward)
        
        for i in range(nAgents):
            rewardAveragers[i].append(reward[i])
            print(reward[i])
            averageReward100[i] = rewardAveragers[i].average()


    if maddpg.episode_done % save_every_episodes == 0:
        print('saving now...')
        # maddpg.saveModel()

    if win is None:
        win = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array([averageReward100]),
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
                 Y=np.array([averageReward100]),
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

