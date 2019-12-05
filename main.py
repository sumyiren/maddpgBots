from MADDPG import MADDPG
import numpy as np
import torch as th
import visdom
from params import scale_reward
from world import world

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
batchSize = 60

n_episode = 20000
episodes_before_train = 3

win = None
param = None

maddpg = MADDPG(nAgents, obsSize, nActions, batchSize, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.resetWorld()
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((nAgents,))
    for t in range(totalTime):
        # render every 100 episodes to speed up training
        if i_episode % 100 == 0 and e_render:
            world.render()
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        actionSeller = np.argmax(np.array(action[0]))
        actionBuyer = np.argmax(np.array(action[1]))
        obs_, reward, done = world.stepWorld([actionSeller], [actionBuyer])

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != totalTime - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on WaterWorld\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % nAgents +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n')

    if win is None:
        win = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array([
                           np.append(total_reward, rr)]),
                       opts=dict(
                           ylabel='Reward',
                           xlabel='Episode',
                           title='MADDPG on WaterWorld_mod\n' +
                           'agent=%d' % nAgents +
                           ', sensor_range=0.2\n',
                           legend=['Total'] +
                           ['Agent-%d' % i for i in range(nAgents)]))
    else:
        vis.line(X=np.array(
            [np.array(i_episode).repeat(nAgents+1)]),
                 Y=np.array([np.append(total_reward,
                                       rr)]),
                 win=win,
                 update='append')
    if param is None:
        param = vis.line(X=np.arange(i_episode, i_episode+1),
                         Y=np.array([maddpg.var[0]]),
                         opts=dict(
                             ylabel='Var',
                             xlabel='Episode',
                             title='MADDPG on WaterWorld: Exploration',
                             legend=['Variance']))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([maddpg.var[0]]),
                 win=param,
                 update='append')

