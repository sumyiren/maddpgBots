#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:18:38 2018

@author: sumyiren
"""

import math
import numpy as np
from sellerEnv import sellerEnv
from buyerEnv import buyerEnv

import random

class world():

    def __init__(self, nSellers, totalTime, teamSpirit):
        self.nSellers = nSellers
        self.totalTime = totalTime
        self.nSellers = nSellers
        self.buyerEnvs = []
        self.sellerEnvs = []
        self.sellerStates = []
        self.buyerStates = []
        self.teamSpirit = teamSpirit

        for i in range(self.nSellers):
            self.sellerEnvs.append(sellerEnv(self.totalTime, 0, 0, 0))
            self.buyerEnvs.append(buyerEnv(self.totalTime, 0, 0, 0))

    def stepWorld(self, actionsSeller, actionsBuyer):
        done = False
        #do seller step first
        for i in range(self.nSellers):
            self.sellerEnvs[i].step(actions_seller[i], actions_buyer[i])

        #do buyer step
        for i in range(self.nSellers):
            self.buyerEnvs[i].step(actions_buyer[i], actions_seller[i])

        #calc rewards for seller and buyer
        for i in range(self.nSellers):
            done = (self.sellerEnvs[i].getState()["sellerDeal"] == 1 and self.buyerEnvs[i].getState()["buyerDeal"] == 1) or (self.sellerEnvs[i].getState()["timeLeft"] <= 0)

            self.sellerEnvs[i].calcSellerReward(done)
            self.buyerEnvs[i].calcBuyerReward(done)

            self.sellerStates[i] = { "state": self.sellerEnvs[i].getState(), "reward": self.sellerEnvs[i].getReward() }
            self.buyerStates[i] = { "state": self.buyerEnvs[i].getState(),  "reward": self.buyerEnvs[i].getReward() }

        obs = np.stack(self.sellerEnvs[0].getState().values(), self.buyerEnvs[0].getState().values())
        reward = np.stack(self.sellerEnvs[0].getReward(), self.buyerEnvs[0].getReward())
        
        return obs, reward, done
        
        
    def resetWorld(self):
        n1 = 0
        n2 = self.totalTime/2
        
        minPrice = random.randint(n1,n2)
        sellerStartingPrice = minPrice + random.randint(n1,n2-1)
        
        self.sellerEnvs = []
        self.buyerEnvs = []
        self.sellerStates = []
        self.buyerStates = []

        for i in range(self.nSellers):
            
            buyerStartingPrice = random.randint(n1, sellerStartingPrice)
            maxPrice = random.randint(buyerStartingPrice, buyerStartingPrice + n2)
            
            self.sellerEnvs.append(sellerEnv(self.totalTime, sellerStartingPrice, buyerStartingPrice, minPrice))
            self.buyerEnvs.append(buyerEnv(self.totalTime, sellerStartingPrice, buyerStartingPrice, maxPrice))
            self.sellerStates.append({ "state": self.sellerEnvs[i].getState(), "reward": self.sellerEnvs[i].getReward() })
            self.buyerStates.append({ "state": self.buyerEnvs[i].getState(), "reward": self.buyerEnvs[i].getReward() })

        return np.stack(self.sellerEnvs[0].getState().values(), self.buyerEnvs[0].getState().values())
        


        