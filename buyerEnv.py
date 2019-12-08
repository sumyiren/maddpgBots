#!/usr/bin/env python5
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:18:58 2018

@author: sumyiren
"""

#Seller Environment


import gym
from gym.utils import seeding
import math
class buyerEnv():

    actionValues = [-1., 0., +1, 0]
    actionDeal = [0, 0, 0, 1]

    def __init__(self, totalTime, sellerStartingPrice, buyerStartingPrice, maxPrice):
        self.state = {
            "sellerAsk": sellerStartingPrice, 
            "buyerAsk": buyerStartingPrice, 
            "maxPrice": maxPrice, 
            "timeLeft": totalTime,
            "sellerDeal": 0,
            "buyerDeal": 0,
        }

        self.reward = 0
        self.shaping = None
        self.prev_shaping = None
        
    def step(self, actionSeller, actionBuyer):
        # state = self.state
        # sellerask, buyerask, minPrice, timeLeft = state

        plusMinusBuyer = self.actionValues[actionBuyer]
        dealBuyer = self.actionDeal[actionBuyer]
        plusMinusSeller = self.actionValues[actionSeller]
        dealSeller = self.actionDeal[actionSeller]
        self.state["sellerAsk"] += plusMinusSeller
        self.state["buyerAsk"] += plusMinusBuyer
        self.state["sellerDeal"] = dealSeller
        self.state["buyerDeal"] = dealBuyer
        self.state["timeLeft"] -= 1


    def calcBuyerReward(self, done):
        reward = 0
        shaping = 0
        sellerAsk = self.state["sellerAsk"]
        buyerAsk = self.state["buyerAsk"]
        maxPrice = self.state["maxPrice"]
        
        if done:
            if (buyerAsk >= sellerAsk and buyerAsk <= maxPrice): 
                reward = maxPrice - buyerAsk
                    
            else:
                reward = 0
                
            if buyerAsk <=0:
                reward = -100
                    
        # else:

        #     shaping = -2/timeLeft*abs(sellerask-buyerask) # 
        #     shaping += 1/timeLeft if (sellerask - minPrice) > 0 else -1/timeLeft
        
        #     if (sellerask - buyerask) < 0:
        #         shaping += -10
                
        #     if sellerask <=0:
        #         shaping += -10
                
        #     if sellerEnv.prev_shaping is not None:
        #         reward = shaping - sellerEnv.prev_shaping
        #     sellerEnv.prev_shaping = shaping
            
        self.reward = reward
        
    def getReward(self):
        return self.reward

    def getState(self):
        return self.state
        
    
    
    
    
    
    