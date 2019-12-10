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

    def __init__(self, totalTime = 0, sellerStartingPrice = 0, buyerStartingPrice = 0, maxPrice = 0, nSellers = 0):
        self.state = {
            "sellerAsk": sellerStartingPrice, 
            "buyerAsk": buyerStartingPrice, 
            "sellerDeal": 0,
            "buyerDeal": 0,
            "maxPrice": maxPrice, 
            "timeLeft": totalTime,
        }

        self.nSellers = nSellers
        self.reward = 0
        self.shaping = None
        self.prev_shaping = None
        self.done = False
        
    def step(self, actionSeller, actionBuyer):
        # state = self.state
        # sellerask, buyerask, minPrice, timeLeft = state
        if not self.done:
            plusMinusBuyer = self.actionValues[actionBuyer]
            dealBuyer = self.actionDeal[actionBuyer]
            plusMinusSeller = self.actionValues[actionSeller]
            dealSeller = self.actionDeal[actionSeller]
            self.state["sellerAsk"] += plusMinusSeller
            self.state["buyerAsk"] += plusMinusBuyer
            self.state["sellerDeal"] = dealSeller
            self.state["buyerDeal"] = dealBuyer
            self.state["timeLeft"] -= 1
            self.done = (self.state["sellerDeal"] == 1 and self.state["buyerDeal"] == 1) or (self.state["timeLeft"] <= 0)

    
    def calcBuyerReward(self):
        done = self.done
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

    def getDone(self):
        return self.done


    def getListState(self):
        # state should be array of [sellerAsk, buyerAsk, sellerDeal, buyerDeal, maxPrice, timeLeft]
        
        listState = [
            self.state["sellerAsk"],
            self.state["buyerAsk"], 
            self.state["sellerDeal"],
            self.state["buyerDeal"],
            self.state["maxPrice"], 
            self.state["timeLeft"], 
        ]

        listState = listState + ([0]*(4*(self.nSellers-1)))
        return listState
        
    
    
    
    
    
    