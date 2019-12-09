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
            self.sellerEnvs.append(sellerEnv())
            self.buyerEnvs.append(buyerEnv())

    def stepWorld(self, actions):
        done = False
        actions = [np.argmax(action) for action in actions]
        actionsSeller = actions[0:self.nSellers]
        actionsBuyer = actions[self.nSellers:self.nSellers*2]
        for i in range(self.nSellers):
            otherSellerActions = actionsSeller.copy()
            otherSellerActions = np.delete(otherSellerActions, i)
            otherBuyerActions = actionsBuyer.copy()
            otherBuyerActions = np.delete(otherBuyerActions, i)

            self.sellerEnvs[i].step(actionsSeller[i], actionsBuyer[i], otherSellerActions, otherBuyerActions)
            self.buyerEnvs[i].step(actionsSeller[i], actionsBuyer[i])

        #calc rewards for seller and buyer
        for i in range(self.nSellers):
            self.sellerEnvs[i].calcSellerReward()
            self.buyerEnvs[i].calcBuyerReward()

        obs = self.getObs()
        rewards = self.getReward()
        done = self.getDone()

        return obs, rewards, done
        
        
    def resetWorld(self):
        n1 = 0
        n2 = self.totalTime/2
        
        minPrice = random.randint(n1,n2)
        sellerStartingPrice = minPrice + random.randint(n1,n2-1)
        
        self.sellerEnvs = []
        self.buyerEnvs = []

        sellerAsks = []
        buyerAsks = []

        for i in range(self.nSellers):
            
            buyerStartingPrice = random.randint(n1, sellerStartingPrice)
            maxPrice = random.randint(buyerStartingPrice, buyerStartingPrice + n2)
            
            buyerAsks.append(buyerStartingPrice)
            self.buyerEnvs.append(buyerEnv(self.totalTime, sellerStartingPrice, buyerStartingPrice, maxPrice))

        for i in range(self.nSellers):
            otherSellerAsks = [sellerStartingPrice] * (self.nSellers-1)
            otherBuyerAsks = buyerAsks.copy()
            del otherBuyerAsks[i]
            self.sellerEnvs.append(sellerEnv(self.totalTime, sellerStartingPrice, buyerStartingPrice, minPrice, otherSellerAsks, otherBuyerAsks))
        
        return self.getObs()
        
    def getObs(self):
        obs = []
        for i in range(self.nSellers):
            obs.append(self.sellerEnvs[i].getListState())
        for i in range(self.nSellers):
            obs.append(self.buyerEnvs[i].getListState())

        return np.array(obs)

    def getReward(self):
        rewards = []
        for i in range(self.nSellers):
            rewards.append(self.sellerEnvs[i].getReward())
        for i in range(self.nSellers):
            rewards.append(self.buyerEnvs[i].getReward())

        return np.array(rewards)


    def getDone(self):
        dones = []
        for i in range(self.nSellers):
            dones.append(self.sellerEnvs[i].getDone())
        for i in range(self.nSellers):
            dones.append(self.buyerEnvs[i].getDone())
        done = all(dones)

        return done


        