#!/usr/bin/env python5
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:18:58 2018

@author: sumyiren
"""

#Seller Environment

import math
class sellerEnv():

    actionValues = [-1., 0., +1, 0]
    actionDeal = [0, 0, 0, 1]

    def __init__(self, totalTime = 0, sellerStartingPrice = 0, buyerStartingPrice = 0, minPrice = 0, otherSellerAsks = [], otherBuyerAsks = []):

        assert isinstance(otherBuyerAsks, list)
        assert isinstance(otherSellerAsks, list)
        assert (len(otherBuyerAsks) == len(otherSellerAsks))

        self.state = {
            "sellerAsk": sellerStartingPrice, 
            "buyerAsk": buyerStartingPrice, 
            "minPrice": minPrice, 
            "sellerDeal": 0,
            "buyerDeal": 0,
            "otherSellerAsks": otherSellerAsks,
            "otherBuyerAsks": otherBuyerAsks,
            "otherSellerDeals": [0]*len(otherSellerAsks),
            "otherBuyerDeals": [0]*len(otherBuyerAsks),
            "timeLeft": totalTime,
        }

        self.reward = 0
        self.shaping = None
        self.prev_shaping = None
        self.done = False
        
    def step(self, sellerAction, buyerAction, otherSellerActions, otherBuyerActions):
        # state = self.state
        # sellerask, buyerask, minPrice, timeLeft = state
        for i in range(len(otherSellerActions)):

            done = (self.state["otherSellerDeals"][i] == 1 and self.state["otherBuyerDeals"][i] == 1)
            
            otherBuyerAction = otherBuyerActions[i]
            plusMinusBuyer = self.actionValues[otherBuyerAction]
            dealBuyer = self.actionDeal[otherBuyerAction]

            otherSellerAction = otherSellerActions[i]
            plusMinusSeller = self.actionValues[otherSellerAction]
            dealSeller = self.actionDeal[otherSellerAction]

            if not done:
                self.state["otherSellerAsks"][i] += plusMinusSeller
                self.state["otherSellerDeals"][i] = dealSeller
                self.state["otherBuyerAsks"][i] += plusMinusBuyer
                self.state["otherBuyerDeals"][i] = dealBuyer


        if not self.done:
            plusMinusBuyer = self.actionValues[buyerAction]
            dealBuyer = self.actionDeal[buyerAction]
            plusMinusSeller = self.actionValues[sellerAction]
            dealSeller = self.actionDeal[sellerAction]
            self.state["sellerAsk"] += plusMinusSeller
            self.state["buyerAsk"] += plusMinusBuyer
            self.state["sellerDeal"] = dealSeller
            self.state["buyerDeal"] = dealBuyer
            self.state["timeLeft"] -= 1
            self.done = (self.state["sellerDeal"] == 1 and self.state["buyerDeal"] == 1) or (self.state["timeLeft"] <= 0)


    def calcSellerReward(self):
        done = self.done
        reward = 0
        shaping = 0
        sellerAsk = self.state["sellerAsk"]
        buyerAsk = self.state["buyerAsk"]
        minPrice = self.state["minPrice"]
        
        if done:
            if (buyerAsk >= sellerAsk and sellerAsk >= minPrice):
                # reward = 10
                reward = buyerAsk - minPrice
                    
            else:
                reward = 0
            
            if sellerAsk <=0:
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
        # state should be array of [sellerAsk, buyerAsk, sellerDeal, buyerDeal, minPrice, timeLeft, otherSellerAsk1, otherBuyerAsk1, otherSellerDeal1, otherBuyerDeal1 ]
        
        listState = [
            self.state["sellerAsk"],
            self.state["buyerAsk"], 
            self.state["sellerDeal"],
            self.state["buyerDeal"],
            self.state["minPrice"], 
            self.state["timeLeft"], 
        ]
        for i in range(len(self.state["otherSellerAsks"])):
            listState.append(self.state["otherSellerAsks"][i])
            listState.append(self.state["otherBuyerAsks"][i])
            listState.append(self.state["otherSellerDeals"][i])
            listState.append(self.state["otherBuyerDeals"][i])

        return listState
        
    
    
    
    
    
    