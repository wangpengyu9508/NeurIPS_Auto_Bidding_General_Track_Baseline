import numpy as np
import pandas as pd
import math
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test(data_loader, env, i, df, adv_index = 0):
    """
    offline evaluation
    """
    period_adv = data_loader.keys[adv_index]
    cpa_constraint = data_loader.test_dict[period_adv]['CPAConstraint'].iloc[0]
    adv_budget = data_loader.test_dict[period_adv]['budget'].iloc[0]
    adv_category = data_loader.test_dict[period_adv]['advertiserCategoryIndex'].iloc[0]
    agent = PlayerBiddingStrategy(cpa=cpa_constraint, budget=adv_budget, category=adv_category)
    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(period_adv)
    rewards = np.zeros(num_timeStepIndex)
    history = {
        'historyBids': [],
        'historyAuctionResult': [],
        'historyImpressionResult': [],
        'historyLeastWinningCost': [],
        'historyPValueInfo': []
    }
    for timeStep_index in range(num_timeStepIndex):
        pValue = pValues[timeStep_index]
        pValueSigma = pValueSigmas[timeStep_index]
        leastWinningCost = leastWinningCosts[timeStep_index]
        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pValue.shape[0])
        else:
            bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])
        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                      leastWinningCost)
        over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        agent.remaining_budget -= np.sum(tick_cost)
        rewards[timeStep_index] = np.sum(tick_conversion)
        temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
        history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(leastWinningCost)
        temAuctionResult = np.array(
            [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
        history["historyAuctionResult"].append(temAuctionResult)
        temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
        history["historyImpressionResult"].append(temImpressionResult)
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    budget_consumer_ratio = all_cost / agent.budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    cpa_exceed_rate = (cpa_real - cpa_constraint) / (cpa_constraint + 1e-10)
    score = getScore_nips(all_reward, cpa_real, cpa_constraint)
    df_new = pd.DataFrame({'epoch': i,
        'Periods': period_adv[0],
        'Adv': period_adv[1],
        'Score':score,
        'Reward': all_reward,
        'cpa_exceed_rate': cpa_exceed_rate,
        'budget_consumer_ratio': budget_consumer_ratio,
        'Cost': all_cost,
        'CPA-real': cpa_real,
        'CPA-constraint': cpa_constraint}, index=[0])
    if len(df) == 0:
        df = df_new
    else:
        df = pd.concat([df, df_new], ignore_index=True)
    return df

def run_mult_adv(file_path, test_epoch, save_path=None):
    print(file_path.split('/')[-1])
    data_loader = TestDataLoader(file_path)
    env = OfflineEnv()
    columns = ['epoch','Periods', 'Adv', 'Score', 'Reward', 'cpa_exceed_rate', 'budget_consumer_ratio', 'Cost', 'CPA-real', 'CPA-constraint']
    df = pd.DataFrame(columns=columns)
    for adv_index in range(len(data_loader.keys)):
        for i in range(test_epoch):
            df = run_test(data_loader, env, i, df, adv_index)
    result = df.groupby(['Adv']).mean().reset_index()
    result2 = result.groupby(['Periods']).mean().reset_index()
    score_all = result2['Score'].sum()
    if save_path:
        result.to_csv(save_path, index=False, encoding='utf-8')
    return score_all


if __name__ == '__main__':
    
    root_dir = "/Users/wangpengyu03/NeurIPS_Auto_Bidding_General_Track_Baseline/data/output/"
    data_list = ['period-27.csv']
    score = []
    
    for i in range(len(data_list)):
        for j in range(5): # 5次取平均值
            cur_score = run_mult_adv(file_path = root_dir + data_list[i], test_epoch=1, save_path = root_dir + data_list[i])
            score.append(cur_score)
            print("start evaluate ... at ", j + 1, " and cur_score: ", cur_score)

    avg_score = sum(score) * 1.0 / len(score)
    print("avg score: ", avg_score, ", score for every day data: ", score)
    