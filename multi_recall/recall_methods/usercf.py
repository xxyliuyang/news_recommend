from multi_recall.recall_utils import get_item_user_time_dict
from tqdm import tqdm
from collections import defaultdict
import math
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class UserCF(object):
    def __init__(self):
        self.save_path = './output/'

    def get_user_activate_degree_dict(self, all_click_df: pd.DataFrame):
        all_click_df_ = all_click_df.groupby("user_id")['click_article_id'].count().reset_index()
        # 用户活跃度归一化
        mm = MinMaxScaler()
        all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
        user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))
        return user_activate_degree_dict

    def usercf_sim(self, all_click_df, user_activate_degree_dict):
        """
            用户相似性矩阵计算
            :param all_click_df: 数据表
            :param user_activate_degree_dict: 用户活跃度的字典
            return 用户相似性矩阵

            思路: 基于用户的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
        """
        item_user_time_dict = get_item_user_time_dict(all_click_df)

        u2u_sim = {}
        user_cnt = defaultdict(int)
        for item, user_time_list in tqdm(item_user_time_dict.items()):
            for u, click_time in user_time_list:
                user_cnt[u] += 1
                u2u_sim.setdefault(u, {})
                for v, click_time in user_time_list:
                    if u == v:
                        continue
                    u2u_sim[u].setdefault(v, 0)

                    # 用户平均活跃度作为活跃度的权重，这里的式子也可以改善
                    activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                    u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

        u2u_sim_ = u2u_sim.copy()
        for u, related_users in u2u_sim.items():
            for v, wij in related_users.items():
                u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

        # 将得到的相似性矩阵保存到本地
        pickle.dump(u2u_sim_, open(self.save_path + 'usercf_u2u_sim.pkl', 'wb'))

        return u2u_sim_