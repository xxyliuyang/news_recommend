import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_utils import get_all_click_sample, get_all_click_df, submit
from base_model.cf_model import *

# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

def recal_by_cf():
    """协同过滤召回：基于物品的召回"""
    user_recall_items_dict = defaultdict(dict)
    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(all_click_df)
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))

    # 相似文章的数量
    sim_item_topk = 10
    # 召回文章数量
    recall_item_num = 10
    # 用户热度补全
    item_topk_click = get_item_topk_click(all_click_df, k=50)
    for user in tqdm(all_click_df['user_id'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim,
                                                            sim_item_topk, recall_item_num, item_topk_click)

    # 将字典的形式转换成df
    user_item_score_list = []
    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
    return recall_df


def output():
    # 获取测试集
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    tst_users = tst_click['user_id'].unique()

    # 从所有的召回数据中将测试集中的用户选出来
    tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]

    # 生成提交文件
    submit(tst_recall, topk=5, model_name='itemcf_baseline')

if __name__ == '__main__':
    data_path = './data/'
    save_path = './output/'

    # 全量训练集
    # all_click_df = get_all_click_df(data_path, offline=False)
    all_click_df = get_all_click_sample(data_path)
    i2i_sim = itemcf_sim(all_click_df)

    # 召回
    recall_df = recal_by_cf()
    output()