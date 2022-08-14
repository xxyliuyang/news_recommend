import numpy as np
from data_utils import get_all_click_df, get_item_info_df, get_item_emb_dict
from multi_recall.recall_utils import *

if __name__ == '__main__':
    data_path = './data_raw/'
    save_path = './temp_results/'
    # 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
    metric_recall = False

    # 采样数据
    # all_click_df = get_all_click_sample(data_path)
    # 全量训练集
    all_click_df = get_all_click_df(offline=False)
    item_info_df = get_item_info_df(data_path)
    item_emb_dict = get_item_emb_dict(data_path)

    # 对时间戳进行归一化,用于在关联规则的时候计算权重
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

    # 获取文章的属性信息，保存成字典的形式方便查询
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

    # 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
    user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                               'embedding_sim_item_recall': {},
                               'youtubednn_recall': {},
                               'youtubednn_usercf_recall': {},
                               'cold_start_recall': {}}

    # 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
    # 如果不是召回评估，直接使用全量数据进行召回，不用将最后一次提取出来
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
