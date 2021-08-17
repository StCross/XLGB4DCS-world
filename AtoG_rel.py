import os
import random
import sys
from math import asin, cos, radians, sin, sqrt

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

sys.path.append('..')

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class AtoG_rel:
    def __init__(self, mode, type='attack', seed=2021):
        self.seed = seed  # 随机种子
        self.window = 0  # 在线数据读取中滑动窗口的长度
        self.win_df = pd.DataFrame()  # 在线数据读取中始终维持一个长度为最大滑动窗口的dataframe
        self.mode = mode  # 离线或在线
        self.type = type  # attack 或者 locked

    @staticmethod
    def _haversine(lon1, lat1):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
        # 目标经纬度
        lon2 = 42.03421111
        lat2 = 42.24306389
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # 地球平均半径，单位为公里
        return c * r * 1000

    # 选择有分数的可以用于训练或测试的行并且打分(攻击对脱离)
    def get_score_attack(self, data_):
        """ 对攻击决策进行评分
        Args:
            data_(DataFrame): 原始数据
        Returns:
            dfs(DataFrame): 评分后的数据
        """
        # 在线模式下无需评分
        if self.mode == 'online':
            data = data_.copy()
            # 删除未来时刻的特征值
            for f in ['x', 'y', 'z']:
                del data[f'Missile_position_{f}_after_t']
                del data[f'Position_{f}_after_t']
            return data
        # 在线模式下评分
        elif self.mode == 'offline':
            data = data_.copy()
            dfs = pd.DataFrame()
            # 获得所有不重复的id
            id_list = list(set(data['id'].values))
            for id in id_list:
                df = data[data['id'] == id]
                # 通过时刻排序并重置索引
                df = df.sort_values(by=['Time'])
                df.reset_index(drop=True, inplace=True)
                # 由于只有30.5秒后被导弹攻击和安全逃离时刻可以打分,所以设置四个索引点,30.5秒后飞机被导弹攻击的第一个时刻,最后一个时刻以及当前飞机攻击的第一个和最后一个时刻
                first_attack = df[df['action type'] ==
                                  'attack target'].index.values[0]
                first_attacked_after_t = df[
                    df['Missile_position_x_after_t'] != 0].index.values[0]
                last_attacked_after_t = df[
                    df['Missile_position_x_after_t'] != 0].index.values[-1]
                last_attack = df[df['action type'] ==
                                 'attack target'].index.values[-1]
                # 只截取可以打分的行，这里需要比较飞机攻击的第一个时刻和30.5秒后飞机被导弹攻击的第一个时刻
                if first_attacked_after_t >= first_attack:
                    df = df.loc[first_attacked_after_t:last_attack]
                else:
                    df = df.loc[first_attack:last_attack]
                # 存在正在逃离和逃离成功两个阶段，分别评分
                if last_attacked_after_t < last_attack:
                    # 安全逃离期间直接打分为1
                    for index in np.arange(last_attacked_after_t + 1,
                                           last_attack + 1, 1):
                        df.loc[index, 'score'] = 1
                    # 飞机被导弹攻击时打分
                    df.loc[: last_attacked_after_t]['score'] = \
                        df.loc[: last_attacked_after_t].apply(
                            lambda x: np.sqrt(
                                (float(x['Position_x_after_t']) -
                                 float(x['Missile_position_x_after_t'])) ** 2 +
                                (float(x['Position_y_after_t']) -
                                 float(x['Missile_position_y_after_t'])) ** 2 +
                                (float(x['Position_z_after_t']) -
                                 float(x['Missile_position_z_after_t'])) ** 2) / 7500,
                            axis=1)
                # 飞机还未成功逃离时的评分
                else:
                    df['score'] = df.apply(lambda x: np.sqrt(
                        (float(x['Position_x_after_t']) - float(x[
                            'Missile_position_x_after_t']))**2 +
                        (float(x['Position_y_after_t']) - float(x[
                            'Missile_position_y_after_t']))**2 +
                        (float(x['Position_z_after_t']) - float(x[
                            'Missile_position_z_after_t']))**2) / 7500,
                                           axis=1)
                dfs = dfs.append(df).reset_index(drop=True)
            dfs.reset_index(drop=True, inplace=True)
            # 删除未来时刻的特征数据
            for f in ['x', 'y', 'z']:
                del dfs[f'Missile_position_{f}_after_t']
                del dfs[f'Position_{f}_after_t']
            return dfs

    def get_score_locked(self, data_):
        """ 瞄准决策评分
        Args:
            data_(DataFrame): 未打分数据
        Returns:
            data(DataFrame): 在线模式返回数据
            dfs(DataFrame): 离线模式返回打分后数据
        """
        # 在线模式无需评分
        if self.mode == 'online':
            data = data_.copy()
            # 删除未来时刻数据
            for f in ['MyLong', 'MyLat']:
                del data[f'{f}_after_t']
            return data
        elif self.mode == 'offline':
            data = data_.copy()
            dfs = pd.DataFrame()
            # 获取所有不相同的id
            id_list = list(set(data['id'].values))
            # min_distance变量用于对比获得最小值
            min_distance = 1000000
            for id in id_list:
                df = data[data['id'] == id]
                df = df.sort_values(by=['Time'])
                df.reset_index(drop=True, inplace=True)
                # 获取第一个瞄准时刻和最后一个瞄准时刻的索引
                first_target_locked_time = df[df['action type'] ==
                                              'target locked'].index.values[0]
                last_target_locked_time = df[df['action type'] ==
                                             'target locked'].index.values[-1]
                # 截取打分时刻
                df = df.loc[first_target_locked_time:last_target_locked_time]
                # 计算距离
                df['distance'] = df.apply(lambda x: self._haversine(
                    x['MyLong_after_t'], x['MyLat_after_t']),
                                          axis=1)
                # 获取最小距离
                if df['distance'].min() < min_distance:
                    min_distance = df['distance'].min()
                dfs = dfs.append(df).reset_index(drop=True)
            # 归一化评分
            dfs['score'] = dfs.apply(
                lambda x: (min(x['distance'], 6500) - min_distance) /
                (6500 - min_distance),
                axis=1)
            dfs.reset_index(drop=True, inplace=True)
            # 删除距离以及未来时刻的特征
            del dfs['distance']
            for f in ['MyLong', 'MyLat']:
                del dfs[f'{f}_after_t']
            return dfs

    def FE_attack(self, data_):
        """ 攻击决策的特征工程
        Args:
            data_(DataFrame): 原始数据
        Returns:
            data(DataFrame): 添加特征后数据
        """
        data = data_.copy()
        # 离线模式的特征工程
        if self.mode == 'offline':
            # 以id和time进行排序
            data = data.sort_values(by=['id', 'Time'])
            data.reset_index(drop=True, inplace=True)  # 这步一定要加，否则结果不可复现

            # 本机长度为3滑动窗口均值、最大值、最小值、标准差
            for f in ['speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll']:
                data_mean = pd.DataFrame()
                data_max = pd.DataFrame()
                data_min = pd.DataFrame()
                data_std = pd.DataFrame()
                data_mean[f'rolling_3_{f}_mean'] = data.groupby(
                    'id')[f].rolling(
                        window=3, min_periods=1).mean().reset_index(drop=True)
                data_max[f'rolling_3_{f}_max'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).max().reset_index(drop=True)
                data_min[f'rolling_3_{f}_min'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).min().reset_index(drop=True)
                data_std[f'rolling_3_{f}_std'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).std().reset_index(drop=True)
                # 合并到data中
                data = pd.concat(
                    [data, data_mean, data_max, data_min, data_std],
                    sort=False,
                    axis=1)
            # 空值补0
            data.fillna(0, inplace=True)

        # 在线模式的特征构造
        if self.mode == 'online':
            # 本机长度为3滑动窗口均值、最大值、最小值、标准差
            for f in ['speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll']:
                data_mean = pd.DataFrame()
                data_max = pd.DataFrame()
                data_min = pd.DataFrame()
                data_std = pd.DataFrame()

                # 本机长度为3滑动窗口均值、最大值、最小值、标准差
                data_mean[f'rolling_3_{f}_mean'] = data[f].rolling(
                    window=3, min_periods=1).mean().reset_index(drop=True)
                data_max[f'rolling_3_{f}_max'] = data[f].rolling(
                    window=3, min_periods=1).max().reset_index(drop=True)
                data_min[f'rolling_3_{f}_min'] = data[f].rolling(
                    window=3, min_periods=1).min().reset_index(drop=True)
                data_std[f'rolling_3_{f}_std'] = data[f].rolling(
                    window=3, min_periods=1).std().reset_index(drop=True)
                # 合并到data中
                data = pd.concat(
                    [data, data_mean, data_max, data_min, data_std],
                    sort=False,
                    axis=1)
            # 空值补0
            data.fillna(0, inplace=True)

        return data

    def online_FE_attack(self, row_dict):
        """ 攻击决策的在线模式下特征工程
        Args:
             row_dict(dict): 字典形式输入数据
        Returns:
            FE_row(DataFrame): 特征构造后的额数据
        """
        # 字典数据转化成dataframe
        row = pd.DataFrame(row_dict, index=[0])
        # 每输入一条数据滑动窗口加一
        self.window += 1
        self.win_df = self.win_df.append(row, ignore_index=True)  # 存储dataframe
        if self.window < 3:
            # 取特征工程后的最后一条数据
            FE_row = self.FE_attack(self.win_df).iloc[-1:]
            FE_row.fillna(0, inplace=True)
        else:
            # 取特征工程后的最后一条数据
            FE_row = self.FE_attack(self.win_df).iloc[-1:]
            FE_row.fillna(0, inplace=True)
            self.win_df.drop(0, inplace=True)
            self.win_df.reset_index(drop=True)
        return FE_row

    def FE_locked(self, data_):
        """ 瞄准决策的特征工程
        Args:
            data_(DataFrame): 原始数据
        Returns:
            data(DataFrame): 特征构造后的数据
        """
        data = data_.copy()
        if self.mode == 'offline':
            data = data.sort_values(by=['id', 'Time'])
            data.reset_index(drop=True, inplace=True)  # 这步一定要加，否则结果不可复现
            # 本机长度为3滑动窗口均值、最大值、最小值、标准差
            for f in ['speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll']:
                data_mean = pd.DataFrame()
                data_max = pd.DataFrame()
                data_min = pd.DataFrame()
                data_std = pd.DataFrame()
                data_mean[f'rolling_3_{f}_mean'] = data.groupby(
                    'id')[f].rolling(
                        window=3, min_periods=1).mean().reset_index(drop=True)
                data_max[f'rolling_3_{f}_max'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).max().reset_index(drop=True)
                data_min[f'rolling_3_{f}_min'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).min().reset_index(drop=True)
                data_std[f'rolling_3_{f}_std'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).std().reset_index(drop=True)
                data = pd.concat(
                    [data, data_mean, data_max, data_min, data_std],
                    sort=False,
                    axis=1)

            data.fillna(0, inplace=True)

        if self.mode == 'online':
            # 本机长度为3滑动窗口均值、最大值、最小值、标准差
            for f in ['speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll']:
                data_mean = pd.DataFrame()
                data_max = pd.DataFrame()
                data_min = pd.DataFrame()
                data_std = pd.DataFrame()
                data_mean[f'rolling_3_{f}_mean'] = data.groupby(
                    'id')[f].rolling(
                        window=3, min_periods=1).mean().reset_index(drop=True)
                data_max[f'rolling_3_{f}_max'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).max().reset_index(drop=True)
                data_min[f'rolling_3_{f}_min'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).min().reset_index(drop=True)
                data_std[f'rolling_3_{f}_std'] = data.groupby('id')[f].rolling(
                    window=3, min_periods=1).std().reset_index(drop=True)
                data = pd.concat(
                    [data, data_mean, data_max, data_min, data_std],
                    sort=False,
                    axis=1)

            data.fillna(0, inplace=True)

        return data

    def online_FE_locked(self, row_dict):
        """ 在线模式下瞄准决策的特征工程
        Args:
            row_dict(dict): 字典形式的输入数据
        Returns:
            FE_row(DataFrame): 特征构造后的数据
        """
        # 字典数据转dataframe
        row = pd.DataFrame(row_dict, index=[0])
        # 每次输入一条数据，窗口加一
        self.window += 1
        self.win_df = self.win_df.append(row, ignore_index=True)
        if self.window < 3:
            # 取特征工程后的最后一条数据
            FE_row = self.FE_locked(self.win_df).iloc[-1:]
            FE_row.fillna(0, inplace=True)
        else:
            # 取特征工程后的最后一条数据
            FE_row = self.FE_locked(self.win_df).iloc[-1:]
            FE_row.fillna(0, inplace=True)
            self.win_df.drop(0, inplace=True)
            self.win_df.reset_index(drop=True)
        return FE_row

    def train_val_split(self, df_train, percent=0.8):
        """ 数据集划分，划分数据集为训练集与测试
        Args:
            df_train(dataframe): 原始数据
            percent(float): 切分比例
        Returns:
            train(dataframe): 训练集
            val_data(dataframe): 验证集
        """
        all_ids = df_train['id'].values.tolist()
        all_ids = list(set(all_ids))
        # 每次 set 的结果都不一样，所以要先排序，防止结果不可复现
        all_ids.sort()
        # random.seed 只能生效一次，所以每次 random.sample 之前都要设置
        random.seed(self.seed)
        train_ids = random.sample(all_ids, int(len(all_ids) * percent))
        # print(val_ids)
        val_ids = list(set(all_ids) - set(train_ids))
        train_data = df_train[df_train['id'].isin(train_ids)]
        val_data = df_train[df_train['id'].isin(val_ids)]
        # 连续序列数据，但是是以单个样本建模的情况下，需要 shuffle 打乱

        train_data = train_data.sample(
            frac=1, random_state=self.seed).reset_index(drop=True)
        val_data = val_data.sample(
            frac=1, random_state=self.seed).reset_index(drop=True)
        return train_data, val_data

    def _feature_name(self):
        """ 获取保留列名
        Returns:
            feature_names(list): 列名信息
        """
        feature_names = []
        # 攻击决策的特征名
        if self.type == 'attack':
            feature_names = [
                'speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll',
                'Position_x', 'Position_y', 'Position_z', 'AngleAttack',
                'MyLat', 'MyLong', 'MyAlt', 'MyAngleSpeedx', 'MyAngleSpeedy',
                'MyAngleSpeedz', 'rolling_3_speed_x_mean',
                'rolling_3_speed_x_max', 'rolling_3_speed_x_min',
                'rolling_3_speed_x_std', 'rolling_3_speed_y_mean',
                'rolling_3_speed_y_max', 'rolling_3_speed_y_min',
                'rolling_3_speed_y_std', 'rolling_3_speed_z_mean',
                'rolling_3_speed_z_min', 'rolling_3_speed_z_std',
                'rolling_3_yaw_max', 'rolling_3_yaw_std',
                'rolling_3_pitch_mean', 'rolling_3_pitch_max',
                'rolling_3_pitch_min', 'rolling_3_pitch_std',
                'rolling_3_roll_mean', 'rolling_3_roll_max',
                'rolling_3_roll_min', 'rolling_3_roll_std'
            ]
        else:
            # 瞄准决策的特征名
            feature_names = [
                'speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll',
                'Position_x', 'Position_y', 'Position_z', 'AngleAttack',
                'MyLat', 'MyLong', 'MyAlt', 'MyAngleSpeedx', 'MyAngleSpeedy',
                'MyAngleSpeedz', 'rolling_3_speed_x_mean',
                'rolling_3_speed_x_max', 'rolling_3_speed_x_min',
                'rolling_3_speed_x_std', 'rolling_3_speed_y_mean',
                'rolling_3_speed_y_max', 'rolling_3_speed_y_min',
                'rolling_3_speed_y_std', 'rolling_3_speed_z_mean',
                'rolling_3_speed_z_min', 'rolling_3_speed_z_std',
                'rolling_3_yaw_max', 'rolling_3_yaw_std',
                'rolling_3_pitch_mean', 'rolling_3_pitch_max',
                'rolling_3_pitch_min', 'rolling_3_pitch_std',
                'rolling_3_roll_mean', 'rolling_3_roll_max',
                'rolling_3_roll_min', 'rolling_3_roll_std'
            ]

        return feature_names

    # 留出法数据
    def _hold_out(self, raw_train, percent_train):
        """ 获取留出法的训练数据
        Args:
            raw_train(dataframe): 原始数据
            percent_train(float): 训练集占比
        Returns:
            train(dataframe): 训练集
            val(dataframe): 验证集
        """
        # 获取特征名
        feature_names = self._feature_name()
        # 分割训练集和验证集
        val_data, train_data = self.train_val_split(raw_train,
                                                    percent=1 -
                                                    float(percent_train))
        X_train, X_val, y_train, y_val = train_data[feature_names], val_data[
            feature_names], train_data['score'], val_data['score']
        return X_train, X_val, y_train, y_val

    # k折交叉验证数据
    def _k_fold(self, raw_train, k):
        """ 获取交叉验证数据
        Args:
            raw_train(dataframe): 原始数据
            k(int): 交叉折数
        Returns:
            train(dataframe): k折交叉验证的训练集
            val(dataframe): 验证集
        """
        # 获取特征名
        feature_names = self._feature_name()
        # 根据id分组
        groups = list(raw_train['id'])
        # 分组交叉验证
        gkf = GroupKFold(n_splits=k)
        data_list = []
        # 获取交叉验证的数据
        for train_index, val_index in gkf.split(raw_train[feature_names],
                                                raw_train['score'],
                                                groups=groups):
            # 获取每一折数据
            X_train, y_train, X_val, y_val = raw_train.iloc[train_index][feature_names], \
                                             raw_train.iloc[train_index]['score'], \
                                             raw_train.iloc[val_index][feature_names], \
                                             raw_train.iloc[val_index]['score']
            # 加入列表
            data_list.append([X_train, X_val, y_train, y_val])
        return data_list

    # bootstrap验证数据
    def _bootstrap(self, raw_train):
        """ 获取自助法数据
        Args:
            raw_train(dataframe): 原始数据
        Returns:
            train(dataframe): 训练集
            val(dataframe): 验证集
        """
        # 获取特征值
        feature_names = self._feature_name()
        # 获取不重复的id
        ids = pd.DataFrame(set(raw_train['id']), columns=['id'], index=None)
        # 使用随机种子
        random.seed(self.seed)
        # 使用有放回的随机采样
        train_group_ids = ids.sample(frac=1.0,
                                     replace=True,
                                     random_state=self.seed)
        # 得到未被采样到的id
        val_group_ids = ids.loc[ids.index.difference(
            train_group_ids.index)].copy()
        train_data = pd.DataFrame()
        val_data = pd.DataFrame()
        train_group_ids = list(train_group_ids['id'])
        val_group_ids = list(val_group_ids['id'])
        # 根据采样到的id得到训练集
        for train_group_id in train_group_ids:
            train_data = train_data.append(
                raw_train[raw_train['id'] == train_group_id])
        # 根据未采样到的id得到验证集
        for val_group_id in val_group_ids:
            val_data = val_data.append(
                raw_train[raw_train['id'] == val_group_id])
        X_train, X_val, y_train, y_val = train_data[feature_names], val_data[
            feature_names], train_data['score'], val_data['score']

        return X_train, X_val, y_train, y_val

    # 定义lightGBM模型
    def _train_lgb(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练lightgbm模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
            importance(dataframe): 特征重要度
        """
        feature_names = self._feature_name()

        lgb_model = lgb.LGBMRegressor(boosting_type='gbdt',
                                      num_leaves=32,
                                      max_depth=6,
                                      learning_rate=0.01,
                                      n_estimators=10000,
                                      subsample=0.8,
                                      feature_fraction=0.6,
                                      reg_alpha=10,
                                      reg_lambda=12,
                                      random_state=self.seed,
                                      is_unbalance=True)

        # 留出法
        if val_type == 'hold_out':
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 模型训练
            lgb_model.fit(X_train,
                          y_train,
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          verbose=50,
                          early_stopping_rounds=50)

            # 获取模型的特征重要性特征
            df_importance = pd.DataFrame({
                'column':
                feature_names,
                'importance':
                lgb_model.feature_importances_,
            }).sort_values(by='importance',
                           ascending=False).reset_index(drop=True)
            # 特征重要性归一化
            df_importance['importance'] = (
                df_importance['importance'] -
                df_importance['importance'].min()) / (
                    df_importance['importance'].max() -
                    df_importance['importance'].min())

            return lgb_model, df_importance

        # k折交叉验证
        elif val_type == 'k-fold':
            lgb_model_list = []
            importance_list = []
            # 划分的数据列表
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                # 划分训练集和验证集
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 模型训练
                names['lgb_%s' % i] = lgb_model.fit(X_train,
                                                    y_train,
                                                    eval_set=[(X_train,
                                                               y_train),
                                                              (X_val, y_val)],
                                                    verbose=50,
                                                    early_stopping_rounds=50)

                # 获取模型的特征重要性特征
                df_importance = pd.DataFrame({
                    'column':
                    feature_names,
                    'importance':
                    names['lgb_%s' % i].feature_importances_,
                }).sort_values(by='importance',
                               ascending=False).reset_index(drop=True)
                # 放入模型列表和特征重要性列表
                lgb_model_list.append(names['lgb_%s' % i])
                importance_list.append(df_importance)

            # 设置一个字典存储特征重要性的平均值
            mean_dict = dict()
            for feat in feature_names:
                mean_dict[feat] = 0
            # 不同划分的相同特征的重要性求和
            for df_importance in importance_list:
                for feat in feature_names:
                    mean_dict[feat] += int(
                        df_importance[df_importance['column'] ==
                                      feat]['importance'].values[0])
            # 每个特征的重要性取平均值
            for feat in feature_names:
                mean_dict[feat] /= k
            # 特征重要性放入dataframe
            mean_imp_df = pd.DataFrame({
                'column': feature_names,
                'importance': list(mean_dict.values()),
            }).sort_values(by='importance',
                           ascending=False).reset_index(drop=True)
            # 特征重要性归一化
            mean_imp_df['importance'] = (mean_imp_df['importance'] -
                                         mean_imp_df['importance'].min()) / (
                                             mean_imp_df['importance'].max() -
                                             mean_imp_df['importance'].min())

            return lgb_model_list, mean_imp_df

        # 自助法
        elif val_type == 'bootstrap':
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 模型训练
            lgb_model.fit(X_train,
                          y_train,
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          verbose=50,
                          early_stopping_rounds=50)

            # 获取模型的特征重要性特征
            df_importance = pd.DataFrame({
                'column':
                feature_names,
                'importance':
                lgb_model.feature_importances_,
            }).sort_values(by='importance',
                           ascending=False).reset_index(drop=True)
            # 特征重要性归一化
            df_importance['importance'] = (
                df_importance['importance'] -
                df_importance['importance'].min()) / (
                    df_importance['importance'].max() -
                    df_importance['importance'].min())

            return lgb_model, df_importance

        else:
            print("无该验证方法!")
            exit(-1)

    # 贝叶斯岭回归
    def _train_br(self, raw_train, val_type, k, percent_train=0.8):
        """ 贝叶斯岭回归
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
        """
        # 留出法
        if val_type == 'hold_out':
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 定义模型
            br_model = linear_model.BayesianRidge()
            # 模型训练
            br_model.fit(X_train, y_train)

            return br_model, None

        # k折交叉验证
        elif val_type == 'k-fold':
            br_model_list = []
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                # 得到训练集和验证集
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 定义模型
                names['br_%s' % i] = linear_model.BayesianRidge()
                # 模型训练
                names['br_%s' % i].fit(X_train, y_train)
                # 模型添加到模型列表
                br_model_list.append(names['br_%s' % i])

            return br_model_list, None

        # 自助法
        elif val_type == 'bootstrap':
            # 得到训练集和验证集
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 定义模型
            br_model = linear_model.BayesianRidge()
            # 模型训练
            br_model.fit(X_train, y_train)

            return br_model, None

        else:
            print('无该验证方法')
            exit(-1)

    # 线性回归
    def _train_linearReg(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练线性回归模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
        """
        # 留出法
        if val_type == 'hold_out':
            # 得到训练集和验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 定义线性回归模型
            linearReg_model = linear_model.LinearRegression()
            # 模型训练
            linearReg_model.fit(X_train, y_train)

            return linearReg_model, None

        # k折交叉验证
        elif val_type == 'k-fold':
            linearReg_model_list = []
            # 划分得到数据列表
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                # 得到训练集和验证集
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 定义线性回归模型
                names['br_%s' % i] = linear_model.LinearRegression()
                # 模型训练
                names['br_%s' % i].fit(X_train, y_train)
                # 加入模型列表
                linearReg_model_list.append(names['br_%s' % i])

            return linearReg_model_list, None

        # 自助法
        elif val_type == 'bootstrap':
            # 得到训练集和验证集
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 定义模型
            linearReg_model = linear_model.LinearRegression()
            # 模型训练
            linearReg_model.fit(X_train, y_train)

            return linearReg_model, None

        else:
            print('无该验证方法')
            exit(-1)

    # 支持向量机回归
    def _train_svm(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练支持向量机模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
        """
        # 留出法
        if val_type == 'hold_out':
            # 得到训练集和验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 定义模型
            svm_model = svm.SVR()
            # 模型训练
            svm_model.fit(X_train, y_train)

            return svm_model, None

        # k折交叉验证
        elif val_type == 'k-fold':
            svm_model_list = []
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                # 得到训练集和验证集
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 定义模型
                names['br_%s' % i] = svm.SVR()
                # 模型训练
                names['br_%s' % i].fit(X_train, y_train)
                # 添加到模型列表
                svm_model_list.append(names['br_%s' % i])

            return svm_model_list, None

        # 自助法
        elif val_type == 'bootstrap':
            # 得到训练集和验证集
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 定义模型
            svm_model = svm.SVR()
            # 模型训练
            svm_model.fit(X_train, y_train)

            return svm_model, None

        else:
            print('无该验证方法')
            exit(-1)

    # 集成模型
    def _train_ensemble(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练集成模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
        """
        # 定义随机森林模型
        r1 = RandomForestRegressor(random_state=self.seed)
        # 定义贝叶斯模型
        r2 = linear_model.BayesianRidge()

        # 留出法
        if val_type == 'hold_out':
            # 得到训练集和验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 定义集成模型
            model = VotingRegressor([('rfr', r1), ('br', r2)])
            # 模型训练
            model.fit(X_train, y_train)

            return model, None

        # k折交叉验证
        elif val_type == 'k-fold':
            model_list = []
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                # 得到训练集和验证集
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 定义模型
                names['br_%s' % i] = VotingRegressor([('rfr', r1), ('br', r2)])
                # 训练模型
                names['br_%s' % i].fit(X_train, y_train)
                # 添加到模型列表
                model_list.append(names['br_%s' % i])

            return model_list, None

        # 自助法
        elif val_type == 'bootstrap':
            # 得到训练集和验证集
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 定义模型
            model = VotingRegressor([('rfr', r1), ('br', r2)])
            # 模型训练
            model.fit(X_train, y_train)

            return model, None

        else:
            print('无该验证方法')
            exit(-1)

    # 模型训练
    def train_model(self, train, model_type, val_type, k=5, percent_train=0.8):
        # 训练lgb模型
        if model_type == 'lgb':
            model, df_importance = self._train_lgb(train,
                                                   val_type,
                                                   k,
                                                   percent_train=percent_train)
            return model, df_importance
        # 训练贝叶斯模型
        elif model_type == 'br':
            model, df_importance = self._train_br(train,
                                                  val_type,
                                                  k,
                                                  percent_train=percent_train)
            return model, df_importance
        # 训练线性回归模型
        elif model_type == 'linear':
            model, df_importance = self._train_linearReg(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance
        # 训练支持向量机模型
        elif model_type == 'svm':
            model, df_importance = self._train_svm(train,
                                                   val_type,
                                                   k,
                                                   percent_train=percent_train)
            return model, df_importance
        # 训练集成模型
        elif model_type == 'ensemble':
            model, df_importance = self._train_ensemble(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance
        else:
            print('无该模型！')
            exit(-1)

    @staticmethod
    def save_model_attack(model, save_dir, name, val_type):
        """ 保存攻击决策的模型
        Args:
            model: 模型
            save_dir(str): 存储路径
            name(str): 模型名
            val_type(str): 验证方式
        Returns:
            None
        """
        # 新建存储文件夹
        if not os.path.exists(os.path.join(save_dir, 'model_save_attack')):
            os.makedirs(os.path.join(save_dir, 'model_save_attack'))
        if val_type == 'k-fold':
            for i in range(len(model)):
                # 命名保存路径
                save_path = os.path.join(
                    save_dir, 'model_save_attack',
                    name + str(i) + '_' + val_type + '_.pkl')
                # 长久保存模型
                joblib.dump(model[i], save_path)
        else:
            # 保存路径
            save_path = os.path.join(save_dir, 'model_save_attack',
                                     name + '_' + val_type + '_.pkl')
            # 长久保存模型
            joblib.dump(model, save_path)

    @staticmethod
    def save_model_locked(model, save_dir, name, val_type):
        """ 保存瞄准决策的模型
        Args:
            model: 模型
            save_dir(str): 存储路径
            name(str): 模型名
            val_type(str): 验证方式
        Returns:
            None
        """
        # 新建存储文件夹
        if not os.path.exists(os.path.join(save_dir, 'model_save_locked')):
            os.makedirs(os.path.join(save_dir, 'model_save_locked'))
        if val_type == 'k-fold':
            for i in range(len(model)):
                # 保存路径
                save_path = os.path.join(
                    save_dir, 'model_save_locked',
                    name + str(i) + '_' + val_type + '_.pkl')
                # 保存模型
                joblib.dump(model[i], save_path)
        else:
            # 保存路径
            save_path = os.path.join(save_dir, 'model_save_locked',
                                     name + '_' + val_type + '_.pkl')
            # 长久保存模型
            joblib.dump(model, save_path)

    @staticmethod
    def load_model(path):
        """ 加载模型
        Args:
            path(string): 模型路径
        Returns:
            None
        """
        model = joblib.load(path)
        return model

    def offline_predict(self, model, df_test):
        """ 离线预测
        Args:
            model: 模型
            df_test(dataframe): 测试集
        Returns:
            pre_score_list(list): 预测评分列表
        """
        feature_names = self._feature_name()
        df_test['pred_score'] = 0

        # 对k折交叉验证得到的模型列表进行预测评分后求平均值
        if isinstance(model, list):
            for i in range(len(model)):
                y_pred = model[i].predict(df_test[feature_names])
                df_test['pred_score'] += y_pred

            df_test['pred_score'] /= len(model)

        else:
            # 预测评分
            y_pred = model.predict(df_test[feature_names])
            df_test['pred_score'] = y_pred

        # 大于1的评分置为1， 小于0的评分置为0
        df_test[df_test['pred_score'] > 1] = 1
        df_test[df_test['pred_score'] < 0] = 0

        return df_test['pred_score'].values

    def online_predict(self, model, df_test):
        """ 在线预测
        Args:
            model: 模型
            df_test(dataframe): 测试集
        Returns:
            pre_score(float): 在线的预测评分
        """
        # 获取特征名
        feature_names = self._feature_name()
        # 在线预测
        y_pred = model.predict(df_test[feature_names])
        # 放入dataframe
        df_pred = pd.DataFrame(y_pred, columns=['pred_score'])
        # 评分大于1的置为1
        df_pred[df_pred['pred_score'] > 1] = 1
        # 评分小于0的置为0
        df_pred[df_pred['pred_score'] < 0] = 0

        return df_pred['pred_score'].values[0]

    @staticmethod
    def score(label, predict):
        """ 误差分析
        Args:
            label(float): 实际评分
            predict(float): 预测评分
        Returns:
            mse(float): 均方误差
            mae(float): 平均绝对误差
        """
        # 计算均方误差
        mse = mean_squared_error(label, predict)
        # 计算平均绝对误差
        mae = mean_absolute_error(label, predict)

        return {'mse': mse, 'mae': mae}

    def evaluate_attack(self, dictionary, model_name, val_type, save_dir=''):
        """ 保存攻击决策的误差分析的结果
        Args:
            dictionary(dict): 字典类型误差结果
            model_name(str): 模型名称
            val_type(str): 验证方式
            save_dir(str): 保存路径
        Returns:
            None
        """
        # 新建文件夹
        if not os.path.exists(os.path.join(save_dir, 'evaluation_attack')):
            os.makedirs(os.path.join(save_dir, 'evaluation_attack'))
        # 获取均方误差和平均绝对误差
        mse = dictionary['mse']
        mae = dictionary['mae']
        print(f'mse:{mse}')
        print(f'mae:{mae}')
        # 将误差分析结果写入文本文档
        with open(os.path.join(save_dir, 'evaluation_attack', model_name +
                               '_' + val_type + '_evaluation.txt'),
                  'w',
                  encoding='utf-8') as fr:
            line = f'{model_name}模型的mse为' + '%.4f' % mse + ',mae为' + '%.4f' % mae + '\n'
            fr.write(line)

    def evaluate_locked(self, dictionary, model_name, val_type, save_dir=''):
        """ 保存瞄准决策的误差分析的结果
        Args:
            dictionary(dict): 字典类型误差结果
            model_name(str): 模型名称
            val_type(str): 验证方式
            save_dir(str): 保存路径
        Returns:
            None
        """
        # 新建文件夹
        if not os.path.exists(os.path.join(save_dir, 'evaluation_locked')):
            os.makedirs(os.path.join(save_dir, 'evaluation_locked'))
        # 获取均方误差和平均绝对误差
        mse = dictionary['mse']
        mae = dictionary['mae']
        print(f'mse:{mse}')
        print(f'mae:{mae}')
        # 保存误差分析结果
        with open(os.path.join(save_dir, 'evaluation_locked', model_name +
                               '_' + val_type + '_evaluation.txt'),
                  'w',
                  encoding='utf-8') as fr:
            line = f'{model_name}模型的mse为' + '%.4f' % mse + ',mae为' + '%.4f' % mae + '\n'
            fr.write(line)
