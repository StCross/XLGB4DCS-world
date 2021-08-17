import os
import random
import sys
from math import asin, cos, radians, sin, sqrt
from warnings import simplefilter

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize

simplefilter(action='ignore')

sys.path.append('..')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class AtoG:
    def __init__(self, seed, scale, mode):
        self.seed = seed
        self.scale = scale
        self.window = 0  # 在线数据读取中滑动窗口的长度
        self.win_df = pd.DataFrame()  # 在线数据读取中始终维持一个长度为最大滑动窗口的dataframe
        self.mode = mode  # 离线或在线
        self.pre_label = 0  # 在类属性中维护前一时刻预测的标签，供在线预测使用

    @staticmethod
    def _curvature(x, y):  # 计算某一点曲率半径，在FE中调用
        '''
        input  : 三个二维点(x1,y1)(x2,y2)(x3,y3)的坐标列表，形如x = (x1,x2,x3),y = (y1,y2,y3)
        output : 点(x2,y2)处的曲率半径
        '''
        t_a = la.norm([x[1] - x[0], y[1] - y[0]])
        t_b = la.norm([x[2] - x[1], y[2] - y[1]])

        M = np.array([[1, -t_a, t_a**2], [1, 0, 0], [1, t_b, t_b**2]])

        a = np.matmul(la.inv(M), x)
        b = np.matmul(la.inv(M), y)

        kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1]**2. + b[1]**2.)**(1.5)

        return kappa

    @staticmethod
    def _geodistance(lng1, lat1, lng2, lat2):  # 给定经纬度计算两点间距离(米)
        lng1, lat1, lng2, lat2 = map(
            radians,
            [float(lng1), float(lat1),
             float(lng2), float(lat2)])
        # 经纬度转换成弧度
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
        distance = round(distance / 1000, 3)
        return distance

    def FE(self, data_):  # 离线特征工程
        """ AtoG特征工程
        Args:
            data (dataframe): 原始数据
        Returns:
            DataFrame: 特征工程后数据
        """
        data = data_.copy(deep=True)

        if self.mode == 'offline':
            data = data.sort_values(by=['id', 'Time'])
            if self.scale == 'all':  # 全量模型
                # 本机长度为window滑动窗口均值、最大值、最小值、标准差
                for f in [
                        'speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll'
                ]:
                    data[f'rolling_6_{f}_mean'] = data.groupby([
                        'id'
                    ])[f].rolling(window=6,
                                  min_periods=1).mean().reset_index(drop=True)
                for f in [
                        'speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll'
                ]:
                    data[f'rolling_6_{f}_max'] = data.groupby([
                        'id'
                    ])[f].rolling(window=6,
                                  min_periods=1).max().reset_index(drop=True)
                for f in [
                        'speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll'
                ]:
                    data[f'rolling_6_{f}_min'] = data.groupby([
                        'id'
                    ])[f].rolling(window=6,
                                  min_periods=1).min().reset_index(drop=True)
                for f in [
                        'speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch', 'roll'
                ]:
                    data[f'rolling_6_{f}_std'] = data.groupby([
                        'id'
                    ])[f].rolling(window=6,
                                  min_periods=1).std().reset_index(drop=True)
                '''
                导弹特征工程
                '''
                cols = [
                    'position_x', 'position_y', 'position_z', 'longitude',
                    'dimensionality', 'high', 'yaw', 'pitch', 'roll'
                ]

                # 导弹长度为window滑动窗口均值、最大值、最小值、标准差
                # 使用用rolling函数求得
                for f in cols:
                    data[f'Missile_1_rolling_3_{f}_mean'] = data.groupby([
                        'id'
                    ])[f'Missile_1_{f}'].rolling(
                        window=3, min_periods=1).mean().reset_index(drop=True)
                    data[f'Missile_1_rolling_3_{f}_max'] = data.groupby(
                        ['id'])[f'Missile_1_{f}'].rolling(
                            window=3,
                            min_periods=1).max().reset_index(drop=True)
                    data[f'Missile_1_rolling_3_{f}_min'] = data.groupby(
                        ['id'])[f'Missile_1_{f}'].rolling(
                            window=3,
                            min_periods=1).min().reset_index(drop=True)
                    data[f'Missile_1_rolling_3_{f}_std'] = data.groupby(
                        ['id'])[f'Missile_1_{f}'].rolling(
                            window=3,
                            min_periods=1).std().reset_index(drop=True)

                # 导弹与本机距离
                data['dist_to_1_Missile'] = data.apply(
                    lambda x: 0 if x['Missile_num'] == 0 else np.sqrt(
                        (x['Missile_1_position_x'] - x['Position_x'])**2 +
                        (x['Missile_1_position_y'] - x['Position_y'])**2 +
                        (x['Missile_1_position_z'] - x['Position_z'])**2),
                    axis=1)  # 注意加axis=1,否则报错
                '''
                导弹特征工程
                '''

                # 本机上一时刻的曲率半径
                pre_data_1 = data.shift(periods=1)  # 上一个时间点的数据
                pre_data_2 = data.shift(periods=2)  # 上两个时间点的数据
                merge_data = pd.DataFrame({
                    'pre2_pos_x':
                    pre_data_2['Position_x'],
                    'pre1_pos_x':
                    pre_data_1['Position_x'],
                    'now_pos_x':
                    data['Position_x'],
                    'pre2_pos_y':
                    pre_data_2['Position_y'],
                    'pre1_pos_y':
                    pre_data_1['Position_y'],
                    'now_pos_y':
                    data['Position_y']
                })
                data_list = np.array(merge_data).tolist()
                x_list = []
                y_list = []
                for member in data_list:
                    x_list.append(member[0:3])
                    y_list.append(member[3:6])
                curve_list = []
                for x, y in zip(x_list, y_list):
                    curve_list.append(self._curvature(x, y))
                data['curve'] = curve_list

                # 实现可变变量名
                names = locals()

                # 上window个时刻的数据
                for i in range(1, 3):
                    names['pre_data_%s' % i] = data.shift(periods=i)

                for f in [
                        'Position_x', 'Position_y', 'Position_z', 'speed_x',
                        'speed_y', 'speed_z', 'yaw', 'pitch', 'roll'
                ]:
                    for i in range(1, 3):
                        data[f'pre_{i}_{f}'] = names['pre_data_%s' % i][f]

                # 我机与目标距离
                target_pos = [42.246944444444445, 42.04805555555556]
                data['dist'] = data.apply(lambda x: self._geodistance(
                    target_pos[1], target_pos[0], x['MyLong'], x['MyLat']),
                                          axis=1)
            elif self.scale == 'light':  # 轻量模型
                # 实现可变变量名
                names = locals()
                # 上window个时刻的数据
                for i in range(1, 3):
                    names['pre_data_%s' % i] = data.shift(periods=i)

                for f in ['Position_x', 'Position_y']:
                    for i in range(1, 3):
                        data[f'pre_{i}_{f}'] = names['pre_data_%s' % i][f]
            data.fillna(0, inplace=True)

        elif self.mode == 'online':
            # 实现可变变量名
            names = locals()

            # 上window个时刻的数据
            for i in range(1, 3):
                names['pre_data_%s' % i] = data.shift(periods=i)

            for f in ['Position_x', 'Position_y']:
                for i in range(1, 3):
                    data[f'pre_{i}_{f}'] = names['pre_data_%s' % i][f]
            data.fillna(0, inplace=True)
        return data

    def online_FE(self, row):
        """ AtoG在线特征工程
        Args:
            row_dict(dict): 传入的一行字典记录
        Returns:
            DataFrame: 加入特征后的记录dataframe
        """
        # 窗口值加一
        self.window += 1
        # 将数据加入窗口
        self.win_df = self.win_df.append(row, ignore_index=True)
        # 窗口没有达到最大值，加入数据
        if self.window < 3:
            FE_row = self.FE(self.win_df).iloc[-1:]
            # 没有数据的用0填充
            FE_row.fillna(0, inplace=True)
        else:
            # 窗口达到最大值，获取第一行加入的数据
            FE_row = self.FE(self.win_df).iloc[-1:]
            FE_row.fillna(0, inplace=True)
            # 弹出第一行数据
            self.win_df.drop(0, inplace=True)
            # 重新设置下标
            self.win_df.reset_index(drop=True)
        return FE_row

    def train_val_split(self, df_train, percent=0.8):
        """ 数据集划分
        划分数据集为训练集与测试
        Args:
            df_train(dataframe): 原始数据
            percent(int): 切分比例
        Returns:
            train(dataframe): 训练集
            val_data(dataframe): 验证集
        """
        # 获取所有id
        all_ids = df_train['id'].values.tolist()
        all_ids = list(set(all_ids))
        # 每次 set 的结果都不一样，所以要先排序，防止结果不可复现
        all_ids.sort()
        # random.seed 只能生效一次，所以每次 random.sample 之前都要设置
        random.seed(self.seed)
        train_ids = random.sample(all_ids, int(len(all_ids) * percent))
        val_ids = list(set(all_ids) - set(train_ids))
        # 根据id获取训练数据
        train_data = df_train[df_train['id'].isin(train_ids)]
        # 根据id获取验证数据
        val_data = df_train[df_train['id'].isin(val_ids)]
        # 连续序列数据，但是是以单个样本建模的情况下，需要 shuffle 打乱

        train_data = train_data.sample(
            frac=1, random_state=self.seed).reset_index(drop=True)

        return train_data, val_data

    def _feature_name(self):
        """ 获取模型保留列名
        Returns:
            feature_names(list): 全量级列名信息
        """
        # 固定顺序，否则模型预测会出错
        if self.scale == 'all':  # 全量级模型
            feature_names = [
                'Time', 'speed_x', 'speed_y', 'speed_z', 'yaw', 'pitch',
                'roll', 'Position_x', 'Position_y', 'Position_z',
                'AngleAttack', 'MyLong', 'MyAlt', 'MyAngleSpeedx',
                'MyAngleSpeedy', 'MyAngleSpeedz', 'Missile_1_dimensionality',
                'Missile_1_high', 'Missile_1_longitude', 'Missile_1_pitch',
                'Missile_1_position_x', 'Missile_1_position_y',
                'Missile_1_position_z', 'Missile_1_roll', 'Missile_1_yaw',
                'Missile_num', 'rolling_6_speed_x_mean',
                'rolling_6_speed_y_mean', 'rolling_6_speed_z_mean',
                'rolling_6_yaw_mean', 'rolling_6_pitch_mean',
                'rolling_6_roll_mean', 'rolling_6_speed_x_max',
                'rolling_6_speed_y_max', 'rolling_6_speed_z_max',
                'rolling_6_yaw_max', 'rolling_6_pitch_max',
                'rolling_6_roll_max', 'rolling_6_speed_x_min',
                'rolling_6_speed_y_min', 'rolling_6_speed_z_min',
                'rolling_6_yaw_min', 'rolling_6_pitch_min',
                'rolling_6_roll_min', 'rolling_6_speed_x_std',
                'rolling_6_speed_y_std', 'rolling_6_speed_z_std',
                'rolling_6_yaw_std', 'rolling_6_roll_std',
                'Missile_1_rolling_3_position_x_mean',
                'Missile_1_rolling_3_position_x_max',
                'Missile_1_rolling_3_position_x_min',
                'Missile_1_rolling_3_position_x_std',
                'Missile_1_rolling_3_position_y_mean',
                'Missile_1_rolling_3_position_y_max',
                'Missile_1_rolling_3_position_y_min',
                'Missile_1_rolling_3_position_y_std',
                'Missile_1_rolling_3_position_z_mean',
                'Missile_1_rolling_3_position_z_max',
                'Missile_1_rolling_3_position_z_min',
                'Missile_1_rolling_3_position_z_std',
                'Missile_1_rolling_3_longitude_mean',
                'Missile_1_rolling_3_longitude_max',
                'Missile_1_rolling_3_longitude_min',
                'Missile_1_rolling_3_longitude_std',
                'Missile_1_rolling_3_dimensionality_mean',
                'Missile_1_rolling_3_dimensionality_max',
                'Missile_1_rolling_3_dimensionality_min',
                'Missile_1_rolling_3_dimensionality_std',
                'Missile_1_rolling_3_high_mean',
                'Missile_1_rolling_3_high_max', 'Missile_1_rolling_3_high_min',
                'Missile_1_rolling_3_high_std', 'Missile_1_rolling_3_yaw_mean',
                'Missile_1_rolling_3_yaw_max', 'Missile_1_rolling_3_yaw_min',
                'Missile_1_rolling_3_yaw_std',
                'Missile_1_rolling_3_pitch_mean',
                'Missile_1_rolling_3_pitch_max',
                'Missile_1_rolling_3_pitch_min',
                'Missile_1_rolling_3_roll_mean',
                'Missile_1_rolling_3_roll_max', 'Missile_1_rolling_3_roll_min',
                'dist_to_1_Missile', 'curve', 'pre_1_Position_x',
                'pre_2_Position_x', 'pre_1_Position_y', 'pre_2_Position_y',
                'pre_1_Position_z', 'pre_2_Position_z', 'pre_1_speed_x',
                'pre_2_speed_x', 'pre_1_speed_y', 'pre_2_speed_y',
                'pre_1_speed_z', 'pre_2_speed_z', 'pre_1_yaw', 'pre_2_yaw',
                'pre_1_pitch', 'pre_2_pitch', 'pre_1_roll', 'pre_2_roll',
                'dist'
            ]
        elif self.scale == 'light':  # 轻量级模型
            feature_names = [
                'pre_2_Position_x', 'pre_2_Position_y', 'pre_1_Position_x',
                'pre_1_Position_y', 'Position_y', 'Position_x', 'MyAlt',
                'AngleAttack'
            ]
        return feature_names

    # 留出法数据
    def _hold_out(self, raw_train, percent_train):
        """ 获取留出法的训练数据
        Args:
            raw_train(dataframe): 原始数据
            percent_train(int): 训练集占比
        Returns:
            train(dataframe): 训练集
            val(dataframe): 验证集
        """

        # 获取保留列名
        feature_names = self._feature_name()
        # 切分训练集、验证集
        train_data, val_data = self.train_val_split(raw_train,
                                                    percent=percent_train)
        # 获取训练验证数据和标签数据
        X_train, X_val, y_train, y_val = train_data[feature_names], val_data[
            feature_names], train_data['label'], val_data['label']

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
        # 获取保留列名
        feature_names = self._feature_name()

        # 根据id分组
        groups = list(raw_train['id'])
        # 分组交叉验证
        gkf = GroupKFold(n_splits=k)
        data_list = []
        # 获取交叉验证数据
        for train_index, val_index in gkf.split(raw_train[feature_names],
                                                raw_train['label'],
                                                groups=groups):
            # 根据index索引获取每一折数据
            X_train, y_train, X_val, y_val = raw_train.iloc[train_index][feature_names], \
                                            raw_train.iloc[train_index]['label'], \
                                            raw_train.iloc[val_index][feature_names], \
                                            raw_train.iloc[val_index]['label']
            # 将数据加入列表保存
            data_list.append([X_train, X_val, y_train, y_val])
        # 返回列表
        return data_list

    # bootstrap验证数据
    def _bootstrap(self, raw_train):
        """ 获取提升法数据
        Args:
            raw_train(dataframe): 原始数据
        Returns:
            train(dataframe): 提升法训练集
            val(dataframe): 验证集
        """
        # 获取保留列名
        feature_names = self._feature_name()
        # 获取所有数据id，并去重
        ids = pd.DataFrame(set(raw_train['id']), columns=['id'], index=None)
        random.seed(self.seed)
        # 根据id采样
        train_group_ids = ids.sample(frac=1.0,
                                     replace=True,
                                     random_state=self.seed)
        # 总id减去训练集的id，得到验证集id
        val_group_ids = ids.loc[ids.index.difference(
            train_group_ids.index)].copy()
        # 创建两个dataframe
        train_data = pd.DataFrame()
        val_data = pd.DataFrame()
        # 获取训练与验证数据id号
        train_group_ids = list(train_group_ids['id'])
        val_group_ids = list(val_group_ids['id'])
        # 根据id获取数据
        for train_group_id in train_group_ids:
            train_data = train_data.append(
                raw_train[raw_train['id'] == train_group_id])
        for val_group_id in val_group_ids:
            val_data = val_data.append(
                raw_train[raw_train['id'] == val_group_id])
        # 切分训练数据与真实标签
        X_train, X_val, y_train, y_val = train_data[feature_names], val_data[
            feature_names], train_data['label'], val_data['label']
        return X_train, X_val, y_train, y_val

    # 定义lightgbm模型
    def _lgb(self):
        lgb_model = lgb.LGBMClassifier(boosting_type='gbdt',
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
        return lgb_model

    # 定义svm模型
    @staticmethod
    def _svm():
        svm_model = svm.SVC(C=1.0,
                            kernel='rbf',
                            degree=3,
                            gamma='auto',
                            coef0=0.0,
                            shrinking=True,
                            probability=True,
                            tol=0.001,
                            class_weight=None,
                            verbose=True,
                            max_iter=10000,
                            decision_function_shape='ovr',
                            random_state=None)
        return svm_model

    # 定义集成模型
    @staticmethod
    def _ensemble():
        clf1 = LogisticRegression(random_state=1)
        clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
        clf3 = GaussianNB()

        ensemble_model = VotingClassifier(estimators=[('lr', clf1),
                                                      ('rf', clf2),
                                                      ('gnb', clf3)],
                                          voting='soft')

        return ensemble_model

    # 训练lightgbm模型
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
        # 获取保留列名
        feature_names = self._feature_name()
        if val_type == 'hold-out':  # 留出法
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            lgb_model = self._lgb()
            # 模型训练，使用早停策略
            lgb_model.fit(X_train,
                          y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          verbose=100,
                          early_stopping_rounds=50)
            # 获取特征重要性
            df_importance = pd.DataFrame({
                'column':
                feature_names,
                'importance':
                lgb_model.feature_importances_,
            }).sort_values(by='importance',
                           ascending=False).reset_index(drop=True)
            # 最大最小归一
            df_importance['importance'] = (
                df_importance['importance'] -
                df_importance['importance'].min()) / (
                    df_importance['importance'].max() -
                    df_importance['importance'].min())

            return lgb_model, df_importance

        elif val_type == 'k-fold':
            # 创建模型保存列表
            model_list = []
            # 创建重要性保存列表
            importance_list = []
            # 获取k折数据
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 获取lgb模型并且编号
                names['lgb_%s' % i] = self._lgb()
                # 模型训练，使用早停策略
                names['lgb_%s' % i].fit(X_train,
                                        y_train,
                                        eval_names=['train', 'valid'],
                                        eval_set=[(X_train, y_train),
                                                  (X_val, y_val)],
                                        verbose=100,
                                        early_stopping_rounds=50)
                # 获取特征重要性
                df_importance = pd.DataFrame({
                    'column':
                    feature_names,
                    'importance':
                    names['lgb_%s' % i].feature_importances_,
                }).sort_values(by='importance',
                               ascending=False).reset_index(drop=True)
                # 保存k个模型
                model_list.append(names['lgb_%s' % i])
                # 保存k个重要性
                importance_list.append(df_importance)

            mean_dict = dict()
            # 获取平均特征重要度
            for feat in feature_names:
                mean_dict[feat] = 0

            for df_importance in importance_list:
                for feat in feature_names:
                    mean_dict[feat] += int(
                        df_importance[df_importance['column'] ==
                                      feat]['importance'].values[0])

            for feat in feature_names:
                mean_dict[feat] /= k
            # 重要度排序
            mean_imp_df = pd.DataFrame({
                'column': feature_names,
                'importance': list(mean_dict.values()),
            }).sort_values(by='importance',
                           ascending=False).reset_index(drop=True)
            # 最大最小归一
            mean_imp_df['importance'] = (
                df_importance['importance'] -
                df_importance['importance'].min()) / (
                    df_importance['importance'].max() -
                    df_importance['importance'].min())

            return model_list, mean_imp_df

        elif val_type == 'bootstrap':
            # 获取提升法数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取lgb模型
            lgb_model = self._lgb()
            # 模型训练，使用早停策略
            lgb_model.fit(X_train,
                          y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          verbose=100,
                          early_stopping_rounds=50)
            # 获取模型的特征重要性特征
            df_importance = pd.DataFrame({
                'column':
                feature_names,
                'importance':
                lgb_model.feature_importances_,
            }).sort_values(by='importance',
                           ascending=False).reset_index(drop=True)
            # 最大最小归一
            df_importance['importance'] = (
                df_importance['importance'] -
                df_importance['importance'].min()) / (
                    df_importance['importance'].max() -
                    df_importance['importance'].min())

            return lgb_model, df_importance

        else:
            print("无该验证方法!")
            exit(-1)

    # 训练朴素贝叶斯模型
    def _train_nb(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练朴素贝叶斯模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
        """
        if val_type == 'hold-out':  # 留出法
            # 获取训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取朴素贝叶斯模型
            gnb_model = GaussianNB()
            # 模型训练
            gnb_model.fit(X_train, y_train)
            return gnb_model, None

        elif val_type == 'k-fold':
            # 创建模型保存列表
            model_list = []
            # 获取k折交叉验证数据
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 定义模型
                names['gnb_%s' % i] = GaussianNB()
                # 模型训练
                names['gnb_%s' % i].fit(X_train, y_train)
                # 模型加入列表
                model_list.append(names['gnb_%s' % i])

            return model_list, None

        elif val_type == 'bootstrap':
            # 提升法获取训练、验证数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取模型
            gnb_model = GaussianNB()
            # 模型训练
            gnb_model.fit(X_train, y_train)
            return gnb_model, None
        else:
            print("无该验证方法!")
            exit(-1)

    # 训练线性回归模型
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

        if val_type == 'hold-out':  # 留出法
            # 获取训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取线性回归模型
            linear_model = LinearRegression()
            # 模型训练
            linear_model.fit(X_train, y_train)
            return linear_model, None

        elif val_type == 'k-fold':
            # 创建模型保存列表
            model_list = []
            # k折数据
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 获取模型并编号
                names['linear_%s' % i] = LinearRegression()
                # 模型训练
                names['linear_%s' % i].fit(X_train, y_train)
                # 模型加入列表
                model_list.append(names['linear_%s' % i])

            return model_list, None

        elif val_type == 'bootstrap':
            # 提升法获取训练、验证数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 模型训练
            linear_model = LinearRegression()
            # 模型预测
            linear_model.fit(X_train, y_train)
            return linear_model, None
        else:
            print("无该验证方法!")
            exit(-1)

    # 训练逻辑回归模型
    def _train_logisticReg(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练逻辑回归模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
        """

        if val_type == 'hold-out':  # 留出法
            # 获取训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取逻辑回归模型
            logistic_model = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
            # 模型训练
            logistic_model.fit(X_train, y_train)
            return logistic_model, None

        elif val_type == 'k-fold':
            # 模型保存列表
            model_list = []
            # k交叉数据
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 创建模型并命名
                names['logistic_%s' % i] = LogisticRegression(C=1.0,
                                                              penalty='l2',
                                                              tol=0.01)
                # 模型训练
                names['logistic_%s' % i].fit(X_train, y_train)
                # 加入模型列表
                model_list.append(names['logistic_%s' % i])
            return model_list, None
        elif val_type == 'bootstrap':
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            logistic_model = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
            logistic_model.fit(X_train, y_train)
            return logistic_model, None
        else:
            print("无该验证方法!")
            exit(-1)

    # 训练支持向量机模型
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

        if val_type == 'hold-out':  # 留出法
            # 提升法获取训练、验证数据
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取模型
            svm_model = self._svm()
            # 模型训练
            svm_model.fit(X_train, y_train)

            return svm_model, None

        elif val_type == 'k-fold':
            model_list = []
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                names['svm_%s' % i] = self._svm()
                # 模型训练
                names['svm_%s' % i].fit(X_train, y_train)
                # 模型加入列表
                model_list.append(names['svm_%s' % i])
            return model_list, None
        elif val_type == 'bootstrap':
            # 获取数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取模型
            svm_model = self._svm()
            # 模型训练
            svm_model.fit(X_train, y_train)
            return svm_model, None

        else:
            print("无该验证方法!")
            exit(-1)

    # 训练集成模型
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

        if val_type == 'hold-out':  # 留出法
            # 留出法获得训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取集成模型
            ensemble_model = self._ensemble()
            # 模型训练
            ensemble_model.fit(X_train, y_train)

            return ensemble_model, None

        elif val_type == 'k-fold':
            # 模型保存列表
            model_list = []
            # k折交叉数据
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 获取模型并命名
                names['ensemble_%s' % i] = self._ensemble()
                # 模型训练
                names['ensemble_%s' % i].fit(X_train, y_train)
                # 模型加入列表
                model_list.append(names['ensemble_%s' % i])

            return model_list, None

        elif val_type == 'bootstrap':
            # 获取数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取模型
            ensemble_model = self._ensemble()
            # 模型训练
            ensemble_model.fit(X_train, y_train)
            return ensemble_model, None

        else:
            print("无该验证方法!")
            exit(-1)

    @staticmethod
    def _renotation_util(group_, per0, per1, per2, per3, per4):  # 去随机标注工具函数
        """ 去除随机标注工具函数
        Args:
            group_(dataframe): 一批数据
            per0(int): 无动删除比例
            per1(int): 搜索删除比例
            per2(int): 瞄准删除比例
            per3(int): 攻击删除比例
            per4(int): 脱离删除比例
        Returns:
            group(dataframe): 删除随机标注后的一批数据
        """
        group = group_.copy(deep=True)
        group.index = range(1, len(group) + 1)  # 设置index从1开始
        label = list(group['label'])
        # 获取标签转换结点
        loc = [label.index(1), label.index(2), label.index(3), label.index(4)]
        # 获取五类标签长度
        len0 = loc[0]
        len1 = loc[1] - loc[0]
        len2 = loc[2] - loc[1]
        len3 = loc[3] - loc[2]
        len4 = len(label) - loc[3] + 1
        # 定义要删除的索引
        drop = list()
        # 无动(后段)、搜索(前段)要删除的索引
        list0 = [
            i for i in range(loc[0] - int(len0 * per0), loc[0] +
                             int(len1 * per1) + 1)
        ]
        # 搜索(后段)、瞄准(前段)要删除的索引
        list1 = [
            i for i in range(loc[1] - int(len1 * per1), loc[1] +
                             int(len2 * per2) + 1)
        ]
        # 瞄准(后段)、攻击(前段)要删除的索引
        list2 = [
            i for i in range(loc[2] - int(len2 * per2), loc[2] +
                             int(len3 * per3) + 1)
        ]
        # 攻击(后段)、脱离(前段)要删除的索引
        list3 = [
            i for i in range(loc[3] - int(len3 * per3), loc[3] +
                             int(len4 * per4) + 1)
        ]
        # 加入各段索引
        drop.extend(list0)
        drop.extend(list1)
        drop.extend(list2)
        drop.extend(list3)
        # 删除随机标注
        group = group.drop(drop)
        return group

    def renotation(self, data, per0, per1, per2, per3, per4):  # 去随机标注
        """ 去除随机标注主函数
        Args:
            data(dataframe): 多批数据
            per0(int): 无动删除比例
            per1(int): 搜索删除比例
            per2(int): 瞄准删除比例
            per3(int): 攻击删除比例
            per4(int): 脱离删除比例
        Returns:
            renotation_data(dataframe): 删除随机标注后的多批数据
        """

        renotation_data = pd.DataFrame()
        # 根据id分组
        grouped = data.groupby('id')
        # 遍历分组
        for ids, group_data in grouped:
            try:
                # 调用工具函数
                renotation_group = self._renotation_util(
                    group_data, per0, per1, per2, per3, per4)
                renotation_data = renotation_data.append(renotation_group)
            except ValueError:
                # 原始数据有标签不连续的情况
                print(f'{ids}批标签不连续！')
                continue
        return renotation_data

    # 模型训练
    def train_model(self, train, model_type, val_type, k=0, percent_train=0.8):
        """ 模型训练
        Args:
            train(dataframe): 原始数据
            model_type(string): 模型类型
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
            importance(dataframe): 特征重要度
            best_thread(float): 最佳阈值
        """
        # lightgbm模型
        if model_type == 'lgb':
            model, df_importance = self._train_lgb(train,
                                                   val_type,
                                                   k,
                                                   percent_train=percent_train)
            return model, df_importance
        # 朴素贝叶斯
        elif model_type == 'nb':
            model, df_importance = self._train_nb(train,
                                                  val_type,
                                                  k,
                                                  percent_train=percent_train)
            return model, df_importance
        # 线性模型
        elif model_type == 'linear':
            model, df_importance = self._train_linearReg(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance
        # 逻辑回归
        elif model_type == 'logistic':
            model, df_importance = self._train_logisticReg(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance
        # 支持向量机
        elif model_type == 'svm':
            model, df_importance = self._train_svm(train,
                                                   val_type,
                                                   k,
                                                   percent_train=percent_train)
            return model, df_importance
        # 集成模型
        elif model_type == 'ensemble':
            model, df_importance = self._train_ensemble(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance
        else:
            print('无该模型！')
            exit(-1)

    @staticmethod
    def save_model(model, save_dir, name, val_type):
        """ 模型保存
        Args:
            model: 模型
            save_dir(string): 保存路径
            name(string): 模型名称
            val_type(string): 验证方式
            best_thread(float): 阈值
        Returns:
            None
        """
        # 如果是k折交叉验证，保存k个模型
        if val_type == 'k-fold':
            # 遍历k个模型
            for i in range(len(model)):
                # 保存路径
                save_path = os.path.join(
                    save_dir, name + str(i) + '_' + val_type + '.pkl')
                # 模型保存
                joblib.dump(model[i], save_path)
        else:
            # 非交叉验证模型保存
            save_path = os.path.join(save_dir, name + '_' + val_type + '.pkl')
            joblib.dump(model, save_path)

    @staticmethod
    def load_model(path):
        """ 加载模型
        Args:
            path(string): 模型路径
        Returns:
            None
        """
        # 模型加载
        model = joblib.load(path)
        return model

    def predict_util(self, model, row, pre_label, feature_names):
        """ 离线预测工具函数
        Args:
            model: 模型
            row(dataframe): 一行数据
            pre_label(int): 前一标签
            feature_names(list):保留用于预测的特征名
        Returns:
            pred_prob(np.1darray or list): 预测概率
            mask_prob(np.1darray or list): 去除标签穿越后的预测概率
            pred_label(np.1darray or list): 预测标签
        """

        try:
            # 非线性模型预测
            pred_prob = model.predict_proba(
                np.array(row[feature_names]).reshape(1, -1))
            # 预防标签穿越
            mask = [1 if (0 <= i - pre_label <= 1) else 0 for i in range(5)]

            mask_prob = mask * pred_prob
            # 获取预测标签
            pred_label = mask_prob.argmax(axis=1)
            return pred_prob[0], mask_prob[0], pred_label
            # 一个坑,将pred_prob,mask_prob从二维转成一维

        except AttributeError:
            # 处理线性模型，情况较为复杂
            pred_label = model.predict(
                np.array(row[feature_names]).reshape(1, -1))
            # 由于是线性回归，会有预测值不在[0,4]范围内的情况
            pred_label = round(float('{:.1f}'.format(pred_label[0])))
            if pred_label > 4:
                pred_label = 4
            if pred_label < 0:
                pred_label = 0
            # 处理穿越
            if pred_label < self.pre_label:
                pred_label = self.pre_label

            # 根据预测结果反推两类预测概率(为了参数统一,实际预测不返回)
            pred_prob = [0 if i != pred_label else 1 for i in range(5)]
            mask_prob = [0 if i != pred_label else 1 for i in range(5)]

            return pred_prob, mask_prob, pred_label

    def offline_predict(self, model, df_test):
        """ 离线预测
        Args:
            model: 模型
            df_test(dataframe): 测试集
            threshold(float): 最佳阈值
        Returns:
            pred_prob(dataframe): 预测概率
            pred_label(dataframe): 预测标签
        """
        # 获取需要的列名
        feature_names = self._feature_name()
        # 去除穿越后的预测概率
        mask_prob_list = list()
        # 用于存储原来的五类概率，因在画roc时要求五类概率之和必须为1
        prob_list = list()
        # 用于存储预测标签
        pred_label_list = list()
        # k折交叉验证有加操作,故特别定义为np.array
        kfold_mask_prob = np.array([[0 for _ in range(5)]
                                    for _ in range(len(df_test))])
        # k折交叉验证有加操作,故特别定义为np.array
        kfold_prob = np.array([[0 for _ in range(5)]
                               for _ in range(len(df_test))])
        # k折交叉验证预测
        if isinstance(model, list):
            # 遍历五个模型
            for i in range(len(model)):
                # 暂时存储mask_prob与原始prob
                ith_mask_list = list()
                ith_prob_list = list()
                print(f'第{i + 1}个模型开始预测')
                # 遍历分组
                for _, group in df_test.groupby('id'):
                    # 前一标签从0开始
                    pre_label = 0
                    # 遍历一批数据
                    for _, row in group.iterrows():
                        # 为了避免穿越,离线也用在线方式预测
                        prob, mask_prob, pred_label = self.predict_util(
                            model[i], row, pre_label, feature_names)
                        # 更新前一标签
                        pre_label = pred_label
                        # 第k折验证加入一行数据的预测信息
                        ith_mask_list.append(mask_prob)
                        ith_prob_list.append(prob)
                # 对k折累加
                kfold_mask_prob = np.add(kfold_mask_prob,
                                         np.array(ith_mask_list))
                kfold_prob = np.add(kfold_prob, np.array(ith_prob_list))
                print(f'第{i + 1}个模型预测完毕')

            # 对k折取平均
            kfold_mask_prob = np.true_divide(kfold_mask_prob, len(model))
            kfold_prob = np.true_divide(kfold_prob, len(model))
            # 用求平均后的概率预测标签
            pred_label_list = np.argmax(kfold_mask_prob, axis=1)
            return kfold_prob, kfold_mask_prob, pred_label_list
        else:
            # 非k折交叉验证，遍历分组
            for _, group in df_test.groupby('id'):
                # 前一标签从0开始
                pre_label = 0
                # 每一组遍历每行
                for _, row in group.iterrows():
                    # 用在线的方式预测一行
                    prob, mask_prob, pred_label = self.predict_util(
                        model, row, pre_label, feature_names)
                    # 更新前一标签
                    pre_label = pred_label
                    # 加入一行的预测信息
                    prob_list.append(prob)
                    mask_prob_list.append(mask_prob)
                    pred_label_list.append(pred_label)

            return prob_list, mask_prob_list, pred_label_list

    def online_predict(self, model, row):  # 在线预测
        """ 在线预测
        Args:
            model: 模型
            row(dict): 一行数据
            threshold(float): 最佳阈值
        Returns:
            pred_prob(float): 预测概率
            pred_label(int): 预测标签
        """

        # 获取需要的列名
        feature_names = self._feature_name()

        try:
            # 如果非线性模型预测方式
            pred_prob = model.predict_proba(
                np.array(row[feature_names]).reshape(1, -1))
            # 防止标签穿越,定义mask
            mask = [
                1 if (0 <= i - self.pre_label <= 1) else 0 for i in range(5)
            ]
            # 获取非穿越预测概率
            mask_prob = mask * pred_prob
            # 根据非穿越预测概率获取预测标签
            pred_label = mask_prob.argmax(axis=1)
            # 更新前一标签(定义为类属性)
            self.pre_label = pred_label
            return pred_prob[0], mask_prob[0], pred_label[0]
            # 一个坑,将pred_prob, mask_prob从二维转成一维

        except AttributeError:  # 处理线性模型，情况较为复杂
            pred_label = model.predict(
                np.array(row[feature_names]).reshape(1, -1))

            # 由于是线性回归，会有预测值不在[0,4]范围内的情况
            pred_label = round(float('{:.1f}'.format(pred_label[0])))
            if pred_label > 4:
                pred_label = 4
            if pred_label < 0:
                pred_label = 0
            # 处理穿越
            if pred_label < self.pre_label:
                pred_label = self.pre_label

            # 更新前一标签
            self.pre_label = pred_label

            # 根据预测标签获取两类预测概率
            pred_prob = [0 if i != pred_label else 1 for i in range(5)]
            mask_prob = [0 if i != pred_label else 1 for i in range(5)]

            return pred_prob, mask_prob, pred_label

    def score(self, label, pred_prob, pred_label, name, val_type):
        """ 评分函数
        Args:
            label: 真实标签
            pred_prob: 预测概率
            pred_label: 预测标签
            name:模型名称
            val_type：验证方式
        Returns:
            dict: 评分字典
        """
        # 获得评价指标
        cr = classification_report(label, pred_label, digits=4)
        print(cr)

        # 获取auc值
        auc = roc_auc_score(label, pred_prob, multi_class='ovo')  # 注意设置多分类参数
        print('auc:', auc)

        # 获取acc值
        acc = accuracy_score(label, pred_label)
        print('acc:', acc)

        roc_title = f'AtoG offline {name} {val_type} ROC Curve'
        roc = self._plot_roc(label, pred_prob, roc_title)
        # 获取pr曲线
        pr_title = f'AtoG offline {name} {val_type} PR Curve'
        pr = self._plot_pr(label, pred_prob, pr_title)

        # 混淆矩阵
        cm = confusion_matrix(label, pred_label)
        print("---------------混淆矩阵\n", cm)

        # 评价字典
        dictionary = {'auc': auc, 'cr': cr, 'acc': acc, 'roc': roc, 'pr': pr}
        return dictionary

    @staticmethod
    def _plot_roc(label, pred_prob, roc_title):
        """ roc画图
        Args:
            label: 真实标签
            pred_prob: 预测概率
        Returns:
            fig: roc图
        """
        # 画ROC
        fig = plt.figure()
        # 因是多分类,独热编码成两类进行评价
        label_one_hot = label_binarize(label, classes=[0, 1, 2, 3, 4])
        fpr_test, tpr_test, _ = roc_curve(label_one_hot.ravel(),
                                          np.array(pred_prob).ravel())
        plt.plot(fpr_test, tpr_test)
        plt.plot([0, 1], [0, 1], 'd--')
        # x轴名称
        plt.xlabel('False Positive Rate')
        # y轴名称
        plt.ylabel('True Positive Rate')
        # 图名称
        plt.title(roc_title)

        return fig

    @staticmethod
    def _plot_pr(label, pred_prob, pr_title):
        """ pr画图
        Args:
            label: 真实标签
            pred_prob: 预测概率
        Returns:
            fig: pr图
        """
        # 画PR
        fig = plt.figure()
        # 因是多分类,独热编码成两类进行评价
        label_one_hot = label_binarize(label, classes=[0, 1, 2, 3, 4])
        precision, recall, _ = precision_recall_curve(
            label_one_hot.ravel(),
            np.array(pred_prob).ravel())
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        # x轴名称
        plt.xlabel('Recall')
        # y轴名称
        plt.ylabel('Precision')
        # 设置xy轴范围
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

        # 计算AP
        AP = average_precision_score(label_one_hot.ravel(),
                                     np.array(pred_prob).ravel(),
                                     average='macro',
                                     pos_label=1,
                                     sample_weight=None)
        # 图名称
        plt.title(pr_title + ':AP={0:0.2f}'.format(AP))
        return fig

    def evaluate(self, save_dir, dictionary, name, val_type):
        """ 评估保存函数
        Args:
            save_dir(string): 保存路径
            dictionary(dict): 字典
            name(string): 模型名称
            val_type(string): 验证方式
        Returns:
            None
        """
        # roc图
        plt_roc = dictionary['roc']
        # pr图
        pr_roc = dictionary['pr']
        # 保存roc图
        plt_roc.savefig(
            os.path.join(save_dir, name + '_' + val_type + '_roc.png'))
        # 保存pr图
        pr_roc.savefig(
            os.path.join(save_dir, name + '_' + val_type + '_pr.png'))
        # 保存评价指标，形成txt文件
        with open(os.path.join(save_dir, name + '_' + val_type + '_score.txt'),
                  'a+',
                  encoding='utf-8') as fr:
            line = '模型{}的'.format(
                name) + self.mode + ' auc为{:.4f},acc为{:.4f}\n'.format(
                    dictionary['auc'], dictionary['acc'])
            # 一行数据写入文件
            fr.write(line)
            # 评估报告写入文件
            fr.write(self.mode + '评估报告为:\n')
            # 评估字典写入文件
            fr.write(dictionary['cr'])
