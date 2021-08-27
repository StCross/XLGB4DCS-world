import os
import random
import sys

import joblib
import math
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import keras.callbacks as kcallbacks
from utilClass import RocAucMetricCallback
from utils import series_to_supervised
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

sys.path.append('..')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class AtoA:
    def __init__(self, mode, type='df', seed=2021, scale='all'):
        self.seed = seed
        self.window = 0  # 在线数据读取中滑动窗口的长度
        self.win_df = pd.DataFrame()  # 在线数据读取中始终维持一个长度为最大滑动窗口的dataframe
        self.mode = mode  # 离线或在线
        self.type = type  # df 或者 dcs
        self.scale = scale  # 模型量级

    # dogfight特征工程工具函数
    def FE_DF(self, data):
        """ DF特征工程
        Args:
            data (dataframe): 原始数据
        Returns:
            DataFrame: 特征工程后数据
        """
        data = data.sort_values(by=['id'])
        if self.scale == 'all':
            # 计算敌机的速度，先用diffh函数得到和上一时刻xyz差值，然后除以时间得到速度
            for f in ['x', 'y', 'z']:
                data['enemy_v_{}'.format(f)] = data.groupby('id')[
                    'enemy_{}'.format(f)].diff(1) / 0.02
            # 敌我两机加速度，先用diffh函数得到和上一时刻v_x,v_y,v_z差值，然后除以时间得到加速度
            for f in ['v_x', 'v_y', 'v_z']:
                data[f'my_{f}_acc'] = data.groupby(
                    'id')[f'my_{f}'].diff() / 0.2
                data[f'enemy_{f}_acc'] = data.groupby(
                    'id')[f'enemy_{f}'].diff() / 0.2
            # 敌我两机速度与位置交互式差值
            for f in ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']:
                data[f'{f}_me_minus'] = data[f'my_{f}'] - data[f'enemy_{f}']

        # 飞机之间的距离
        data['distance'] = ((data['my_x'] - data['enemy_x'])**2 +
                            (data['my_y'] - data['enemy_y'])**2 +
                            (data['my_z'] - data['enemy_z'])**2)**0.5

        # 瞄准夹角
        data['cos'] = ((data['my_v_x'] * (data['enemy_x'] - data['my_x'])) +
                       (data['my_v_y'] * (data['enemy_y'] - data['my_y'])) +
                       (data['my_v_z'] * (data['enemy_z'] - data['my_z'])))
        # 合速度
        data['speedAll'] = ((data['my_v_x']**2 + data['my_v_y']**2 +
                             data['my_v_z']**2)**0.5)
        # 夹角cos值
        data['cosValue'] = data['cos'] / (data['speedAll'] * data['distance'])
        # 缺失值补0
        data.fillna(0, inplace=True)

        return data

    def online_FE_DF(self, row_dict):
        """ DF在线特征工程
        Args:
            row_dict(dict): 传入的一行字典记录
        Returns:
            DataFrame: 加入特征后的记录dataframe
        """
        # 将字典转为dataframe格式
        data = pd.DataFrame(row_dict, index=[0])
        # 飞机之间的距离
        data['distance'] = ((data['my_x'] - data['enemy_x'])**2 +
                            (data['my_y'] - data['enemy_y'])**2 +
                            (data['my_z'] - data['enemy_z'])**2)**0.5

        # 瞄准夹角
        data['cos'] = ((data['my_v_x'] * (data['enemy_x'] - data['my_x'])) +
                       (data['my_v_y'] * (data['enemy_y'] - data['my_y'])) +
                       (data['my_v_z'] * (data['enemy_z'] - data['my_z'])))
        # 合速度
        data['speedAll'] = ((data['my_v_x']**2 + data['my_v_y']**2 +
                             data['my_v_z']**2)**0.5)
        # 夹角cos值
        data['cosValue'] = data['cos'] / (data['speedAll'] * data['distance'])
        # 缺失值补0
        data.fillna(0, inplace=True)
        return data

    # DCS特征工程工具函数
    def FE_DCS(self, data_):
        """ DCS特征工程
        Args:
            data (dataframe): 原始数据
        Returns:
            DataFrame: 特征工程后数据
        """
        data = data_.copy(deep=True)
        if self.mode == 'offline':
            # 如果是离线训练，需要根据id进行数据分组
            data = data.sort_values(by=['id'])

        # 飞机之间的距离
        data['distance'] = (
            (data['my_position_x'] - data['enemy_position_x'])**2 +
            (data['my_position_y'] - data['enemy_position_y'])**2 +
            (data['my_position_z'] - data['enemy_position_z'])**2)**0.5
        # 向量乘法，向量 a = （x,y,z） b = (x2,y2,z2) c = (x3,y3,z3),a代表我机速度向量
        # b代表位置向量，c代表敌机位置向量，我机中心到敌机中心向量d = c - b
        # d与a之间cos = d×a/(|d|*|a|)
        data['cos'] = ((data['my_speed_x'] *
                        (data['enemy_position_x'] - data['my_position_x'])) +
                       (data['my_speed_y'] *
                        (data['enemy_position_y'] - data['my_position_y'])) +
                       (data['my_speed_z'] *
                        (data['enemy_position_z'] - data['my_position_z'])))
        # 速度向量
        data['speedAll'] = ((data['my_speed_x']**2 + data['my_speed_y']**2 +
                             data['my_speed_z']**2)**0.5)
        # 向量之间夹角
        data['cosValue'] = data['cos'] / (data['speedAll'] * data['distance'])
        # 敌我两机位置交互式差值
        for f in ['position_x', 'position_y', 'position_z']:
            data[f'{f}_diff'] = data[f'my_{f}'] - data[f'enemy_{f}']

        return data

    @staticmethod
    def _LatLng2Degree(LatZero, LngZero, Lat, Lng):
        """
        计算我敌连线朝向, 正北方向为0度,逆时针
        Args:
            point p1(latA, lonA)
            point p2(latB, lonB)
        Returns:
            bearing between the two GPS points,
            default: the basis of heading direction is north
        """
        radLatA = math.radians(LatZero)
        radLonA = math.radians(LngZero)
        radLatB = math.radians(Lat)
        radLonB = math.radians(Lng)
        dLon = radLonB - radLonA
        y = math.sin(dLon) * math.cos(radLatB)
        x = math.cos(radLatA) * math.sin(radLatB) - math.sin(
            radLatA) * math.cos(radLatB) * math.cos(dLon)
        brng = math.degrees(math.atan2(y, x))
        brng = (brng + 360) % 360
        return brng

    @staticmethod
    def _caculate_speed_connect_cos(x, y, z, enemy_x, enemy_y, enemy_z,
                                    speed_x, speed_y, speed_z):
        """
        计算我敌连线矢量与我机速度矢量夹角
        Args:
            x, y, z: 我机坐标
            enemy_x, enemy_y, enemy_z：敌机坐标
            speed_x, speed_y, speed_z: 我机或敌机速度
        Returns:
            speed_connect_cos:我敌连线矢量与速度矢量夹角余弦值
        """
        connect_vec = np.array([enemy_x - x, enemy_y - y, enemy_z - z])
        my_speed_vec = np.array([speed_x, speed_y, speed_z])

        speed_connect_cos = connect_vec.dot(my_speed_vec) / np.sqrt(
            connect_vec.dot(connect_vec) * my_speed_vec.dot(my_speed_vec))

        return speed_connect_cos

    @staticmethod
    def _is_lead_chase(x, y, z, enemy_x, enemy_y, enemy_z, speed_x, speed_y,
                       speed_z, enemy_speed_x, enemy_speed_y, enemy_speed_z,
                       speed_connect_cos, enemy_speed_connect_cos):
        """
        判断领先追逐态势
        Args:
            x, y, z: 我机坐标
            enemy_x, enemy_y, enemy_z：敌机坐标
            speed_x, speed_y, speed_z: 我机或敌机速度
            speed_connect_cos: 我机速度与我敌连线夹角
            enemy_speed_connect_cos：敌机速度与我敌连线夹角
        Returns:
            R+: 领先追逐
            R-: 滞后追逐
            -1000:非追逐
        """
        point_1 = np.array([x, y, z])
        point_2 = np.array([enemy_x, enemy_y, enemy_z])
        point_3 = np.array(
            [x + speed_x * 0.05, y + speed_y * 0.05, z + speed_z * 0.05])
        point_4 = np.array([
            enemy_x + enemy_speed_x * 0.05, enemy_y + enemy_speed_y * 0.05,
            enemy_z + enemy_speed_z * 0.05
        ])
        mat = np.vstack([point_1, point_2, point_3])
        det = np.linalg.det(mat)

        enemy_mat = np.vstack([point_1, point_2, point_4])
        enemy_det = np.linalg.det(enemy_mat)

        if det * enemy_det >= 0 and speed_connect_cos >= enemy_speed_connect_cos:
            # 领先追逐或纯追逐
            return speed_connect_cos - enemy_speed_connect_cos
        elif det * enemy_det < 0 and speed_connect_cos > enemy_speed_connect_cos:
            # 滞后追逐
            return enemy_speed_connect_cos - speed_connect_cos
        else:
            # 非追逐
            return -1000

    @staticmethod
    def _caculate_speed_cos(speed_x, speed_y, speed_z, enemy_speed_x,
                            enemy_speed_y, enemy_speed_z):
        """
        计算我机速度矢量与敌机速度矢量夹角
        Args:
            speed_x, speed_y, speed_z：我机速度
            enemy_speed_x, enemy_speed_y, enemy_speed_z: 敌机速度
        Returns:
            speed_cos:敌机速度与我机速度矢量夹角余弦值
        """
        my_speed_vec = np.array([speed_x, speed_y, speed_z])
        enemy_speed_vec = np.array(
            [enemy_speed_x, enemy_speed_y, enemy_speed_z])

        speed_cos = my_speed_vec.dot(enemy_speed_vec) / np.sqrt(
            my_speed_vec.dot(my_speed_vec) *
            enemy_speed_vec.dot(enemy_speed_vec))

        return speed_cos

    def FE_DCS_new(self, data_):
        data = data_.copy()
        data = data.sort_values(by=['id', 'ISO time'])
        data.reset_index(drop=True, inplace=True)
        data.rename(columns={
            'U': 'x',
            'V': 'y',
            'Altitude': 'z',
            'enemy_U': 'enemy_x',
            'enemy_V': 'enemy_y',
            'enemy_Altitude': 'enemy_z',
        },
                    inplace=True)
        if self.scale == 'all':
            # 计算我机速度
            data = pd.concat([
                data,
                pd.DataFrame({
                    'speed_x': data.groupby('id')['x'].diff(),
                    'speed_y': data.groupby('id')['y'].diff(),
                    'speed_z': data.groupby('id')['z'].diff()
                })
            ],
                             sort=False,
                             axis=1)
            data.fillna(0, inplace=True)
            data[['speed_x', 'speed_y',
                  'speed_z']] = data[['speed_x', 'speed_y', 'speed_z']] / 0.05
            data['speed'] = data.apply(lambda x: np.sqrt(x['speed_x']**2 + x[
                'speed_y']**2 + x['speed_z']**2),
                                       axis=1)

            # 计算敌机速度
            data = pd.concat([
                data,
                pd.DataFrame(
                    {
                        'enemy_speed_x': data.groupby('id')['enemy_x'].diff(),
                        'enemy_speed_y': data.groupby('id')['enemy_y'].diff(),
                        'enemy_speed_z': data.groupby('id')['enemy_z'].diff()
                    })
            ],
                             sort=False,
                             axis=1)
            data.fillna(0, inplace=True)
            data[[
                'enemy_speed_x', 'enemy_speed_y', 'enemy_speed_z'
            ]] = data[['enemy_speed_x', 'enemy_speed_y', 'enemy_speed_z'
                       ]] / 0.05
            data['enemy_speed'] = data.apply(
                lambda x: np.sqrt(x['enemy_speed_x']**2 + x['enemy_speed_y']**2
                                  + x['enemy_speed_z']**2),
                axis=1)

            # 计算敌我距离
            data['distance'] = data.apply(lambda x: np.sqrt(
                (x['x'] - x['enemy_x'])**2 + (x['y'] - x['enemy_y'])**2 +
                (x['z'] - x['enemy_z'])**2),
                                          axis=1)

            # 计算我机速度与敌我连线夹角余弦值
            data['speed_connect_cos'] = data.apply(
                lambda x: self._caculate_speed_connect_cos(
                    x['x'], x['y'], x['z'], x['enemy_x'], x['enemy_y'], x[
                        'enemy_z'], x['speed_x'], x['speed_y'], x['speed_z']),
                axis=1)

            # 计算敌机速度与敌我连线夹角余弦值
            data['enemy_speed_connect_cos'] = data.apply(
                lambda x: self._caculate_speed_connect_cos(
                    x['x'], x['y'], x['z'], x['enemy_x'], x['enemy_y'], x[
                        'enemy_z'], x['enemy_speed_x'], x['enemy_speed_y'], x[
                            'enemy_speed_z']),
                axis=1)

            # 计算两机速度夹角余弦值
            data['speed_cos'] = data.apply(lambda x: self._caculate_speed_cos(
                x['speed_x'], x['speed_y'], x['speed_z'], x['enemy_speed_x'],
                x['enemy_speed_y'], x['enemy_speed_z']),
                                           axis=1)

            # 我机朝向处理
            data['Heading'] = data['Heading'] % 360

            # 计算相对位置与速度
            for f in ['x', 'y', 'z', 'speed_x', 'speed_y', 'speed_z', 'speed']:
                data[f'relative_{f}'] = data[f'enemy_{f}'] - data[f'{f}']

            # 计算是否领先追逐
            data['is_lead_chase'] = data.apply(lambda x: self._is_lead_chase(
                x['x'], x['y'], x['z'], x['enemy_x'], x['enemy_y'], x[
                    'enemy_z'], x['speed_x'], x['speed_y'], x['speed_z'], x[
                        'speed_connect_cos'], x['enemy_speed_connect_cos']),
                                               axis=1)

            # 筛除不能开火标签(非领先追逐且两机距离大于500)
            data['label'] = data.apply(lambda x: 0
                                       if x['speed_connect_cos'] < 0 or x[
                                           'distance'] > 500 else x['label'],
                                       axis=1)
            data.fillna(0, inplace=True)
            data.dropna(inplace=True)
            data.to_csv('a2a_fe.csv', index=False)
            return data

        elif self.scale == 'light':
            # 计算我机速度
            data = pd.concat([
                data,
                pd.DataFrame({
                    'speed_x': data.groupby('id')['x'].diff(),
                    'speed_y': data.groupby('id')['y'].diff(),
                    'speed_z': data.groupby('id')['z'].diff()
                })
            ],
                             sort=False,
                             axis=1)
            data.fillna(0, inplace=True)
            data[['speed_x', 'speed_y',
                  'speed_z']] = data[['speed_x', 'speed_y', 'speed_z']] / 0.05
            data['speed'] = data.apply(lambda x: np.sqrt(x['speed_x']**2 + x[
                'speed_y']**2 + x['speed_z']**2),
                                       axis=1)

            # 计算敌机速度
            data = pd.concat([
                data,
                pd.DataFrame(
                    {
                        'enemy_speed_x': data.groupby('id')['enemy_x'].diff(),
                        'enemy_speed_y': data.groupby('id')['enemy_y'].diff(),
                        'enemy_speed_z': data.groupby('id')['enemy_z'].diff()
                    })
            ],
                             sort=False,
                             axis=1)
            data.fillna(0, inplace=True)
            data[[
                'enemy_speed_x', 'enemy_speed_y', 'enemy_speed_z'
            ]] = data[['enemy_speed_x', 'enemy_speed_y', 'enemy_speed_z'
                       ]] / 0.05
            data['enemy_speed'] = data.apply(
                lambda x: np.sqrt(x['enemy_speed_x']**2 + x['enemy_speed_y']**2
                                  + x['enemy_speed_z']**2),
                axis=1)

            # 计算敌我距离
            data['distance'] = data.apply(lambda x: np.sqrt(
                (x['x'] - x['enemy_x'])**2 + (x['y'] - x['enemy_y'])**2 +
                (x['z'] - x['enemy_z'])**2),
                                          axis=1)

            # 计算我机速度与敌我连线夹角余弦值
            data['speed_connect_cos'] = data.apply(
                lambda x: self._caculate_speed_connect_cos(
                    x['x'], x['y'], x['z'], x['enemy_x'], x['enemy_y'], x[
                        'enemy_z'], x['speed_x'], x['speed_y'], x['speed_z']),
                axis=1)

            # 计算敌机速度与敌我连线夹角余弦值
            data['enemy_speed_connect_cos'] = data.apply(
                lambda x: self._caculate_speed_connect_cos(
                    x['x'], x['y'], x['z'], x['enemy_x'], x['enemy_y'], x[
                        'enemy_z'], x['enemy_speed_x'], x['enemy_speed_y'], x[
                            'enemy_speed_z']),
                axis=1)

            # 计算两机速度夹角余弦
            data['speed_cos'] = data.apply(lambda x: self._caculate_speed_cos(
                x['speed_x'], x['speed_y'], x['speed_z'], x['enemy_speed_x'],
                x['enemy_speed_y'], x['enemy_speed_z']),
                                           axis=1)

            # 计算相对位置
            for f in ['z', 'speed']:
                data[f'relative_{f}'] = data[f'enemy_{f}'] - data[f'{f}']

            # 计算是否领先追逐
            data['is_lead_chase'] = data.apply(lambda x: self._is_lead_chase(
                x['x'], x['y'], x['z'], x['enemy_x'], x['enemy_y'], x[
                    'enemy_z'], x['speed_x'], x['speed_y'], x['speed_z'],
                x['enemy_speed_x'], x['enemy_speed_y'], x['enemy_speed_z'], x[
                    'speed_connect_cos'], x['enemy_speed_connect_cos']),
                                               axis=1)

            # 筛除不能开火标签(两机距离大于1000或背对)
            data['label'] = data.apply(lambda x: 0
                                       if x['speed_connect_cos'] < 0 or x[
                                           'distance'] > 1000 else x['label'],
                                       axis=1)

            # 筛除不能开火标签(两机距离大于1000或非领先追逐)
            '''
            data['label'] = data.apply(lambda x: 0
                                       if x['is_lead_chase'] < 0 or x[
                                           'distance'] > 1000 else x['label'],
                                       axis=1)
            '''

            data.fillna(0, inplace=True)
            data.dropna(inplace=True)
            data.to_csv('a2a_fe.csv', index=False)
            return data

    # DCS在线特征工程
    def online_FE_DCS(self, row_dict):
        """ DCS在线特征工程
        Args:
            row_dict(dict): 传入的一行字典记录
        Returns:
            DataFrame: 加入特征后的记录dataframe
        """
        # 字典转dataframe
        row = pd.DataFrame(row_dict, index=[0])
        # 调用离线特征工程函数
        FE_row = self.FE_DCS(row)
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
        # id去重
        all_ids = list(set(all_ids))
        # 每次 set 的结果都不一样，所以要先排序，防止结果不可复现
        all_ids.sort()
        # random.seed 只能生效一次，所以每次 random.sample 之前都要设置
        random.seed(self.seed)
        # 训练集id采样
        train_ids = random.sample(all_ids, int(len(all_ids) * percent))
        # 获取验证集id
        val_ids = list(set(all_ids) - set(train_ids))
        # 根据id获取训练数据
        train_data = df_train[df_train['id'].isin(train_ids)]
        # 根据id获取验证数据
        val_data = df_train[df_train['id'].isin(val_ids)]
        # 连续序列数据，但是是以单个样本建模的情况下，需要 shuffle 打乱
        train_data = train_data.sample(
            frac=1, random_state=self.seed).reset_index(drop=True)

        return train_data, val_data

    def smote(self, data_):
        data = data_.copy()
        over = SMOTE(sampling_strategy=0.2, random_state=self.seed)
        under = RandomUnderSampler(sampling_strategy=1.0,
                                   random_state=self.seed)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X, y = pipeline.fit_resample(
            data[[i for i in data.columns if i not in ['label']]],
            data['label'])

        return pd.concat([X, y], axis=1)

    def _feature_name(self):
        """ 获取保留列名
        Returns:
            feature_names(list): 列名信息
        """
        # 固定顺序，否则模型预测会出错
        if self.type == 'df':
            if self.scale == 'all':
                feature_names = [
                    'my_x', 'my_y', 'my_z', 'my_v_x', 'my_v_y', 'my_v_z',
                    'my_rot_x', 'my_rot_y', 'my_rot_z', 'enemy_x', 'enemy_y',
                    'enemy_z', 'enemy_v_x', 'enemy_v_y', 'enemy_v_z',
                    'my_v_x_acc', 'enemy_v_x_acc', 'my_v_y_acc',
                    'enemy_v_y_acc', 'my_v_z_acc', 'enemy_v_z_acc',
                    'x_me_minus', 'y_me_minus', 'z_me_minus', 'v_x_me_minus',
                    'v_y_me_minus', 'v_z_me_minus', 'distance', 'cos',
                    'speedAll', 'cosValue'
                ]
            else:
                feature_names = ['cosValue', 'speedAll', 'distance']
        elif self.type == 'dcs':
            if self.scale == 'all':
                feature_names = [
                    'z', 'Roll', 'Pitch', 'Yaw', 'x', 'y', 'Heading',
                    'enemy_z', 'enemy_x', 'enemy_y', 'speed_x', 'speed_y',
                    'speed_z', 'enemy_speed_x', 'enemy_speed_y',
                    'enemy_speed_z', 'distance', 'speed', 'speed_connect_cos',
                    'enemy_speed_connect_cos', 'relative_x', 'relative_z',
                    'relative_y', 'relative_speed_x', 'relative_speed_y',
                    'relative_speed_z', 'relative_speed', 'speed_cos'
                ]
            elif self.scale == 'light':
                feature_names = [
                    'z', 'distance', 'speed', 'speed_connect_cos',
                    'enemy_speed_connect_cos', 'relative_z', 'relative_speed',
                    'speed_cos', 'is_lead_chase'
                ]
            else:
                feature_names = [
                    'z', 'Roll', 'Pitch', 'Yaw', 'x', 'y', 'Heading',
                    'enemy_z', 'enemy_x', 'enemy_y'
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
        # 获取保留的列名
        feature_names = self._feature_name()
        # 切分训练集、验证集
        train_data, val_data = self.train_val_split(raw_train,
                                                    percent=percent_train)
        if self.type == 'dcs':
            train_data = self.smote(train_data)
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

    # 定义LSTM模型
    def _lstm(self, n_steps, n_features):
        model = Sequential()
        model.add(
            LSTM(units=100,
                 activation='relu',
                 input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['AUC'])
        return model

    # 定义lightgbm模型
    def _lgb(self):
        lgb_model = lgb.LGBMClassifier(objective='binary',
                                       boosting_type='gbdt',
                                       num_leaves=32,
                                       max_depth=6,
                                       learning_rate=0.01,
                                       n_estimators=100000,
                                       subsample=0.8,
                                       feature_fraction=0.6,
                                       reg_alpha=10,
                                       reg_lambda=12,
                                       random_state=self.seed,
                                       is_unbalance=True,
                                       metric='auc')
        return lgb_model

    def _xgb(self):
        xgb_model = xgb.XGBClassifier(booster='gbtree',
                                      objective='binary:logistic',
                                      eval_metric='auc',
                                      silent=0,
                                      eta=0.01,
                                      gamma=0.1,
                                      max_depth=6,
                                      min_child_weight=3,
                                      subsample=0.7,
                                      colsample_bytree=0.5,
                                      reg_alpha=0,
                                      reg_lambda=1,
                                      n_estimators=100000,
                                      seed=2021)
        return xgb_model

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
                            max_iter=1000,
                            decision_function_shape='ovr',
                            random_state=None)
        return svm_model

    # 定义集成模型
    @staticmethod
    def _ensemble():
        clf1 = LogisticRegression(random_state=1)
        clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
        clf3 = CatBoostClassifier(iterations=100,
                                  depth=5,
                                  learning_rate=0.5,
                                  loss_function='Logloss',
                                  logging_level='Verbose')

        ensemble_model = VotingClassifier(estimators=[('lr', clf1),
                                                      ('rf', clf2),
                                                      ('gnb', clf3)],
                                          voting='soft')

        return ensemble_model

    def _train_lstm(self, raw_train, n_steps, val_type, k, percent_train=0.8):
        """ 训练lstm模型
        Args:
            raw_train(dataframe): 原始数据
            n_steps: 前向依赖时间步
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
            importance(dataframe): 特征重要度
            best_thread(float): 最佳阈值
        """
        # 获取保留列名
        if val_type == 'hold-out':  # 留出法
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 数据准备
            X_train = series_to_supervised(X_train, n_steps)
            X_val = series_to_supervised(X_val, n_steps)
            y_train = y_train[n_steps - 1:]
            y_val = y_val[n_steps - 1:]
            lstm_model = self._lstm(n_steps=n_steps,
                                    n_features=X_train.shape[-1])
            # 模型训练，使用早停策略

            my_callbacks = [
                RocAucMetricCallback(),  # include it before EarlyStopping!
                kcallbacks.EarlyStopping(monitor='val_auc',
                                         patience=3,
                                         verbose=1,
                                         mode='max')
            ]
            lstm_model.fit(X_train,
                           y_train,
                           epochs=10,
                           batch_size=256,
                           validation_data=(X_val, y_val),
                           verbose=True,
                           shuffle=False,
                           callbacks=my_callbacks)
            # 模型预测
            pred_val = lstm_model.predict(X_val)
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)
            return lstm_model, None, best_thread

        elif val_type == 'k-fold':
            # 创建模型保存列表
            model_list = []
            # 创建最佳阈值列表
            BC_list = []
            # 获取k折数据
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                y_train = to_categorical(y_train.values.tolist(),
                                         num_classes=None)
                y_val = to_categorical(y_val.values.tolist(), num_classes=None)
                # 获取lgb模型并且编号
                names['lstm_%s' % i] = self._lstm(n_steps_in=20,
                                                  n_features=len(
                                                      X_train.columns))
                # 模型训练，使用早停策略
                names['lstm_%s' % i].fit(X_train,
                                         y_train,
                                         epochs=100,
                                         batch_size=256,
                                         validation_data=(X_val, y_val),
                                         verbose=True,
                                         shuffle=False)
                # 预测验证集
                pred_val = names['lstm_%s' % i].predict(X_val)[:, 1]
                # 获取最佳阈值
                names['best_thread % s' % i] = self._BC_thread_search(
                    y_val, pred_val)
                # 保存k个模型
                model_list.append(names['lstm_%s' % i])
                # 保存k个阈值
                BC_list.append(names['best_thread % s' % i])

            return model_list, None, BC_list

        elif val_type == 'bootstrap':
            # 获取提升法数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            y_train = to_categorical(y_train.values.tolist(), num_classes=None)
            y_val = to_categorical(y_val.values.tolist(), num_classes=None)
            # 获取lgb模型
            lstm_model = self._lstm(n_steps_in=20,
                                    n_features=len(X_train.columns))
            # 模型训练，使用早停策略
            lstm_model.fit(X_train,
                           y_train,
                           epochs=100,
                           batch_size=256,
                           validation_data=(X_val, y_val),
                           verbose=True,
                           shuffle=False)

            # 模型预测
            pred_val = lstm_model.predict(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)

            return lstm_model, None, best_thread
        else:
            print("无该验证方法!")
            exit(-1)

    def _train_xgb(self, raw_train, val_type, k, percent_train=0.8):
        # 获取保留列名
        feature_names = self._feature_name()
        if val_type == 'hold-out':  # 留出法
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            xgb_model = self._xgb()
            xgb_model.fit(X_train,
                          y_train,
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          verbose=100,
                          early_stopping_rounds=50)
            # 获取特征重要性
            df_importance = pd.DataFrame({
                'column':
                feature_names,
                'importance':
                xgb_model.feature_importances_,
            }).sort_values(by='importance',
                           ascending=False).reset_index(drop=True)
            # 最大最小归一
            df_importance['importance'] = (
                df_importance['importance'] -
                df_importance['importance'].min()) / (
                    df_importance['importance'].max() -
                    df_importance['importance'].min())
            # 模型预测
            pred_val = xgb_model.predict_proba(X_val)[:, 1]
            X_val['pred_prob'] = pred_val
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)
            return xgb_model, df_importance, best_thread

        elif val_type == 'k-fold':
            # 创建模型保存列表
            model_list = []
            # 创建阈值保存列表
            BC_list = []
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
                names['lgb_%s' % i] = self._xgb()
                # 模型训练，使用早停策略
                names['lgb_%s' % i].fit(X_train,
                                        y_train,
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
                # 预测验证集
                pred_val = names['lgb_%s' % i].predict_proba(X_val)[:, 1]
                # 获取最佳阈值
                names['best_thread % s' % i] = self._BC_thread_search(
                    y_val, pred_val)
                # 保存k个模型
                model_list.append(names['lgb_%s' % i])
                # 保存k个重要性
                importance_list.append(df_importance)
                # 保存k个阈值
                BC_list.append(names['best_thread % s' % i])

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
            # 获取平均最佳阈值
            mean_BC = np.array(BC_list).mean()

            return model_list, mean_imp_df, mean_BC

        elif val_type == 'bootstrap':
            # 获取提升法数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取lgb模型
            xgb_model = self._xgb()
            # 模型训练，使用早停策略
            xgb_model.fit(X_train,
                          y_train,
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          verbose=100,
                          early_stopping_rounds=50)
            # 获取模型的特征重要性特征
            df_importance = pd.DataFrame({
                'column':
                feature_names,
                'importance':
                xgb_model.feature_importances_,
            }).sort_values(by='importance',
                           ascending=False).reset_index(drop=True)
            # 最大最小归一
            df_importance['importance'] = (
                df_importance['importance'] -
                df_importance['importance'].min()) / (
                    df_importance['importance'].max() -
                    df_importance['importance'].min())

            # 模型预测
            pred_val = xgb_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)

            return xgb_model, df_importance, best_thread

        else:
            print("无该验证方法!")
            exit(-1)

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
            best_thread(float): 最佳阈值
        """
        # 获取保留列名
        feature_names = self._feature_name()
        if val_type == 'hold-out':  # 留出法
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取lgb模型
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
            print(df_importance)
            # 模型预测
            pred_val = lgb_model.predict_proba(X_val)[:, 1]
            X_val['pred_prob'] = pred_val
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)
            return lgb_model, df_importance, best_thread

        elif val_type == 'k-fold':
            # 创建模型保存列表
            model_list = []
            # 创建阈值保存列表
            BC_list = []
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
                # 预测验证集
                pred_val = names['lgb_%s' % i].predict_proba(X_val)[:, 1]
                # 获取最佳阈值
                names['best_thread % s' % i] = self._BC_thread_search(
                    y_val, pred_val)
                # 保存k个模型
                model_list.append(names['lgb_%s' % i])
                # 保存k个重要性
                importance_list.append(df_importance)
                # 保存k个阈值
                BC_list.append(names['best_thread % s' % i])

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
            # 获取平均最佳阈值
            mean_BC = np.array(BC_list).mean()

            return model_list, mean_imp_df, mean_BC

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

            # 模型预测
            pred_val = lgb_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)

            return lgb_model, df_importance, best_thread

        else:
            print("无该验证方法!")
            exit(-1)

    def _train_nb(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练朴素贝叶斯模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
            best_thread(float): 最佳阈值
        """
        if val_type == 'hold-out':  # 留出法
            # 获取训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取朴素贝叶斯模型
            gnb_model = GaussianNB()
            # 模型训练
            gnb_model.fit(X_train, y_train)
            # 模型预测
            pred_val = gnb_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)

            return gnb_model, None, best_thread

        elif val_type == 'k-fold':
            # 创建模型保存列表
            model_list = []
            # 创建阈值保存列表
            BC_list = []
            # 获取k折交叉验证数据
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                names['gnb_%s' % i] = GaussianNB()
                # 模型训练
                names['gnb_%s' % i].fit(X_train, y_train)
                # 模型预测
                pred_val = names['gnb_%s' % i].predict_proba(X_val)[:, 1]
                # 阈值搜索
                names['best_thread % s' % i] = self._BC_thread_search(
                    y_val, pred_val)
                # 模型加入列表
                model_list.append(names['gnb_%s' % i])
                # 阈值加入列表
                BC_list.append(names['best_thread % s' % i])
            # 平均最佳阈值
            mean_BC = np.array(BC_list).mean()

            return model_list, None, mean_BC

        elif val_type == 'bootstrap':
            # 提升法获取训练、验证数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取模型
            gnb_model = GaussianNB()
            # 模型训练
            gnb_model.fit(X_train, y_train)
            # 模型预测
            pred_val = gnb_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)
            return gnb_model, None, best_thread

        else:
            print("无该验证方法!")
            exit(-1)

    def _train_linearReg(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练线性回归模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
            best_thread(float): 最佳阈值
        """
        if val_type == 'hold-out':  # 留出法
            # 获取训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取线性回归模型
            linear_model = LinearRegression()
            # 模型训练
            linear_model.fit(X_train, y_train)
            # 模型预测
            pred_val = linear_model.predict(X_val)
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)

            return linear_model, None, best_thread

        elif val_type == 'k-fold':
            # 创建模型保存列表
            model_list = []
            # 阈值保存列表
            BC_list = []
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
                # 模型预测
                pred_val = names['linear_%s' % i].predict(X_val)
                # 阈值搜索
                names['best_thread % s' % i] = self._BC_thread_search(
                    y_val, pred_val)
                # 模型加入列表
                model_list.append(names['linear_%s' % i])
                # 阈值加入列表
                BC_list.append(names['best_thread % s' % i])
            # 平均最佳阈值
            mean_BC = np.array(BC_list).mean()

            return model_list, None, mean_BC

        elif val_type == 'bootstrap':
            # 提升法获取训练、验证数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            linear_model = LinearRegression()
            # 模型训练
            linear_model.fit(X_train, y_train)
            # 模型预测
            pred_val = linear_model.predict(X_val)
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)
            return linear_model, None, best_thread

        else:
            print("无该验证方法!")
            exit(-1)

    def _train_logisticReg(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练逻辑回归模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
            best_thread(float): 最佳阈值
        """
        if val_type == 'hold-out':  # 留出法
            # 获取训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取逻辑回归模型
            logistic_model = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
            # 模型训练
            logistic_model.fit(X_train, y_train)
            # 模型预测
            pred_val = logistic_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)

            return logistic_model, None, best_thread

        elif val_type == 'k-fold':
            # 模型保存列表
            model_list = []
            # 阈值保存列表
            BC_list = []
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
                # 模型预测
                pred_val = names['logistic_%s' % i].predict_proba(X_val)[:, 1]
                # 阈值搜索
                names['best_thread % s' % i] = self._BC_thread_search(
                    y_val, pred_val)
                # 加入模型列表
                model_list.append(names['logistic_%s' % i])
                # 加入阈值列表
                BC_list.append(names['best_thread % s' % i])
            # 平均最佳阈值
            mean_BC = np.array(BC_list).mean()

            return model_list, None, mean_BC

        elif val_type == 'bootstrap':
            # 提升法获取训练、验证数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取模型
            logistic_model = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
            # 模型训练
            logistic_model.fit(X_train, y_train)
            # 模型预测
            pred_val = logistic_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)
            return logistic_model, None, best_thread

        else:
            print("无该验证方法!")
            exit(-1)

    def _train_svm(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练支持向量机模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
            best_thread(float): 最佳阈值
        """
        if val_type == 'hold-out':  # 留出法
            # 留出法获得训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取模型
            svm_model = self._svm()
            # 模型训练
            svm_model.fit(X_train, y_train)
            # 模型预测
            pred_val = svm_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)

            return svm_model, None, best_thread

        elif val_type == 'k-fold':
            # 模型保存列表
            model_list = []
            # 阈值保存列表
            BC_list = []
            # k交叉数据
            data_list = self._k_fold(raw_train, k)
            # 实现可变变量名
            names = locals()
            for member, i in zip(data_list, range(k)):
                X_train, X_val, y_train, y_val = member[0], member[1], member[
                    2], member[3]
                # 获取模型并命名
                names['svm_%s' % i] = self._svm()
                # 模型训练
                names['svm_%s' % i].fit(X_train, y_train)
                # 模型预测
                pred_val = names['svm_%s' % i].predict_proba(X_val)[:, 1]
                # 阈值搜索
                names['best_thread % s' % i] = self._BC_thread_search(
                    y_val, pred_val)
                # 模型保存
                model_list.append(names['svm_%s' % i])
                # 阈值保存
                BC_list.append(names['best_thread % s' % i])
            # 平均最佳阈值
            mean_BC = np.array(BC_list).mean()

            return model_list, None, mean_BC

        elif val_type == 'bootstrap':
            # 获取数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取模型
            svm_model = self._svm()
            # 模型训练
            svm_model.fit(X_train, y_train)
            # 模型预测
            pred_val = svm_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)
            return svm_model, None, best_thread

        else:
            print("无该验证方法!")
            exit(-1)

    def _train_ensemble(self, raw_train, val_type, k, percent_train=0.8):
        """ 训练集成模型
        Args:
            raw_train(dataframe): 原始数据
            val_type(string): 验证方式
            k(int): 交叉折数
            percent_train(float): 训练集比例
        Returns:
            model: 训练模型
            best_thread(float): 最佳阈值
        """
        if val_type == 'hold-out':  # 留出法
            # 留出法获得训练集、验证集
            X_train, X_val, y_train, y_val = self._hold_out(
                raw_train, percent_train)
            # 获取集成模型
            ensemble_model = self._ensemble()
            # 模型训练
            ensemble_model.fit(X_train, y_train)
            # 模型预测
            pred_val = ensemble_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)

            return ensemble_model, None, best_thread

        elif val_type == 'k-fold':
            # 模型保存列表
            model_list = []
            # 阈值保存列表
            BC_list = []
            # k交叉数据
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
                # 模型预测
                pred_val = names['ensemble_%s' % i].predict_proba(X_val)[:, 1]
                # 阈值搜索
                names['best_thread % s' % i] = self._BC_thread_search(
                    y_val, pred_val)
                # 模型保存
                model_list.append(names['ensemble_%s' % i])
                # 阈值保存
                BC_list.append(names['best_thread % s' % i])
            # 阈值平均
            mean_BC = np.array(BC_list).mean()

            return model_list, None, mean_BC

        elif val_type == 'bootstrap':
            # 获取数据
            X_train, X_val, y_train, y_val = self._bootstrap(raw_train)
            # 获取模型
            ensemble_model = self._ensemble()
            # 模型训练
            ensemble_model.fit(X_train, y_train)
            # 模型预测
            pred_val = ensemble_model.predict_proba(X_val)[:, :1]
            # 搜寻最佳阈值
            best_thread = self._BC_thread_search(y_val, pred_val)
            return ensemble_model, None, best_thread

        else:
            print("无该验证方法!")
            exit(-1)

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
            model, df_importance, best_thread = self._train_lgb(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance, best_thread
        # xgboost模型
        if model_type == 'xgb':
            model, df_importance, best_thread = self._train_xgb(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance, best_thread
        # 朴素贝叶斯
        elif model_type == 'nb':
            model, df_importance, best_thread = self._train_nb(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance, best_thread
        # 线性模型
        elif model_type == 'linear':
            model, df_importance, best_thread = self._train_linearReg(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance, best_thread
        # 逻辑回归
        elif model_type == 'logistic':
            model, df_importance, best_thread = self._train_logisticReg(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance, best_thread
        # 支持向量机
        elif model_type == 'svm':
            model, df_importance, best_thread = self._train_svm(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance, best_thread
        # 集成模型
        elif model_type == 'ensemble':
            model, df_importance, best_thread = self._train_ensemble(
                train, val_type, k, percent_train=percent_train)
            return model, df_importance, best_thread
        elif model_type == 'lstm':
            model, df_importance, best_thread = self._train_lstm(
                train, 20, val_type, k, percent_train=percent_train)
            return model, df_importance, best_thread
        else:
            print('无该模型！')
            exit(-1)

    @staticmethod
    def save_model(model, save_dir, name, val_type, best_thread):
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
                    save_dir, name + str(i) + '_' + val_type + '_' +
                    str(best_thread) + '.pkl')
                # 模型保存
                joblib.dump(model[i], save_path)
        else:
            # 非交叉验证模型保存
            save_path = os.path.join(
                save_dir,
                name + '_' + val_type + '_' + str(best_thread) + '.pkl')
            # 模型保存
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

    @staticmethod
    def _BC_thread_search(y_val, pred_val):
        """ 阈值搜索
        Args:
            y_val: 真实标签
            pred_val: 预测标签
        Returns:
            best_t(float): 阈值
        """
        # f1阈值敏感，暴力搜索最佳阈值
        t0 = 0.05
        v = 0.002
        best_t = t0
        best_f1 = 0
        # 循环搜索
        for step in tqdm(range(500)):
            curr_t = t0 + step * v
            y = [1 if x >= curr_t else 0 for x in pred_val]
            curr_f1 = f1_score(y_val, y)
            # 如果f1变大则替换
            if curr_f1 > best_f1:
                best_t = curr_t
                best_f1 = curr_f1

        print(f'search finish. best thread is {best_t}')

        return best_t

    def offline_predict(self, model, df_test, threshold=0.5):
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
        test_pred = df_test.copy(deep=True)
        # 创建预测列名
        test_pred['pred_prob'] = 0
        # 如果是k折交叉验证预测k次
        if isinstance(model, list):
            for i in range(len(model)):
                try:
                    # 预测概率相加
                    test_pred['pred_prob'] += model[i].predict_proba(
                        df_test[feature_names])[:, 1]
                except AttributeError:
                    # 预测概率相加
                    test_pred['pred_prob'] += model[i].predict(
                        df_test[feature_names])
            # 平均k个预测概率
            test_pred['pred_prob'] /= len(model)
            # 创建标签预测列
            test_pred['pred_label'] = 0
            # 大于阈值的预测为1，小于的预测为0
            test_pred.loc[test_pred['pred_prob'] > threshold, 'pred_label'] = 1
            # 转列表
            pred_prob = test_pred['pred_prob'].values.tolist()
            # 转列表
            pred_label = test_pred['pred_label'].values.tolist()
        # 其他验证方法
        else:
            try:
                # 如果非线性模型预测方式
                test_pred['pred_prob'] = model.predict_proba(
                    df_test[feature_names])[:, 1]
            except AttributeError:
                # 线性模型预测方式
                test_pred['pred_prob'] = model.predict(df_test[feature_names])
            except ValueError:
                # lstm预测方式
                lstm_test = series_to_supervised(df_test[feature_names], 20)
                pred_prob = model.predict(lstm_test)
                pred_prob = np.concatenate(
                    [np.zeros(19).reshape(19, 1),
                     np.array(pred_prob)])
                test_pred['pred_prob'] = pred_prob
            test_pred['pred_label'] = 0
            # 大于阈值的预测为1，小于的预测为0
            test_pred.loc[test_pred['pred_prob'] > threshold, 'pred_label'] = 1
            # 转列表
            pred_prob = test_pred['pred_prob'].values.tolist()
            # 转列表
            pred_label = test_pred['pred_label'].values.tolist()

        test_pred.to_csv('a2a_pred.csv', index=False)

        return pred_prob, pred_label

    def online_predict(self, model, row, threshold=0.5):
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
                np.array(row[feature_names]).reshape(1, -1))[:, 1]
        except AttributeError:
            # 线性模型预测方式
            pred_prob = model.predict(
                np.array(row[feature_names]).reshape(1, -1))[:, 1]
        # 在线场景下只有一条数据，返回实数，而非列表
        pred_prob = pred_prob[0]
        # 标签预测，大于阈值预测为1否则为0
        pred_label = 1 if pred_prob > threshold else 0

        return pred_prob, pred_label

    def score(self, label, pred_prob, pred_label, name, val_type):
        """ 评分函数
        Args:
            label: 真实标签
            pred_prob: 预测概率
            pred_label: 预测标签
            name:模型名称
            val_type:验证方式
        Returns:
            dict: 评分字典
        """
        # 获得评价指标
        cr = classification_report(label, pred_label, digits=4)
        print(cr)
        # 获取auc值
        auc = roc_auc_score(label, pred_prob)
        print('auc:', auc)
        # 获取acc值
        acc = accuracy_score(label, pred_label)
        print('acc:', acc)
        # 获取roc曲线
        roc_title = f'AtoA offline {name} {val_type} ROC Curve'
        roc = self._plot_roc(label, pred_prob, roc_title)
        # 获取pr曲线
        pr_title = f'AtoA offline {name} {val_type} PR Curve'
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
        fpr_test, tpr_test, _ = roc_curve(label, pred_prob)
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
        precision, recall, _ = precision_recall_curve(label, pred_prob)
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
        AP = average_precision_score(label,
                                     pred_prob,
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
