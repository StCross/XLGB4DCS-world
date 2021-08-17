import os
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2

from config import (feature_mean_most_map, feature_mean_std_map,
                    feature_min_max_map)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class FeatureEngine():
    def __init__(self, mode='offline', task_type='a2a_df'):
        """FeatureEngine 模块初始化函数

        初始化 FeatureEngine 模块，并传递初始化参数

        Args:
            mode (string): offline(离线，默认取值)/online(在线)，表明模块工作在离线模式还是在线模式，该参数会决定部分函数是否可以被调用
            task_type (string): a2a_df(dog fight 空对空任务，默认取值)/a2a_dcs(DCS 空对空任务)/a2g(DCS 空对地任务)，表明当前的任务场景

        Returns:
            None
        """

        # 参数合理性检查，mode 参数只支持 offline 和 online
        if mode not in ['offline', 'online']:
            # 出现不合法参数，抛出异常停止运行
            raise Exception(f'不支持 {mode} 模式')
        # task_type 参数只支持 a2a_df，a2a_dcs 和 a2g
        if task_type not in ['a2a_df', 'a2a_dcs', 'a2g']:
            # 出现不合法参数，抛出异常停止运行
            raise Exception(f'不支持 {task_type} 场景')

        # 注册为类变量，从而类中的其他函数可以使用
        self.mode = mode
        self.task_type = task_type

        # fillna 函数“前一个数值填充”模式下，上一时间的各特征值
        # 类型是带有默认值的字典，键是特征名称，值是上一个时刻对应的特征值，默认为0
        self.pre_vals = defaultdict(int)

        # 将 config 文件中的各个特征对应的最大值，最小值，平均值，众数，方差注册为类变量
        # 在填充空值函数中被使用
        self.feature_mean_most_map = feature_mean_most_map
        # 在归一化函数中被使用
        self.feature_min_max_map = feature_min_max_map
        # 在归一化函数中被使用
        self.feature_mean_std_map = feature_mean_std_map

    # 用来确定空对空 dcs 数据集的某一时刻点是否为一批数据的起始点
    def _dcs_is_start(self, features, preFeatures):
        # 如果前面还没有数据，肯定是起始点
        if len(preFeatures) == 0:
            return True
        # 如果 speed_x 和前一时刻的差值变得很大，说明前一个点是结束，当前点是开始
        # preFeatures 的第0列是 id，所以要和 preFeatures[-1][1] 比较
        if abs(features[0] - preFeatures[-1][1]) > 50:
            return True
        # 以上提交都不满足说明不是起始点
        return False

    # 用来确定空对空 df 数据集的某一时刻点是否为一批数据的起始点
    def _df_is_start(self, features):
        # 当一个时刻的我机速度分量 v_x, v_y, v_z 都小于1，说明是起始点
        if abs(features[3]) < 1 and abs(features[4]) < 1 and abs(
                features[5]) < 1:
            return True
        # 否则不是起始点
        return False

    @staticmethod
    def help_mean_most_output(data):
        """以字典格式输出数据的平均值或众数(辅助函数，不参与任务逻辑)

        Args:
            data (DataFrame): 输入数据

        Returns:
            dict: 字典格式的输入数据的平均值或众数
        """
        # 字典储存返回值
        output = {}

        # 遍历输入 dataframe 的每一列
        for f in data.columns:
            # 跳过不涉及模型运算的列
            if f in ['id', 'label', 'action type']:
                continue
            # 如果这一列是类别，计算众数
            if data[f].dtypes == object:
                # 可能存在多个众数，直接取第一个
                output[f] = data[f].mode()[0]
            # 如果这一列是数值特征，计算平均值
            else:
                output[f] = data[f].mean()

        # 返回字典
        return output

    @staticmethod
    def help_min_max_output(data):
        """以字典格式输出数据的最小值和最大值(辅助函数，不参与任务逻辑)

        Args:
            data (DataFrame): 输入数据

        Returns:
            dict: 字典格式的输入数据的最小值和最大值
        """
        # 字典储存返回值
        output = {}

        # 遍历输入的 dataframe 的每一列
        for f in data.columns:
            # 跳过不涉及模型运算的列
            if f in ['id', 'label', 'action type']:
                continue

            # 这一列是类别，无法计算最大最小值，跳过
            if data[f].dtypes == object:
                continue

            # 这一列是数值的情况
            # 因为是两层嵌套的字典，所以要先检查第一层嵌套的key是否存在，如果不存在要先创建
            if f not in output:
                output[f] = {}

            # 取这一列的最大值和最小值
            output[f]['max'] = data[f].max()
            output[f]['min'] = data[f].min()

        # 返回最大最小值字典
        return output

    @staticmethod
    def help_mean_std_output(data):
        """以字典格式输出数据的平均值和方差(辅助函数，不参与任务逻辑)

        Args:
            data (DataFrame): 输入数据

        Returns:
            dict: 字典格式的输入数据的平均值和方差
        """
        output = {}

        # 遍历输入的 dataframe 的每一列
        for f in data.columns:
            # 跳过不涉及模型运算的列
            if f in ['id', 'label', 'action type']:
                continue

            # 这一列是类别，无法计算平均值和方差，跳过
            if data[f].dtypes == object:
                continue

            # 这一列是数值的情况
            # 因为是两层嵌套的字典，所以要先检查第一层嵌套的key是否存在，如果不存在要先创建
            if f not in output:
                output[f] = {}

            # 取这一列的平均值和方差
            output[f]['mean'] = data[f].mean()
            output[f]['std'] = data[f].std()

        # 返回平均值和方差字典
        return output

    def read_A2A_dcs(self, path):
        """A2A DCS 数据读取

        离线模式下读取 A2A DCS 数据文件，并返回 DataFrame 格式数据

        Args:
            path (string): 数据所在文件夹路径，必须是绝对路径，禁止传入相对路径

        Returns:
            DataFrame: 列名依次是
            'id', 'my_speed_x', 'my_speed_y', 'my_speed_z',
            'my_heading', 'my_pitch', 'my_bank',
            'my_position_x', 'my_position_y', 'my_position_z', 'my_attack',
            'my_lat', 'my_long', 'my_alt',
            'my_angle_speed_x', 'my_angle_speed_y', 'my_angle_speed_z', 'my_sheels',
            'enemy_position_x', 'enemy_position_y', 'enemy_position_z', 'enemy_lat', 'enemy_long', 'enemy_alt', 'label'
        """
        # 前一条样本是否是起始点
        pre_flag = False
        # 批次编号，从0递增
        id = 0
        # 存储处理好的数据
        records_list = []

        # 打开文件
        with open(path) as f:
            # 读取文件中的所有行
            lines = f.readlines()
            # 遍历每一行
            for line in lines:
                # 根据数据格式以;@;进行分割，分割后形成两部分，前面是特征，后面是标签
                features, labels = line.split(';@;')
                # 特征部分用;进行分割
                features = features.split(';')
                # 标签部分用;进行分割
                labels = labels.split(';')
                # 分割后的标签列表只有最后一列是我们需要的标签
                label = labels[-1]
                # 特征转为float数值类型
                features = [float(f) for f in features]
                # 标签转为int数值类型
                label = int(label)

                # 调用 _dcs_is_start 判断当前点是否是起始点
                flag = self._dcs_is_start(features, records_list)
                # 前面一条数据不是起始数据，当前是起始数据，则开始收集新场次数据，id进行自增
                if not pre_flag and flag:
                    id += 1
                # 批id + 特征列表 + 标签作为当前时刻的数据合并到处理好的数据
                records_list.append([id] + features + [label])
                # 更新 pre_flag
                pre_flag = flag

        # 将处理好的数据变为 dataframe 格式
        df_record = pd.DataFrame(records_list)

        # 特征名称列表，第一个是 id
        columns = ['id']
        # 我机的所有特征名称
        for f in [
            'speed_x', 'speed_y', 'speed_z', 'heading', 'pitch', 'bank',
            'position_x', 'position_y', 'position_z', 'attack', 'lat',
            'long', 'alt', 'angle_speed_x', 'angle_speed_y',
            'angle_speed_z', 'sheels'
        ]:
            columns.append(f'my_{f}')
        # 敌机的所有特征名称
        for f in [
            'position_x', 'position_y', 'position_z', 'lat', 'long', 'alt',
            'heading', 'pitch', 'bank'
        ]:
            columns.append(f'enemy_{f}')
        # 最后一个特征是label
        columns += ['label']
        # 给df_record设置列名
        df_record.columns = columns

        # 删除敌机无法真实获取的列
        df_record.drop(columns=['enemy_heading', 'enemy_pitch', 'enemy_bank'],
                       inplace=True)

        return df_record

    def read_A2A_dcs_new(self, raw_data_path):
        convert_data_path = os.path.join('convert_AtoA_dcs_data', 'new_AtoA_dcs_data.csv')
        dfs = pd.DataFrame()
        if os.path.exists(convert_data_path):
            print('开始载入已处理数据')
            dfs = pd.read_csv(convert_data_path)
            print('加载完毕')

        else:
            group_data_path = os.path.join('convert_AtoA_dcs_data', 'group_data')
            if not os.path.exists(group_data_path):
                shutil.rmtree(group_data_path, ignore_errors=True)
                os.makedirs(group_data_path, exist_ok=True)
            for path, _, filenames in os.walk(raw_data_path):
                for filename in filenames:
                    data = pd.read_csv(os.path.join(path, filename))
                    # 找到第一个敌机数据的索引
                    if data[data['Id'] == '102'].empty is True or data[data['Id'] == '103'].empty == True:
                        continue
                    first_enemy_index = data[data['Id'] == '103'].index.values[0]
                    data = data.loc[first_enemy_index - 1:]
                    data.reset_index(drop=True, inplace=True)
                    last_enemy_index = data.index.values[-1]
                    for index in range(0, data.shape[0]):
                        if data.loc[index, 'Id'] == '102' and data.loc[index + 1, 'Id'] != '103':
                            last_enemy_index = index - 1
                            break
                        if data.loc[index, 'Id'] == '103' and data.loc[index - 1, 'Id'] != '102':
                            last_enemy_index = index - 1
                            break
                    data = data.loc[:last_enemy_index]
                    data.reset_index(drop=True, inplace=True)

                    myPlane_index = data[data['Id'] == '102'].index.values
                    for index in myPlane_index:
                        data.loc[index, 'label'] = 0
                        for missile_index in range(index + 2, last_enemy_index + 1):
                            if data.loc[missile_index, 'Id'] != '102' and data.loc[missile_index, 'Id'] != '103':
                                distance1 = (data.loc[index, 'Longitude'] - data.loc[
                                    missile_index, 'Longitude']) ** 2 + (
                                                    data.loc[index, 'Latitude'] - data.loc[
                                                missile_index, 'Latitude']) ** 2 + (
                                                    data.loc[index, 'Altitude'] - data.loc[
                                                missile_index, 'Altitude']) ** 2
                                distance2 = (data.loc[index + 1, 'Longitude'] - data.loc[
                                    missile_index, 'Longitude']) ** 2 + (
                                                    data.loc[index + 1, 'Latitude'] - data.loc[
                                                missile_index, 'Latitude']) ** 2 + (
                                                    data.loc[index + 1, 'Altitude'] - data.loc[
                                                missile_index, 'Altitude']) ** 2
                                if distance1 <= distance2:
                                    data.loc[index, 'label'] = 1
                                    break
                            else:
                                break
                    myPlane = data[data['Id'] == '102'].copy()
                    myPlane.reset_index(drop=True, inplace=True)
                    enemy = data[data['Id'] == '103'].copy()
                    enemy.reset_index(drop=True, inplace=True)
                    for col in ['ISO time', 'Unix time', 'label']:
                        del enemy[col]
                    for col in enemy.columns:
                        enemy.rename(columns={col: 'enemy_' + col}, inplace=True)
                    df = pd.concat([myPlane, enemy], sort=False, axis=1)
                    df.reset_index(drop=True, inplace=True)

                    df.insert(0, 'id', filename.split('.')[0])
                    df.reset_index(drop=True, inplace=True)
                    df['ISO time'] = df['ISO time'].apply(lambda x: round(float(x.split(':')[1]) * 60 + float(
                        x.split(':')[2][0:-1]), 2))
                    df = df.sort_values(by=['ISO time'])
                    label_1_index = []
                    for index in df.index.values:
                        if df.loc[index, 'label'] == 1 and df.loc[index - 1, 'label'] == 0:
                            for i in np.arange(index - 10, index + 11, 1):
                                label_1_index.append(i)
                    df.loc[:, 'label'] = 0
                    label_1_index = list(set(label_1_index))
                    for i in label_1_index:
                        if i < 0 or i > df.index.values[-1]:
                            label_1_index.remove(i)
                    for index in label_1_index:
                        df.loc[index, 'label'] = 1
                    df.to_csv(os.path.join(group_data_path, filename), index=False)
                    # for index in df.index.values:
                    #     if not df.loc[index].equals(pd.read_csv(os.path.join(group_data_path, filename)).loc[index]):
                    #         print(index)
                    #         print(df.loc[index])
                    #         print(pd.read_csv(os.path.join(group_data_path, filename)).loc[index])
                    dfs = dfs.append(df)
                    dfs.reset_index(drop=True, inplace=True)
                    print('id为' + filename.split('.')[0] + '的批次处理完毕')

            dfs.reset_index(drop=True, inplace=True)
            dfs.to_csv(convert_data_path, index=False)
            dfs = pd.read_csv(convert_data_path)

        return dfs

    def read_A2A_df(self, path):
        """A2A DF 数据读取

        离线模式下读取 A2A DF 数据文件，并返回 DataFrame 格式数据

        Args:
            path (string): 数据所在文件夹路径，必须是绝对路径，禁止传入相对路径

        Returns:
            DataFrame: 列名依次是
            'id', 'my_x', 'my_y', 'my_z',
            'my_v_x', 'my_v_y', 'my_v_z',
            'my_rot_x', 'my_rot_y', 'my_rot_z',
            'enemy_x', 'enemy_y', 'enemy_z',
            'label'
        """
        # 前一条样本是否是起始点
        pre_flag = False
        # 批次编号，从0递增
        id = 0
        # 存储处理好的数据
        records_list = []
        # 打开数据文件
        with open(path) as f:
            # 读取文件中的所有行
            lines = f.readlines()
            # 遍历每一行
            for line in lines:
                # 根据数据格式以#进行分割，分割后形成两部分，前面是特征，后面是标签
                features, label = line.split('#')
                # 特征部分用;进行分割
                features = features.split(';')
                # 特征转为float数值类型
                features = [float(f) for f in features]
                # 标签转为int数值类型
                label = int(label)
                # 调用 _df_is_start 判断当前点是否是起始点
                flag = self._df_is_start(features)

                # 前面一条数据不是起始数据，当前是起始数据，则开始收集新场次数据，id进行自增
                if not pre_flag and flag:
                    id += 1
                # 批id + 特征列表 + 标签作为当前时刻的数据合并到处理好的数据
                records_list.append([id] + features + [label])
                # 更新 pre_flag
                pre_flag = flag
        # 将处理好的数据变为 dataframe 格式
        df_record = pd.DataFrame(records_list)
        # 特征名称列表，第一个是 id
        columns = ['id']
        # 我机和敌机的所有特征名称
        for role in ['my', 'enemy']:
            for f in [
                'x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'rot_x', 'rot_y',
                'rot_z'
            ]:
                columns.append(f'{role}_{f}')
        # 最后一个特征是label
        columns += ['label']
        # 给df_record设置列名
        df_record.columns = columns

        # 删除敌机无法真实获取的列
        df_record.drop(columns=[
            'enemy_v_x', 'enemy_v_y', 'enemy_v_z', 'enemy_rot_x',
            'enemy_rot_y', 'enemy_rot_z'
        ],
            inplace=True)

        return df_record

    @staticmethod
    def _init_data_path(path):
        # 初始化 data 目录，先清空
        shutil.rmtree(path, ignore_errors=True)
        # 创建 data 文件夹，如果已经存在就跳过
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _movefile_util(files, path, id):
        # 创建 id 对应的批数据存放目录，如果已经存在就跳过
        os.makedirs(os.path.join(path, id), exist_ok=True)

        # 依次将空对地数据的enemyMissile，Joystick，Label，myPlane，Throttle文件移动到指定的文件夹下
        for type in [
            'enemyMissile', 'Joystick', 'Label', 'myPlane', 'Throttle'
        ]:
            shutil.copy(files[id][type], os.path.join(path, id, f'{type}.csv'))

    def _movefile(self, raw_data_path):
        # 遍历 path 下所有 csv 文件，获取时间 id，时间 id 只保留保证部分
        # 比如 get_DCS_AtoG_Data_Joystick_1614561248.5681126.csv 的 id 为1614561248

        # 存储所以的 id
        id_list = []
        # 获取 raw_data_path 下的所有文件列表
        for _, _, file_list in os.walk(raw_data_path):
            # 遍历每个文件
            for file_name in file_list:
                # 检查文件后缀是否为 csv
                if 'csv' in file_name:
                    # 对文件名进行分割，以_切分获取最后包含id的部分，然后以.分割获取纯数字id
                    id = file_name.split('_')[-1].split('.')[0]
                    # 添加到id列表
                    id_list.append(id)

        # 读取所有的文件，并按照 id，type 进行存储
        # files 是双层key字典，存储各个任务id下各个类型文件的路径
        files = {}
        for path, _, file_list in os.walk(raw_data_path):
            # 遍历每个文件
            for file_name in file_list:
                # 如果不是csv文件，跳过
                if 'csv' not in file_name:
                    continue
                # 以_分割获取数据类型和id
                type = file_name.split('_')[4]
                id = file_name.split('_')[-1].split('.')[0]
                # 如果id不在files字典中，先创建对应的空字典
                if id not in files:
                    files[id] = {}
                # 保存对应的文件路径
                files[id][type] = os.path.join(path, file_name)

        # 将指定 id 的轨迹相关数据分别移动到专门的convert_AtoG_data文件夹下
        '''
        每一个轨迹文件夹的文件内容

        id/
            enemyMissile.csv
            Joystick.csv
            Label.csv
            myPlane.csv
            Throttle.csv
        '''
        # 在当前程序运行路径下建立新文件夹 convert_AtoG_data/group_data
        move_dir = os.path.join('convert_AtoG_data', 'group_data')
        for ids in id_list:
            self._movefile_util(files, move_dir, ids)

    @staticmethod
    def _dict2df(data_):
        # 将enenyMissile字典格式转换为dataframe格式
        data = data_.copy(deep=True)
        # 获取导弹数据
        missiles = data['MissilesDict']
        # 如果对应数据为NONE
        if missiles == 'NONE':
            # 导弹数量设置为0
            data['Missile_num'] = 0
            # 删除无用的MissilesDict列
            del data['MissilesDict']
            return data
        else:
            # 导弹编号，有且只有一枚导弹
            i = 1
            # 导弹数据包含的所有特征
            cols = [
                'position_x', 'position_y', 'position_z', 'longitude',
                'dimensionality', 'high', 'yaw', 'pitch', 'roll'
            ]
            # 字符串转字典需要使用内置函数eval()
            missiles = eval(missiles)
            # 遍历所有的导弹
            for key in missiles.keys():
                j = 1  # 不去读导弹名称
                # 遍历当前导弹的所有特征
                for f in cols:
                    data[f'Missile_{i}_{f}'] = float(missiles[key][j])
                    j += 1
                # 导弹编号自增
                i += 1
            # 导弹的数量
            data['Missile_num'] = i - 1
            # 删除无用的MissilesDict列
            del data['MissilesDict']

            return data

    @staticmethod
    def _convert_label(data_):
        data = data_.copy(deep=True)

        # 新增 label 列，初始化为负样本
        data['label'] = 0

        # action type 对应行的 label 设置为1
        data.loc[data['action type'] == 'search target', 'label'] = 1
        # action type 对应行的 label 设置为2
        data.loc[data['action type'] == 'target locked', 'label'] = 2
        # action type 对应行的 label 设置为3
        data.loc[data['action type'] == 'attack target', 'label'] = 3
        # action type 对应行的 label 设置为4
        data.loc[data['action type'] == 'escape', 'label'] = 4

        return data

    # 将空对地数据转换成一张表存储
    def _convert_A2G(self, raw_data_path):
        # 初始化存储的文件夹
        move_dir = os.path.join('convert_AtoG_data', 'group_data')
        # 如果文件夹不存在
        if not os.path.exists(move_dir):
            self._init_data_path(move_dir)
            self._movefile(raw_data_path)

        # 初始化空的 dataframe
        df = pd.DataFrame()

        for path, dir_list, _ in os.walk(move_dir):
            # 读取某一组数据
            num = 1
            # 遍历所有的任务id
            for id in dir_list:
                print('\r', f'读取并转换第{num}批原始数据', end='', flush=True)
                num += 1
                # 加载 myPlane 数据
                my_plane = pd.read_csv(os.path.join(path, id, 'myPlane.csv'))
                # 列名['LoadInf']重命名为 LoadInf
                my_plane.rename(columns={"['LoadInf']": 'LoadInf'},
                                inplace=True)
                # 在第一列插入id列
                my_plane.insert(0, 'id', id)
                # 加载 Label 数据
                label = pd.read_csv(os.path.join(path, id, 'Label.csv'))
                # 加载 enemyMissile 数据
                enemyMissile = pd.read_csv(
                    os.path.join(path, id, 'enemyMissile.csv'))
                # 为每一行转换导弹数据
                enemyMissile = enemyMissile.apply(lambda x: self._dict2df(x),
                                                  axis=1)
                # 将没有导弹的列信息填充为0
                enemyMissile.fillna(0, inplace=True)

                # 多个表数据合并
                # 按道理同一次任务的所有数据都是按时间排列好的，且采集的时间点一致
                # 所以我们可以根据时间排序后（排序非必须，但以防原始数据未排序带来的隐患），直接多个表concat，速度更快
                my_plane.sort_values('Time', inplace=True)
                label.sort_values('Time', inplace=True)
                enemyMissile.sort_values('Time', inplace=True)
                # 每个表都有 Time 列，只保留 my_plane 表中的 Time 列
                del label['Time']
                del enemyMissile['Time']

                # 因为所有组的数据全装在一起，所以加个 id 特征区分数据属于哪一组数据

                data = pd.concat([my_plane, label, enemyMissile], axis=1)
                # 存在 Label 数据缺少最后一个时刻的情况，所以 concat 之后可能会出现最后一条数据 action type 为 NaN
                len1 = data.shape[0]
                # 所以删除 action type 为空的数据 ,后面引入其他表的时候需要注意类似的情况
                data = data[data['action type'].notna()]
                len2 = data.shape[0]
                print('\r',
                      f'读取并转换第{num}批原始数据,该批数据删除了{abs(len2 - len1)}行记录',
                      end='',
                      flush=True)

                # 将当前这组数据合并到整体的 dataframe
                df = df.append(data)

        # 提取我们想要的标签
        df = self._convert_label(df)

        csv_path = os.path.join('convert_AtoG_data', 'AtoG_data.csv')
        # 保存到文件中去
        df.to_csv(csv_path, index=0, encoding='utf-8')

        return df

    def read_A2G(self, raw_data_path):
        """A2G 数据读取

        离线模式下读取 A2G 数据文件，并返回 DataFrame 格式数据

        Args:
            path (string): 数据所在文件夹路径，必须是绝对路径，禁止传入相对路径

        Returns:
            DataFrame
        """

        # 初始化空的dataframe
        data = pd.DataFrame()

        csv_path = os.path.join('convert_AtoG_data', 'AtoG_data.csv')
        # 如果数据文件已经处理好，则直接读取，加快速度
        if os.path.exists(csv_path):
            print('开始载入已合并的csv数据')
            data = pd.read_csv(csv_path, encoding='utf-8')
            print('载入完毕')
        # 否则需要处理原始数据，所需要的时间较长
        else:
            print('开始处理原始数据')
            data = self._convert_A2G(raw_data_path)
            print('\n处理完毕')

        # 删除无用的特征列
        data.drop(columns=['ID_Country', 'Name', 'ID_Plane'], inplace=True)

        return data

    def _convert_A2G_rel_attack(self, data_path):
        # 初始化存储的文件夹
        move_dir = os.path.join('convert_AtoG_data', 'group_data')
        # 如果文件夹不存在
        if not os.path.exists(move_dir):
            self._init_data_path(move_dir)
            self._movefile(data_path)

        # 初始化空的 dataframe
        df = pd.DataFrame()

        for path, dir_list, _ in os.walk(move_dir):
            # 读取某一组数据
            for id in dir_list:
                # 加载 myPlane 数据
                my_plane = pd.read_csv(os.path.join(path, id, 'myPlane.csv'))
                my_plane.rename(columns={"['LoadInf']": 'LoadInf'},
                                inplace=True)

                # 加载 Label 数据
                label = pd.read_csv(os.path.join(path, id, 'Label.csv'))
                label.insert(0, 'id', id)

                # 加载 enemy_missile 数据
                enemy_missile = pd.read_csv(
                    os.path.join(path, id, 'enemyMissile.csv'))
                enemy_missile = enemy_missile.apply(lambda x: self._dict2df(x),
                                                    axis=1)
                enemy_missile.fillna(0, inplace=True)  # 将没有导弹的列信息填充为0

                # 加载更多的数据表。。。。

                # 多个表数据合并
                # 按道理同一次任务的所有数据都是按时间排列好的，且采集的时间点一致
                # 所以我们可以根据时间排序后（排序非必须，但以防原始数据未排序带来的隐患），直接多个表concat，速度更快
                my_plane.sort_values('Time', inplace=True)
                my_plane.reset_index(drop=True, inplace=True)
                label.sort_values('Time', inplace=True)
                label.reset_index(drop=True, inplace=True)
                enemy_missile.sort_values('Time', inplace=True)
                enemy_missile.reset_index(drop=True, inplace=True)
                # 每个表都有 Time 列，只保留 label 表中的 Time 列
                del my_plane['Time']
                del enemy_missile['Time']

                # 因为所有组的数据全装在一起，所以加个 id 特征区分数据属于哪一组数据

                data = pd.concat([my_plane, label, enemy_missile],
                                 sort=False,
                                 axis=1)
                # 存在 Label 数据缺少最后一个时刻的情况，所以 concat 之后可能会出现最后一条数据 action type 为 NaN
                # 所以删除 action type 为空的数据
                # 后面引入其他表的时候需要注意类似的情况
                data = data[data['action type'].notna()]

                # 三个表可能会同时缺某一时刻的数据, 用前一个时刻的数据进行补充
                for time in np.arange(data['Time'].values[0],
                                      data['Time'].values[-1] + 0.01, 0.05):
                    time = float('%.2f' % time)
                    if time not in data['Time'].values:
                        data_lost = data[data['Time'] == time -
                                         0.05].reset_index(drop=True)
                        data_lost.loc[0, 'Time'] = time
                        data.append(data_lost)
                # 重新排序,重置索引
                data.sort_values('Time', inplace=True)
                data.reset_index(drop=True, inplace=True)

                # 将missile和myPlane 30.5秒后的数据再加进去,shift(-610)
                data.rename(columns={
                    'Missile_1_dimensionality': 'Missile_dimensionality',
                    'Missile_1_high': 'Missile_high',
                    'Missile_1_longitude': 'Missile_longitude',
                    'Missile_1_pitch': 'Missile_pitch',
                    'Missile_1_position_x': 'Missile_position_x',
                    'Missile_1_position_y': 'Missile_position_y',
                    'Missile_1_position_z': 'Missile_position_z',
                    'Missile_1_roll': 'Missile_roll',
                    'Missile_1_yaw': 'Missile_yaw'
                },
                    inplace=True)
                data_after_t = data[[
                    'Position_x', 'Position_y', 'Position_z',
                    'Missile_position_x', 'Missile_position_y',
                    'Missile_position_z'
                ]].copy()
                for col in data_after_t.columns:
                    data_after_t.rename(columns={col: col + '_after_t'},
                                        inplace=True)
                data_after_t = data_after_t.shift(-610)
                data_after_t.fillna(0, inplace=True)

                data = pd.concat([data, data_after_t], sort=False, axis=1)

                # 将当前这组数据合并到整体的 dataframe
                df = df.append(data).reset_index(drop=True)
                print(f'id为{id}的数据处理完毕')

        # 若不存在文件夹则创建
        if not os.path.exists('AtoG_rel_data'):
            os.makedirs('AtoG_rel_data')

        df.to_csv(os.path.join('AtoG_rel_data', 'AtoG_rel_attack_data.csv'), index=False)

        return df

    def read_A2G_rel_attack(self, data_path):  # 直接读取合并表，加快速度
        """A2G_rel_attack 数据读取

        离线模式下读取并处理 A2G 数据文件，并返回 DataFrame 格式数据

        Args:
            path (string): 数据所在文件夹路径，必须是绝对路径，禁止传入相对路径

        Returns:
            DataFrame
        """
        data = pd.DataFrame()

        if os.path.exists(
                os.path.join('AtoG_rel_data', 'AtoG_rel_attack_data.csv')):
            print('开始载入已合并的csv数据')
            data = pd.read_csv(
                os.path.join('AtoG_rel_data', 'AtoG_rel_attack_data.csv'))
            print('载入完毕')
        else:
            print('开始处理原始数据')
            data = self._convert_A2G_rel_attack(data_path)
            print('处理完毕')

        # data.drop(columns=['ID_Country', 'Name', 'ID_Plane'], inplace=True)

        return data

    def _convert_A2G_rel_locked(self, data_path):
        # 初始化存储的文件夹
        move_dir = os.path.join('convert_AtoG_data', 'group_data')
        # 如果文件夹不存在
        if not os.path.exists(move_dir):
            self._init_data_path(move_dir)
            self._movefile(data_path)

        # 初始化空的 dataframe
        df = pd.DataFrame()

        for path, dir_list, _ in os.walk(move_dir):
            # 读取某一组数据
            for id in dir_list:
                # 加载 Label 数据
                label = pd.read_csv(os.path.join(path, id, 'Label.csv'))
                label.insert(0, 'id', id)

                try:
                    if label[label['action type'] == 'search target'].index.values[-1] > \
                            label[label['action type'] == 'target locked'].index.values[0]:
                        print('id为' + id + '的数据异常进行删除')
                        continue
                except IndexError:  # 有的批次连search target都没有
                    pass
                # 加载 myPlane 数据
                my_plane = pd.read_csv(os.path.join(path, id, 'myPlane.csv'))
                my_plane.rename(columns={"['LoadInf']": 'LoadInf'},
                                inplace=True)

                # 加载更多的数据表。。。。

                # 多个表数据合并
                # 按道理同一次任务的所有数据都是按时间排列好的，且采集的时间点一致
                # 所以我们可以根据时间排序后（排序非必须，但以防原始数据未排序带来的隐患），直接多个表concat，速度更快
                my_plane.sort_values('Time', inplace=True)
                my_plane.reset_index(drop=True, inplace=True)
                label.sort_values('Time', inplace=True)
                label.reset_index(drop=True, inplace=True)
                # 每个表都有 Time 列，只保留 label 表中的 Time 列
                del my_plane['Time']

                # 因为所有组的数据全装在一起，所以加个 id 特征区分数据属于哪一组数据

                data = pd.concat([my_plane, label], sort=False, axis=1)
                # 存在 Label 数据缺少最后一个时刻的情况，所以 concat 之后可能会出现最后一条数据 action type 为 NaN
                # 所以删除 action type 为空的数据
                # 后面引入其他表的时候需要注意类似的情况
                data = data[data['action type'].notna()]

                # 三个表可能会同时缺某一时刻的数据, 用前一个时刻的数据进行补充
                for time in np.arange(data['Time'].values[0],
                                      data['Time'].values[-1] + 0.01, 0.05):
                    time = float('%.2f' % time)
                    if time not in data['Time'].values:
                        data_lost = data[data['Time'] == time -
                                         0.05].reset_index(drop=True)
                        data_lost.loc[0, 'Time'] = time
                        data.append(data_lost)
                # 重新排序,重置索引
                data.sort_values('Time', inplace=True)
                data.reset_index(drop=True, inplace=True)

                # 将myPlane 30秒后的经纬度数据再加进去,shift(-600)
                data_after_t = data[['MyLong', 'MyLat']].copy()
                for col in data_after_t.columns:
                    data_after_t.rename(columns={col: col + '_after_t'},
                                        inplace=True)
                data_after_t = data_after_t.shift(-600)
                data_after_t.fillna(0, inplace=True)

                data = pd.concat([data, data_after_t], sort=False, axis=1)

                # 将当前这组数据合并到整体的 dataframe
                df = df.append(data).reset_index(drop=True)
                print(f'id为{id}的数据处理完毕')

        # 若不存在文件夹则创建
        if not os.path.exists('AtoG_rel_data'):
            os.makedirs('AtoG_rel_data')

        # 保存csv
        df.to_csv(os.path.join('AtoG_rel_data', 'AtoG_rel_locked_data.csv'), index=False)

        return df

    def read_A2G_rel_locked(self, data_path):  # 直接读取合并表，加快速度
        """A2G_rel_locked 数据读取

        离线模式下读取并处理 A2G 数据文件，并返回 DataFrame 格式数据

        Args:
            path (string): 数据所在文件夹路径，必须是绝对路径，禁止传入相对路径

        Returns:
            DataFrame
        """
        data = pd.DataFrame()

        if os.path.exists(os.path.join('AtoG_rel_data', 'AtoG_rel_locked_data.csv')):
            print('开始载入已合并的csv数据')
            data = pd.read_csv(os.path.join('AtoG_rel_data', 'AtoG_rel_locked_data.csv'))
            print('载入完毕')
        else:
            print('开始处理原始数据')
            data = self._convert_A2G_rel_locked(data_path)
            print('处理完毕')

        # data.drop(columns=['ID_Country', 'Name', 'ID_Plane'], inplace=True)

        return data

    def fillna(self, data_, method='mean_most'):
        """缺失值处理

        缺失值填充，支持众数填充/平均值填充/前一个数值填充。
        填充的平均值和众数为对应特征在线下大数据上的统计量。

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据
            method (string): mean_most(对连续特征平均值填充，离散特征众数填充，默认取值)，shift(前一个数值填充),填充方式

        Returns:
            DataFrame: 填充完毕的 DataFrame 数据
        """
        # method 参数合理性验证
        if method not in ['mean_most', 'shift']:
            raise Exception(f'不支持 {method} 填充方式')

        # data_ 参数类型验证
        if not isinstance(data_, pd.DataFrame):
            raise Exception('输入数据不是 DataFrame 格式')

        data = data_.copy(deep=True)
        # 获取任务模式，在线或者离线
        mode = self.mode
        # 获取任务类型
        task_type = self.task_type
        # 在线模式下不支持输入长度大于1的数据
        if mode == 'online' and len(data) != 1:
            raise Exception(f'在线模式下不支持输入长度为{len(data)}的数据')

        # 均值或者众数填充
        if method == 'mean_most':
            # 遍历所有特征列
            for f in data.columns:
                # 没有缺失值跳过
                if data[f].isnull().sum() > 0:
                    # 直接用计算好的特征对应的均值或者众数填充
                    data[f].fillna(self.feature_mean_most_map[task_type][f],
                                   inplace=True)
        # 前一值填充法
        elif method == 'shift':
            # 在线模式
            if mode == 'online':
                # 遍历每一个特征
                for f in data.columns:
                    # 如果有缺失
                    if data[f].isnull().sum() > 0:
                        # 用类变量 pre_vals 字典对应的value填充
                        data[f].fillna(self.pre_vals[f], inplace=True)

                    # 更新类变量 pre_vals 字典对应的value
                    self.pre_vals[f] = data[f].values[0]
            # 离线模式
            else:
                # 遍历每一个特征
                for f in data.columns:
                    # 如果有缺失
                    if data[f].isnull().sum() > 0:
                        # 调用 dataframe 自带的 fillna 函数，pad 参数就代表前一值
                        data[f].fillna(method='pad', inplace=True)

        return data

    def outlier_handle(self, data_):
        """异常值处理

        超过特征值上限的置为上限值，超过特征值下限的置位下限值

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据

        Returns:
            DataFrame: 异常值处理完毕的 DataFrame 数据
        """
        # data_ 参数类型验证
        if not isinstance(data_, pd.DataFrame):
            raise Exception('输入数据不是 DataFrame 格式')

        data = data_.copy(deep=True)
        # 获取任务模式，在线或者离线
        mode = self.mode
        # 获取任务类型
        task_type = self.task_type
        # 在线模式下不支持输入长度大于1的数据
        if mode == 'online' and len(data) != 1:
            raise Exception(f'在线模式下不支持输入长度为{len(data)}的数据')

        # 遍历所有特征列
        for f in data.columns:
            # 该列没有对应的_min_max_就跳过
            if f not in self.feature_min_max_map[task_type]:
                continue
            # 获取当前特征的最大值和最小值
            max_val = self.feature_min_max_map[task_type][f]['max']
            min_val = self.feature_min_max_map[task_type][f]['min']
            # 超过最大值的为异常值，将其置为最大值
            data.loc[data[f] > max_val, f] = max_val
            # 低于最小值的为异常值，将其置为最小值
            data.loc[data[f] < min_val, f] = min_val

        return data

    def normalization(self, data_, method='min_max'):
        """归一化

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据
            method (string): min_max/z-score/sigmoid，归一化方式
        Returns:
            DataFrame: 异常值处理完毕的 DataFrame 数据
        """
        if method not in ['min_max', 'z-score', 'sigmoid']:
            raise Exception(f'不支持 {method} 归一化')

        data = data_.copy(deep=True)
        # 获取任务类型
        task_type = self.task_type
        # 获取任务模式，在线或者离线
        mode = self.mode
        # 在线模式下不支持输入长度大于1的数据
        if mode == 'online' and len(data) != 1:
            raise Exception(f'在线模式下不支持输入长度为{len(data)}的数据')

        # 最大最小归一
        if method == 'min_max':
            # 遍历所有特征列
            for f in data.columns:
                # 该特征没有对应的_min_max_就跳过
                '''
                if f not in self.feature_min_max_map[task_type]:
                    continue
                '''
                # 获取当前特征的最大值和最小值
                '''
                max_val = self.feature_min_max_map[task_type][f]['max']
                min_val = self.feature_min_max_map[task_type][f]['min']
                '''
                max_val = data[f].max()
                min_val = data[f].min()
                # 归一化公式，(x-x_min) / (x_max-x_min)
                data.loc[:,
                f] = (data.loc[:, f] - min_val) / (max_val - min_val)
                # 归一化后范围应该是0-1
                # 将数值小于0的置为0
                data.loc[data[f] < 0, f] = 0
                # 将数值大于1的置为1
                data.loc[data[f] > 1, f] = 1
        # 均值方差归一法
        elif method == 'z-score':
            # 遍历所有特征列
            for f in data.columns:
                # 该特征没有对应的_mean_std_就跳过
                if f not in self.feature_mean_std_map[task_type]:
                    continue
                # 获取当前特征的均值和方差
                mean_val = self.feature_mean_std_map[task_type][f]['mean']
                std_val = self.feature_mean_std_map[task_type][f]['std']
                # 归一化公式：(x-x_mean)  / x_std
                data.loc[:, f] = (data.loc[:, f] - mean_val) / std_val
                # 将数值小于0的置为0
                data.loc[data[f] < 0, f] = 0
                # 将数值大于1的置为1
                data.loc[data[f] > 1, f] = 1
        elif method == 'sigmoid':
            # 遍历所有特征列
            for f in data.columns:
                # 归一化公式：1 / (1+e^(-x))
                data[f] = data[f].apply(lambda x: 1 / (1 + np.exp(-x)))
                # 将数值小于0的置为0
                data.loc[data[f] < 0, f] = 0
                # 将数值大于1的置为1
                data.loc[data[f] > 1, f] = 1

        return data

    def select_chi2(self, data_, num, feature_names, ycol):
        """卡方检验

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据
            num (int): 需保留的特征个数
            feature_names (list): 待筛选特征列表
            ycol (str): 标签列名
        Returns:
            list: 卡方检验筛选后保留的特征列表
            DataFrame: 两列，第一列是所有待筛选特征的名称，第二列是归一化后的卡方检验排序（从大到小）
        """
        # num 的取值范围要小于等于特征总数量
        if num > len(feature_names):
            raise Exception(f'{num}大于 feature_names 的长度')
        # num 的取值范围要大于等于0
        if num < 0:
            raise Exception(f'{num}不能为负数')

        data = data_.copy(deep=True)

        # 调用SelectKBest，使用chi2指标，’all‘表明保留所有特征
        # fit 函数传入两个参数：特征数据和标签数据
        selector = SelectKBest(chi2, k='all').fit(data[feature_names],
                                                  data[ycol])

        # 获取所有待筛选特征的得分
        scores = selector.scores_
        # dict(zip(feature_names, scores)) 获取特征名称和对应的chi2分数的字典
        # 然后将字典转为dataframe个数，总共两列，第一列是特征名称，第二列是对应的chi2分数
        scores = pd.DataFrame(dict(zip(feature_names, scores)), index=[0]).T
        # 将dataframe的索引进行重置
        scores.reset_index(inplace=True)
        # 赋予列名
        scores.columns = ['feature', 'score']
        # 根据score进行降序排序
        scores.sort_values('score', inplace=True, ascending=False)

        # 对score归一化，得到score最小值，最小值减去一个很小的偏置，从而在后面计算的时候避免最小值被归一化为0
        min_val = scores['score'].min() - 0.01
        # 得到score最大值
        max_val = scores['score'].max()
        # 利用最大最小归一化公式进行归一
        scores['score'] = (scores['score'] - min_val) / (max_val - min_val)
        # 将dataframe的索引进行重置
        scores.reset_index(drop=True, inplace=True)

        # 获取筛选后保留的特征名称，head函数保存前num数据，然后获取对应的特征名
        preserved_features = scores.head(num)['feature'].values.tolist()
        return preserved_features, scores

    def select_variance(self, data_, num, feature_names):
        """方差分析

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据
            num (int): 需保留的特征个数
            feature_names (list): 待筛选特征列表
        Returns:
            list: 方差分析筛选后保留的特征列表
            DataFrame: 两列，第一列是所有待筛选特征的名称，第二列是归一化后的方差分析排序（从大到小）
        """
        # num 的取值范围要小于等于特征总数量
        if num > len(feature_names):
            raise Exception(f'{num}大于 numerical_array 的长度')
        # num 的取值范围要大于等于0
        if num < 0:
            raise Exception(f'{num}不能为负数')

        data = data_.copy(deep=True)

        # 调用VarianceThreshold获取所有特征的方差
        selector = VarianceThreshold()
        # fit 函数传入特征数据
        selector.fit(data[feature_names])

        # 获取所有待筛选特征的得分
        scores = selector.variances_
        # dict(zip(feature_names, scores)) 获取特征名称和对应的方差的字典
        # 然后将字典转为dataframe格式，总共两列，第一列是特征名称，第二列是对应的方差
        scores = pd.DataFrame(dict(zip(feature_names, scores)), index=[0]).T
        # 将dataframe的索引进行重置
        scores.reset_index(inplace=True)
        # 赋予列名
        scores.columns = ['feature', 'score']
        # 根据score进行降序排序
        scores.sort_values('score', inplace=True, ascending=False)

        # 对score归一化，得到score最小值，最小值减去一个很小的偏置，从而在后面计算的时候避免最小值被归一化为0
        min_val = scores['score'].min() - 0.01
        # 得到score最大值
        max_val = scores['score'].max()
        # 利用最大最小归一化公式进行归一
        scores['score'] = (scores['score'] - min_val) / (max_val - min_val)
        # 将dataframe的索引进行重置
        scores.reset_index(drop=True, inplace=True)

        # 获取筛选后保留的特征名称，head函数保存前num数据，然后获取对应的特征名
        preserved_features = scores.head(num)['feature'].values.tolist()
        return preserved_features, scores

    def select_pearson(self, data_, num, feature_names):
        """皮尔逊相关系数

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据
            num (int): 需保留的特征个数
            feature_names (list): 待筛选特征列表
        Returns:
            list: 方差分析筛选后保留的特征列表
            DataFrame: 两列，第一列是所有待筛选特征的名称，第二列是归一化后的皮尔逊相关系数排序（从大到小）
        """
        # num 的取值范围要小于等于特征总数量
        if num > len(feature_names):
            raise Exception(f'{num}大于 numerical_array 的长度')
        # num 的取值范围要大于等于0
        if num < 0:
            raise Exception(f'{num}不能为负数')

        data = data_.copy(deep=True)

        # 获取任意两个特征之间的相关度
        scores = data[feature_names].corr()
        # 只取相关的绝对值，去除负相关
        scores = abs(scores)
        # 每个特征和其他特征总的相关度
        scores['score'] = scores.sum(axis=1)
        # 只保留总的相关度 score
        scores = scores[['score']]

        # 将dataframe的索引进行重置，原本处在索引位置的特征名变成列名
        scores.reset_index(inplace=True)
        # 赋予列名
        scores.columns = ['feature', 'score']
        # 根据score进行升序排序，因为相关度是越小越好，越小说明这个特征和其他特征总的相关度越低，冗余性越低
        scores.sort_values('score', inplace=True)

        # 对score归一化，得到score最小值，
        min_val = scores['score'].min()
        # 得到score最大值，最大值加上一个很小的偏置，从而在后面计算的时候避免最大值被归一化为0
        max_val = scores['score'].max() + 0.01
        # 利用最大最小归一化公式进行归一
        scores['score'] = (max_val - scores['score']) / (max_val - min_val)
        # 将dataframe的索引进行重置
        scores.reset_index(drop=True, inplace=True)
        # 获取筛选后保留的特征名称，head函数保存前num数据，然后获取对应的特征名
        preserved_features = scores.head(num)['feature'].values.tolist()
        return preserved_features, scores

    def select_kendall(self, data_, num, feature_names):
        """肯德尔相关系数

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据
            num (int): 需保留的特征个数
            feature_names (list): 待筛选特征列表
        Returns:
            list: 方差分析筛选后保留的特征列表
            DataFrame: 两列，第一列是所有待筛选特征的名称，第二列是归一化后的肯德尔相关系数排序（从大到小）
        """
        # num 的取值范围要小于等于特征总数量
        if num > len(feature_names):
            raise Exception(f'{num}大于 numerical_array 的长度')
            # num 的取值范围要大于等于0
        if num < 0:
            raise Exception(f'{num}不能为负数')

        data = data_.copy(deep=True)
        # 获取任意两个特征之间的相关度
        scores = data[feature_names].corr('kendall')
        # 只取相关的绝对值，去除负相关
        scores = abs(scores)
        # 每个特征和其他特征总的相关度
        scores['score'] = scores.sum(axis=1)
        # 只保留总的相关度 score
        scores = scores[['score']]

        # 将dataframe的索引进行重置，原本处在索引位置的特征名变成列名
        scores.reset_index(inplace=True)
        # 赋予列名
        scores.columns = ['feature', 'score']
        # 根据score进行升序排序，因为相关度是越小越好，越小说明这个特征和其他特征总的相关度越低，冗余性越低
        scores.sort_values('score', inplace=True)

        # 对score归一化，得到score最小值，
        min_val = scores['score'].min()
        # 得到score最大值，最大值加上一个很小的偏置，从而在后面计算的时候避免最大值被归一化为0
        max_val = scores['score'].max() + 0.01
        # 利用最大最小归一化公式进行归一
        scores['score'] = (max_val - scores['score']) / (max_val - min_val)
        # 将dataframe的索引进行重置
        scores.reset_index(drop=True, inplace=True)
        # 获取筛选后保留的特征名称，head函数保存前num数据，然后获取对应的特征名
        preserved_features = scores.head(num)['feature'].values.tolist()
        return preserved_features, scores

    def select(self, data_, num, feature_names, ycol, weight=None):
        """特征筛选（降维）

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据
            num (int): 需保留的特征个数
            feature_names (list): 待筛选特征列表
            ycol (str): 标签列名
            weight (dict): 各个特征筛选方法的权重，默认为 None，权重相等。输入格式为：
            {
                'select_chi2': 0.25,
                'select_variance': 0.25,
                'select_pearson': 0.25,
                'select_kendall': 0.25,
            }
        Returns:
            list: 特征筛选后保留的特征列表
            DataFrame: 两列，第一列是所有待筛选特征的名称，第二列是特征筛选相关分数排名（从大到小）
        """
        # num 的取值范围要小于等于特征总数量
        if num > len(feature_names):
            raise Exception(f'{num}大于 numerical_array 的长度')
        # num 的取值范围要大于等于0
        if num < 0:
            raise Exception(f'{num}不能为负数')

        data = data_.copy(deep=True)

        # 用到的所有特征筛选方式
        select_funcs = [
            'select_chi2', 'select_variance', 
            'select_pearson','select_kendall'
        ]

        # 如果用户没有传入weight参数，则设置默认权重，每个特征筛选方式的权重相等
        if weight is None:
            # 设置为字典
            weight = {}
            # 遍历所有特征筛选方式
            for func in select_funcs:
                # 对应的特征筛选方式的权重为1/总数量
                weight[func] = 1.0 / len(select_funcs)

        # 检查 weight 的关键词是否符合要求，数量要和select_funcs相等，而且包含的内容不能有差异
        if len(weight.keys()) != len(select_funcs) and len(
                set(weight.keys()) - set(select_funcs)) != 0:
            # 否则抛出异常，停止程序
            raise Exception('weight 字典有缺失')

        # 各个方法的分数列表
        score_list = []
        # 遍历所有特征筛选方式
        for fun in select_funcs:
            # 调用对应的特征筛选方式
            if fun == 'select_chi2':
                _, score = self.select_chi2(data, num, feature_names, ycol)
            elif fun == 'select_variance':
                _, score = self.select_variance(data, num, feature_names)
            elif fun == 'select_pearson':
                _, score = self.select_pearson(data, num, feature_names)
            elif fun == 'select_kendall':
                _, score = self.select_kendall(data, num, feature_names)

            # 得到分数排名
            score['rank'] = score.index
            # 排在越前面的代表越好，所以排名要变大
            score['rank'] = len(feature_names) - score['rank'] - 1
            # 根据特征名称排序，方便后面直接 concat 合并
            score.sort_values('feature')
            # 乘上各自方法对应的权重
            score['rank'] = score['rank'] * weight[fun]
            # rank 重命名为 特征筛选方式_rank
            score[f'{fun}_rank'] = score['rank']
            # 设置索引 feature
            score.set_index(['feature'], inplace=True)
            # 删除后面不用的score列和rank列
            del score['score']
            del score['rank']
            # 添加到scores_list列表
            score_list.append(score)

        # 所有方法得到的分数进行横向拼接
        score = pd.concat(score_list, axis=1, sort=True)
        # 计算总的rank数值
        score['rank'] = score.sum(axis=1)
        # 只保留总的rank数值
        score = score[['rank']]

        # 将dataframe的索引进行重置，原本处在索引位置的特征名变成列名
        score.reset_index(inplace=True)
        # 赋予列名
        score.columns = ['feature', 'rank']
        # 根据rank进行降序排序
        score.sort_values('rank', inplace=True, ascending=False)
        # 获取筛选后保留的特征名称，head函数保存前num数据，然后获取对应的特征名
        preserved_features = score.head(num)['feature'].values.tolist()
        return preserved_features, score

    def visualization(self, data_, path):
        """特征的可视化

        Args:
            data_ (DataFrame): 待处理的 DataFrame 数据
            path (String): 输出目录的路径
        Returns:
            None
        """
        # 删除包含空值的数据
        df_record = data_[data_.columns].dropna()
        # 新建各自的绘图的存储文件夹
        for i in ['pdp', 'hist', 'box', 'heat', 'pie', 'bar']:
            os.makedirs(os.path.join(path, 'picture', i), exist_ok=True)

        # 获取所有类型是object的离散特征
        nominal_array = list(df_record.select_dtypes(include=['object']))
        # 获取所有类型是int和float的连续特征
        numerical_array = list(
            df_record.select_dtypes(include=['int', 'float']))
        # 剔除不需要绘图的id特征和label特征
        numerical_array = list(
            filter(lambda x: x not in ['id', 'label'], numerical_array))
        # 剔除不需要绘图的id特征和label特征
        nominal_array = list(
            filter(lambda x: x not in ['id', 'label'], nominal_array))
        # 遍历所有连续特征
        for num in numerical_array:
            # 如果一个连续特征只有不到20个不同的值，则将其视为离散特征
            if len(set(df_record[num])) < 20:
                # 从连续特征列表中删除
                numerical_array.remove(num)
                # 添加到离散特征列表
                nominal_array.append(num)

        # 新建空的画板
        plt.figure()

        # 绘制柱状图
        # 遍历所有的离散特征
        for col in nominal_array:
            # 获取所有独一无二的值作为x轴数据
            x = list(set(df_record[col].values))
            # 坐标 y 列表
            y = []
            # 遍历所有特征值
            for i in x:
                # 获取特征值的数量
                count = df_record[df_record[col] == i].size
                # 添加到y轴数据列表
                y.append(count)
            # 调用bar函数绘制
            plt.bar(x, y)
            # 保存到指定的文件夹下
            plt.savefig(os.path.join(path, 'picture', 'bar', f'{col}_bar.png'))
            # 清空画板
            plt.cla()

        # 绘制拼图
        # 遍历所有的离散特征
        for col in nominal_array:
            # 获取所有独一无二的值作为x轴数据
            x = list(set(df_record[col].values))
            # 坐标 y 列表
            y = []
            # 遍历所有特征值
            for i in x:
                # 获取特征值的数量
                count = df_record[df_record[col] == i].size
                # 添加到y轴数据列表
                y.append(count)
            # 调用pie函数绘制，autopct设置精度
            plt.pie(y, labels=x, autopct='%3.2f%%')
            # 保存到指定的文件夹下
            plt.savefig(os.path.join(path, 'picture', 'pie', f'{col}_pie.png'))
            # 清空画板
            plt.cla()

        # 数值数据直方图
        # 遍历所有的连续特征
        for col in numerical_array:
            # 调用distplot函数绘制
            sns.distplot(df_record[col], kde=False)
            # 保存到指定的文件夹下
            plt.savefig(
                os.path.join(path, 'picture', 'hist', f'{col}_hist.png'))
            # 清空画板
            plt.cla()

        # 数值数据概率密度图
        # 遍历所有的连续特征
        for col in numerical_array:
            # 调用kdeplot函数绘制
            sns.kdeplot(df_record[col])
            # 保存到指定的文件夹下
            plt.savefig(os.path.join(path, 'picture', 'pdp', f'{col}_pdp.png'))
            # 清空画板
            plt.cla()

        # 数值数据箱线图
        # 遍历所有的连续特征
        for col in numerical_array:
            # 调用boxplot函数绘制
            sns.boxplot(y=df_record[col])
            # 保存到指定的文件夹下
            plt.savefig(os.path.join(path, 'picture', 'box', f'{col}_box.png'))
            # 清空画板
            plt.cla()

        # 绘制热图，传入所有离散特征的相关度
        sns.heatmap(df_record[numerical_array].corr())
        # 保存到指定的文件夹下
        plt.savefig(os.path.join(path, 'picture', 'heat', 'heat.png'))
        # 关闭画板
        plt.close()
