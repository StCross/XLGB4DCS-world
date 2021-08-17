import warnings

from AtoG_rel import AtoG_rel
from feature_engine import FeatureEngine

warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, path, type='attack', return_label=False):
        fe = FeatureEngine()

        if type == 'attack':
            data = fe.read_A2G_rel_attack(path)
        else:
            data = fe.read_A2G_rel_locked(path)

        # 只能一批数据，选出一个批id
        id_list = set(data['id'].values.tolist())
        id_list = list(id_list)
        id_list.sort()
        id = id_list[0]
        print(f'初始化数据源成功，选择批号{id}')
        # 筛选出这批下的所有数据
        data = data[data['id'] == id]
        # 删除一些在线场景不能获取到的特征
        for f in ['id']:
            del data[f]
        # 如果 return_label 为 false， 删除数据中的label
        if not return_label:
            del data['label']

        data = data.to_dict(orient='records')

        self.max_size = len(data)
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.max_size:
            x = self.data[self.index]

            self.index += 1
            return x
        else:
            raise StopIteration


type = 'attack'

# 初始化数据加载器，模拟在线场景下的数据流式输入
dataloader = DataLoader('data/AtoG/raw_data', type, return_label=True)
# 初始化质量评价系统，在任务开始之前进行
fire = AtoG_rel(mode='online', seed=2020, type=type)
# 加载训练好的模型
model = fire.load_model(
    'AtoG_rela_offline_lgb_hold_out/model_save_attack/lgb_hold_out_.pkl')

# 任务进行中，不断读取数据
for x in dataloader:
    if type == 'attack':
        row = fire.online_FE_attack(x)
    else:
        row = fire.online_FE_locked(x)
    pred = fire.online_predict(model, row)

    # ###### 测试代码，真实在线场景用不上 #############
    print(f"预测分数: {pred}")
    # ###### 测试代码，真实在线场景用不上 #############
