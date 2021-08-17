import warnings

from sklearn.metrics import f1_score, accuracy_score

from AtoG import AtoG
from feature_engine import FeatureEngine

warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, path, return_label=False):
        fe = FeatureEngine(mode='online', task_type='a2g')
        data = fe.read_A2G(path)

        # 只能一批数据，选出一个批id
        id_list = set(data['id'].values.tolist())
        id_list = list(id_list)
        id_list.sort()
        id = 1614132230
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


# 初始化数据加载器，模拟在线场景下的数据流式输入
dataloader = DataLoader('data/AtoG/raw_data', return_label=True)
# 初始化火控系统，在任务开始之前进行
fire = AtoG(mode='online', seed=2020, scale='light')
# 加载训练好的模型
model = fire.load_model('AtoG_offline_light_lgb_hold-out/lgb_hold-out.pkl')

# ###### 测试代码，真实在线场景用不上 #############
label_list = []
pred_label_list = []
pred_prob_list = []
# ###### 测试代码，真实在线场景用不上 #############

# 任务进行中，不断读取数据
for x in dataloader:
    row = fire.online_FE(x)
    pred_prob, _, pred_label = fire.online_predict(model, row)

    # ###### 测试代码，真实在线场景用不上 #############
    label_list.append(x['label'])
    pred_label_list.append(pred_label)
    pred_prob_list.append(pred_prob)
    print(f"真实标签: {x['label']}\t预测标签: {pred_label}\t预测概率: {pred_prob}")
    # ###### 测试代码，真实在线场景用不上 #############

# ###### 测试代码，真实在线场景用不上 #############
macro_f1 = f1_score(label_list, pred_label_list, average='macro')
acc = accuracy_score(label_list, pred_label_list)
# 多类预测，采用macro-f1指标
print(f'macro-f1: {macro_f1}')
print(f'acc: {acc}')
# ###### 测试代码，真实在线场景用不上 #############
