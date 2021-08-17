import os
import shutil

import matplotlib.pyplot as plt
from terminal_layout import Fore
from terminal_layout.extensions.choice import Choice, StringStyle

from AtoA import AtoA
from AtoG import AtoG
from AtoG_rel import AtoG_rel
from explainer import shapExplainer, limeExplainer
from feature_engine import FeatureEngine


def main_menu():
    # 选项名称，禁止随便更改，否则无法进入对应的模块逻辑
    choices = [
        '空对空 DCS 开火决策离线测试', '空对空 DF 开火决策离线测试', '空对地动作决策离线测试', '空对地动作关联分析离线测试',
        '退出'
    ]

    c = Choice('移动 ↑↓ 光标选择对应的模块 ',
               choices,
               icon_style=StringStyle(fore=Fore.red),
               selected_style=StringStyle(fore=Fore.red))

    choice = c.get_choice()

    if choice:
        return choice[1]
    else:
        return '退出'


def a2a_df():
    # 先让用户输入各种参数，并进行相应的合理性检验，如果是限定候选集的参数，最好在输入提升的括号中进行明示，见下面选择模型类型的例子
    model_type = ''
    df_data_path = ''
    model_val = ''
    k = ''
    percent = '0'
    scale = ''

    while scale not in ['all', 'light']:
        scale = input('请输入模型量级(all or light): ')
        if scale == '':
            scale = 'all'
    while model_type not in [
            'lgb', 'svm', 'nb', 'linear', 'logistic', 'ensemble'
    ]:
        model_type = input(
            '请输入待训练的模型类型(lgb or svm or nb or linear or logistic or ensemble): '
        )
        if model_type == '':
            model_type = 'lgb'
    while not os.path.exists(df_data_path):
        df_data_path = input('请输入数据集路径: ')
        if df_data_path == '':
            df_data_path = 'data/AtoA/DF/fight_01.txt'
    while model_val not in ['hold-out', 'k-fold', 'bootstrap']:
        model_val = input('请输入待训练的模型验证方式(hold-out or k-fold or bootstrap): ')
        if model_val == '':
            model_val = 'hold-out'

    if (model_val == 'hold-out'):
        while not isnumber(
                percent) or float(percent) <= 0 or float(percent) >= 1:
            percent = input('请输入训练集数据比例(大于0且小于1): ')
            if percent == '':
                percent = 0.8
    if (model_val == 'k-fold'):
        while not k.isdigit() or int(k) <= 0 or int(k) >= 10:
            k = input('请输入交叉验证k大小(大于0且小于10): ')
            if k == '':
                k = '5'

    data_path = df_data_path  # 数据路径
    name = model_type  # 模型名称
    val_type = model_val  # 验证方法
    seed = 2021  # 随机种子
    if (model_val == 'k-fold'):
        k_fold = int(k)  # k折交叉验证折数,可选
    else:
        k_fold = 0
    task_type = 'a2a_df'
    save_dir = 'AtoA_' + 'df' + '_offline_' + scale + '_' + name + '_' + val_type

    if os.path.exists(save_dir):
        # 创建前先判断是否存在文件夹,if存在则删除
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)

    offline_AtoA = AtoA(seed=seed, mode='offline', type='df', scale=scale)

    dataClass = FeatureEngine(mode='offline', task_type=task_type)

    data = dataClass.read_A2A_df(data_path)
    FE_data = offline_AtoA.FE_DF(data)
    train, test = offline_AtoA.train_val_split(FE_data)
    model, importance, best_thread = offline_AtoA.train_model(
        train, name, val_type, k_fold, percent_train=float(percent))
    if importance is not None:
        print(importance.head(20))
    offline_AtoA.save_model(model, save_dir, name, val_type, best_thread)
    pred_prob, pred_label = offline_AtoA.offline_predict(model,
                                                         test,
                                                         threshold=best_thread)
    label = test['label']
    dictionary = offline_AtoA.score(label, pred_prob, pred_label, name,
                                    val_type)
    offline_AtoA.evaluate(save_dir, dictionary, name, val_type)
    if model_type == 'lgb':
        index = test[test['label'] == 1].index.tolist()[0]
        test.drop(['id', 'label'], axis=1, inplace=True)
        if scale == 'light':
            feature_names = ['cosValue', 'speedAll', 'distance']
            test = test[feature_names]
        lime = limeExplainer(model, test)
        fig = lime.local_explain(index)
        fig.save_to_file('plot_feature_' + 'lime' + '_predict_lime.html')
        fig.as_pyplot_figure()
        plt.savefig('plot_feature_' + 'predict_lime_detail.png')
        plt.show()
        plt.clf()
        print('lime 变量预测已经完成 ----------')


def a2a_dcs():
    # 先让用户输入各种参数，并进行相应的合理性检验，如果是限定候选集的参数，最好在输入提升的括号中进行明示，见下面选择模型类型的例子
    model_type = ''
    df_data_path = ''
    model_val = ''
    k = ''
    percent = '0'
    scale = ''

    while scale not in ['all', 'light', 'raw']:
        scale = input('请输入模型量级(all or light or raw): ')
        if scale == '':
            scale = 'all'
    while model_type not in [
            'lgb', 'svm', 'nb', 'linear', 'logistic', 'ensemble', 'lstm'
    ]:
        model_type = input(
            '请输入待训练的模型类型(lstm or lgb or svm or nb or linear or logistic or ensemble): '
        )
        if model_type == '':
            model_type = 'lgb'
    while not os.path.exists(df_data_path):
        df_data_path = input('请输入数据集路径: ')
        if df_data_path == '':
            # df_data_path = 'data/AtoA/DCS/o_a2.txt'
            df_data_path = 'data/AtoA/DCS/raw_data'
    while model_val not in ['hold-out', 'k-fold', 'bootstrap']:
        model_val = input('请输入待训练的模型验证方式(hold-out or k-fold or bootstrap): ')
        if model_val == '':
            model_val = 'hold-out'

    if (model_val == 'hold-out'):
        while not isnumber(
                percent) or float(percent) <= 0 or float(percent) >= 1:
            percent = input('请输入训练集数据比例(大于0且小于1): ')
            if percent == '':
                percent = 0.8
    if (model_val == 'k-fold'):
        while not k.isdigit() or int(k) <= 0 or int(k) >= 10:
            k = input('请输入交叉验证k大小(大于0且小于10): ')
            if k == '':
                k = '5'

    data_path = df_data_path  # 数据路径
    name = model_type  # 模型名称
    val_type = model_val  # 验证方法
    seed = 2021  # 随机种子
    if (model_val == 'k-fold'):
        k_fold = int(k)  # k折交叉验证折数,可选
    else:
        k_fold = 0
    task_type = 'a2a_dcs'
    save_dir = 'AtoA_' + 'dcs' + '_offline_' + name + '_' + val_type
    if os.path.exists(save_dir):
        # 创建前先判断是否存在文件夹,if存在则删除
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)
    offline_AtoA = AtoA(seed=seed, mode='offline', type='dcs', scale=scale)

    dataClass = FeatureEngine(mode='offline', task_type=task_type)
    data = dataClass.read_A2A_dcs_new(data_path)
    data = offline_AtoA.FE_DCS_new(data)

    if model_type == 'lstm':
        data = dataClass.normalization(data)

    train, test = offline_AtoA.train_val_split(data)
    model, _, best_thread = offline_AtoA.train_model(
        train, name, val_type, k_fold, percent_train=float(percent))

    if model_type != 'lstm':
        offline_AtoA.save_model(model, save_dir, name, val_type, best_thread)
    pred_prob, pred_label = offline_AtoA.offline_predict(model,
                                                         test,
                                                         threshold=best_thread)

    label = test['label']
    dictionary = offline_AtoA.score(label, pred_prob, pred_label, name,
                                    val_type)
    offline_AtoA.evaluate(save_dir, dictionary, name, val_type)


def a2g():
    # 先让用户输入各种参数，并进行相应的合理性检验，如果是限定候选集的参数，最好在输入提升的括号中进行明示，见下面选择模型类型的例子
    model_type = ''
    df_data_path = ''
    model_val = ''
    k = ''
    scale = ''
    percent = '0'

    while scale not in ['all', 'light']:
        scale = input('请输入模型量级(all or light): ')
        if scale == '':
            scale = 'all'
    while model_type not in [
            'lgb', 'svm', 'nb', 'linear', 'logistic', 'ensemble'
    ]:
        model_type = input(
            '请输入待训练的模型类型(lgb or svm or nb or linear or logistic or ensemble): '
        )
        if model_type == '':
            model_type = 'lgb'
    while not os.path.exists(df_data_path):
        df_data_path = input('请输入数据集路径: ')
        if df_data_path == '':
            df_data_path = 'data/AtoG/raw_data'
    while model_val not in ['hold-out', 'k-fold', 'bootstrap']:
        model_val = input('请输入待训练的模型验证方式(hold-out or k-fold or bootstrap): ')
        if model_val == '':
            model_val = 'hold-out'

    if (model_val == 'hold-out'):
        while not isnumber(
                percent) or float(percent) <= 0 or float(percent) >= 1:
            percent = input('请输入训练集数据比例(大于0且小于1): ')
            if percent == '':
                percent = 0.8
    if (model_val == 'k-fold'):
        while not k.isdigit() or int(k) <= 0 or int(k) >= 10:
            k = input('请输入交叉验证k大小(大于0且小于10): ')
            if k == '':
                k = '5'

    data_path = df_data_path  # 数据路径
    name = model_type  # 模型名称
    val_type = model_val  # 验证方法
    seed = 2021  # 随机种子
    if (model_val == 'k-fold'):
        k_fold = int(k)  # k折交叉验证折数,可选
    else:
        k_fold = 0
    task_type = 'a2g'
    save_dir = 'AtoG' + '_offline_' + scale + '_' + name + '_' + val_type

    if os.path.exists(save_dir):
        # 创建前先判断是否存在文件夹,if存在则删除
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)

    offline_AtoG = AtoG(seed=seed, scale=scale, mode='offline')

    dataClass = FeatureEngine(mode='offline', task_type=task_type)

    data = dataClass.read_A2G(data_path)
    FE_data = offline_AtoG.FE(data)
    renotated_data = offline_AtoG.renotation(FE_data, 0.1, 0.12, 0.12, 0.12,
                                             0.2)

    train, test = offline_AtoG.train_val_split(renotated_data)
    model, importance = offline_AtoG.train_model(train,
                                                 name,
                                                 val_type,
                                                 k_fold,
                                                 percent_train=float(percent))
    if importance is not None:
        print(importance.head(20))
    offline_AtoG.save_model(model, save_dir, name, val_type)
    pred_prob, _, pred_label = offline_AtoG.offline_predict(model, test)
    label = test['label']
    dictionary = offline_AtoG.score(label, pred_prob, pred_label, name,
                                    val_type)
    offline_AtoG.evaluate(save_dir, dictionary, name, val_type)


def a2g_rela():
    # 先让用户输入各种参数，并进行相应的合理性检验，如果是限定候选集的参数，最好在输入提升的括号中进行明示，见下面选择模型类型的例子
    model_type = ''
    df_data_path = ''
    model_val = ''
    k = ''
    percent = '0'
    rel = ''

    while rel not in ['attack', 'locked']:
        rel = input('请输入选择攻击关联决策还是瞄准关联决策(attack or locked): ')
        if rel == '':
            rel = 'attack'
    while model_type not in ['lgb', 'svm', 'br', 'linear', 'ensemble']:
        model_type = input(
            '请输入待训练的模型类型(lgb or svm or br or linear or ensemble): ')
        if model_type == '':
            model_type = 'lgb'
    while not os.path.exists(df_data_path):
        df_data_path = input('请输入数据集路径: ')
        if df_data_path == '':
            df_data_path = 'data/AtoG/raw_data'
    while model_val not in ['hold_out', 'k-fold', 'bootstrap']:
        model_val = input('请输入待训练的模型验证方式(hold_out or k-fold or bootstrap): ')
        if model_val == '':
            model_val = 'hold_out'

    if (model_val == 'hold_out'):
        while not isnumber(
                percent) or float(percent) <= 0 or float(percent) >= 1:
            percent = input('请输入训练集数据比例(大于0且小于1): ')
            if percent == '':
                percent = 0.8
    if (model_val == 'k-fold'):
        while not k.isdigit() or int(k) <= 0 or int(k) >= 10:
            k = input('请输入交叉验证k大小(大于0且小于10): ')
            if k == '':
                k = '5'
    data_path = df_data_path  # 数据路径
    model_name = model_type  # 模型名称
    val_type = model_val  # 验证方法
    seed = 2021  # 随机种子
    if (model_val == 'k-fold'):
        k_fold = int(k)  # k折交叉验证折数,可选
    else:
        k_fold = 0

    save_dir = 'AtoG_rela' + '_offline_' + model_type + '_' + val_type
    # 若不存在保存文件夹，则创建。注意此处无需删除，因瞄准和锁定均需保存在同一文件夹下
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if rel == 'attack':
        fe = FeatureEngine()
        data_attack = fe.read_A2G_rel_attack(data_path)

        offline_AtoG_rel_attack = AtoG_rel(mode='offline', type=rel, seed=seed)
        df_train, df_test = offline_AtoG_rel_attack.train_val_split(
            data_attack)
        df_train = offline_AtoG_rel_attack.get_score_attack(df_train)
        df_test = offline_AtoG_rel_attack.get_score_attack(df_test)
        df_train = offline_AtoG_rel_attack.FE_attack(df_train)
        df_test = offline_AtoG_rel_attack.FE_attack(df_test)
        print(f'开始训练{model_name}模型')
        model, df_importance = offline_AtoG_rel_attack.train_model(
            df_train, model_name, val_type, k_fold, percent)
        if df_importance is not None:
            print(df_importance.head(20))

        print(f'{model_name}模型训练结束')
        print('正在保存模型')
        offline_AtoG_rel_attack.save_model_attack(model, save_dir, model_name,
                                                  val_type)
        print('模型保存结束')
        print('正在保存评估结果')
        pred_score = offline_AtoG_rel_attack.offline_predict(model, df_test)
        label = df_test['score']
        dictionary = offline_AtoG_rel_attack.score(label, pred_score)
        offline_AtoG_rel_attack.evaluate_attack(dictionary, model_name,
                                                val_type, save_dir)
        print('评估结果保存结束')
    elif rel == 'locked':
        fe = FeatureEngine()
        data_locked = fe.read_A2G_rel_locked(data_path)

        offline_AtoG_rel_locked = AtoG_rel(mode='offline', type=rel, seed=seed)
        df_train, df_test = offline_AtoG_rel_locked.train_val_split(
            data_locked)
        df_train = offline_AtoG_rel_locked.get_score_locked(df_train)
        df_test = offline_AtoG_rel_locked.get_score_locked(df_test)
        df_train = offline_AtoG_rel_locked.FE_locked(df_train)
        df_test = offline_AtoG_rel_locked.FE_locked(df_test)
        print(f'开始训练{model_name}模型')
        model, df_importance = offline_AtoG_rel_locked.train_model(
            df_train, model_name, val_type, k_fold)

        if df_importance is not None:
            print(df_importance.head(20))
        print(f'{model_name}模型训练结束')

        print('正在保存模型')
        offline_AtoG_rel_locked.save_model_locked(model, save_dir, model_name,
                                                  val_type)
        print('模型保存结束')
        print('正在保存评估结果')
        pred_score = offline_AtoG_rel_locked.offline_predict(model, df_test)
        label = df_test['score'].values.tolist()
        dictionary = offline_AtoG_rel_locked.score(label, pred_score)
        offline_AtoG_rel_locked.evaluate_locked(dictionary, model_name,
                                                val_type, save_dir)
        print('评估结果保存结束')
    else:
        print('没有这个关联决策')


# 判断是不是小数
def isnumber(aString):
    try:
        float(aString)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    while 1:
        # 展示菜单，获取用户获取的选项
        choice = main_menu()

        if choice == '退出':
            break
        elif choice == '空对空 DCS 开火决策离线测试':
            a2a_dcs()
        elif choice == '空对空 DF 开火决策离线测试':
            a2a_df()
        elif choice == '空对地动作决策离线测试':
            a2g()
        elif choice == '空对地动作关联分析离线测试':
            a2g_rela()

        answer = ''
        while answer != 'y' and answer != 'n':
            answer = input('是否继续(y or n): ')
            if answer == '':
                answer = 'y'
        if answer == 'n':
            break
