import warnings

import shap
import lime.lime_tabular

warnings.filterwarnings('ignore')


class shapExplainer:
    def __init__(self, model, X):
        """
        初始化
        :param model:已拟合的模型
        :param X:全量数据
        """
        self.model = model
        self.X = X

    def global_explain(self):
        """
        全局解释,生成全局特征重要性图、蜂群图、边际依赖图
        :return: None
        """
        feats = [
            'pre_2_Position_x', 'pre_2_Position_y', 'pre_1_Position_x',
            'pre_1_Position_y', 'Position_y', 'Position_x', 'MyAlt',
            'AngleAttack'
        ]  # 轻量级八个特征
        explainer = shap.Explainer(self.model.predict, self.X)  # 构造解释器
        print('计算shap value')
        # 计算shap value为NP问题，时间复杂度高，只选取前三批数据进行解释
        shap_values = explainer(self.X[:4313])
        # 生成全局特征重要性图
        shap.plots.bar(shap_values)
        # 生成蜂群图
        shap.summary_plot(shap_values=shap_values.values,
                          features=shap_values.data,
                          feature_names=feats,
                          plot_type='dot')
        # 生成八个特征边际依赖图
        for feat in feats:
            shap.plots.partial_dependence(feat,
                                          self.model.predict,
                                          self.X[:4313],
                                          model_expected_value=True,
                                          feature_expected_value=True,
                                          ice=False,
                                          shap_values=shap_values[0:1, :])


class limeExplainer:
    def __init__(self, model, train):
        self.model = model
        self.train = train

    def local_explain(self, index):
        feature_names = self.train.columns.values
        class_names = [0, 1]
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.train.values,
            feature_names=feature_names,
            class_names=class_names)
        exp = explainer.explain_instance(self.train.loc[index],
                                         self.model.predict_proba)
        return exp
