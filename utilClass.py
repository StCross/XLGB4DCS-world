from tensorflow import keras
from sklearn.metrics import roc_auc_score

class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=1024):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
 
    def on_batch_begin(self, batch, logs={}):
        pass
 
    def on_batch_end(self, batch, logs={}):
        pass
 
    def on_train_begin(self, logs={}):
        if not ('val_roc_auc' in self.params['metrics']):
            self.params['metrics'].append('val_roc_auc')
 
    def on_train_end(self, logs={}):
        pass
 
    def on_epoch_begin(self, epoch, logs={}):
        pass
 
    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc'] = float('-inf')
        if (self.validation_data):
            logs['roc_auc'] = roc_auc_score(self.validation_data[1],
                                            self.model.predict(self.validation_data[0],
                                                               batch_size=self.predict_batch_size))
            print('ROC_AUC - epoch:%d - score:%.6f' % (epoch + 1, logs['roc_auc']))