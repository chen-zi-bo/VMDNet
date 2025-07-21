import torch.nn as nn

from torch_timeseries.normalizations import *


class Model(nn.Module):

    def __init__(self, f_model_type, forecast_model: nn.Module, norm_model: nn.Module):
        super().__init__()
        # 变量赋值
        self.f_model_type = f_model_type
        self.fm = forecast_model
        self.nm = norm_model
        # 实现归一化
    def normalize(self, batch_x, batch_x_enc=None, dec_inp=None, dec_inp_enc=None):
        # normalize
        # input: B T N
        # output: B, T, N

        dec_inp = dec_inp
        # 如果归一化是Revin
        # 如果type里有former，forward就三个参数(batch_x, 'n', dec_inp)，否则就一个参数
        # n代表要做归一化操作
        if isinstance(self.nm, RevIN):
            batch_x, dec_inp = self.nm(batch_x, 'n', dec_inp) if 'former' in self.f_model_type else self.nm(batch_x)
        # 如果为SAN，直接forward获得结果
        elif isinstance(self.nm, SAN):
            batch_x, self.pred_stats = self.nm(batch_x)
            # 如果是Dishts，和Revin处理一样
        elif isinstance(self.nm, DishTS):
            batch_x, dec_inp = self.nm(batch_x, 'n', dec_inp) if 'former' in self.f_model_type else self.nm(batch_x)
        # No则不做事情
        elif isinstance(self.nm, No):
            pass
        # FAN走这里
        else:
            # 调用FAN的forward，传入batch_x
            batch_x = self.nm(batch_x)
        # 返回结果
        return batch_x, dec_inp
# 逆归一化
    def denormalize(self, pred):
        # denormalize
        # d代表要做逆归一化操作
        # 根据类型调用其对应的逆归一化操作
        if isinstance(self.nm, RevIN):
            pred = self.nm(pred, 'd')
        elif isinstance(self.nm, No):
            pass
        elif isinstance(self.nm, SAN):
            pred = self.nm(pred, 'd', self.pred_stats)
        elif isinstance(self.nm, DishTS):
            pred = self.nm(pred, 'd')

        # Fan走这里，走逆归一化
        else:
            pred = self.nm(pred, 'd')
        # 逆归一化的结果应该即为预测值
        return pred

    def forward(self, batch_x, batch_x_enc=None, dec_inp=None, dec_inp_enc=None):

        # normalize
        # if self.f_model_type == "RevIN":
        #     batch_x, dec_inp = self.nm(batch_x)  if 'former' in self.f_model_type  else  self.nm(batch_x, 'n', dec_inp, dec_inp_enc)
        # elif self.f_model_type == "SAN":
        #     batch_x, pred_stats = self.nm(batch_x) 
        # elif self.f_model_type == "DishTS":
        #     batch_x, dec_inp =self.nm(batch_x)  if 'former' in self.f_model_type  else  self.nm(batch_x, 'n', dec_inp, dec_inp_enc)
        # else:
        #     pass


        # 根据former是否在type里，不同参数调用预测模型的forward
        if 'former' in self.f_model_type:
            pred = self.fm(batch_x, batch_x_enc, dec_inp, dec_inp_enc)
        else:
            pred = self.fm(batch_x)

        # denormalize
        # if self.f_model_type == "RevIN":
        #     pred = self.nm(pred, 'd') 
        # elif self.f_model_type == "SAN":
        #     pred = self.nm(pred, 'd', self.pred_stats)
        # elif self.f_model_type == "DishTS":
        #     pred = self.nm(pred, 'd') 
        # else:
        #     pass

        return pred
