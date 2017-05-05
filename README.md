## LSTM_predict_trade

使用LSTM进行预测并进行交易

`LSTM多层模型`：
为两层及两层以上的LSTM模型，使用basiclstm作为每个cell基本单元。使用dropout封装防止过拟和。

`LSTM单层模型`：
为单层的LSTM模型，，使用basiclstm作为每个cell基本单元。

`LSTM窗口归一化版本`：
该版本是在LSTM单层模型的基础上，放弃对训练数据集总体数据的归一化，而改用在每个batch数据里进行归一化。

`LSTM-platform`:
该版本是适用于紫荆平台上的模型，需要注意的是，这里对数据的归一化使用的是对训练数据集的总体归一化，所以在platform中需要载入训练数据集里面的最大最小值。

`test多层LSTM`：
为本地测试多层LSTM训练结果，并增加recall函数来判明我们所关系的非零标签和可交易的预测准确率。

`test单层LSTM`：
为本地测试单层LSTM训练结果，并增加recall函数来判明我们所关系的非零标签和可交易的预测准确率。

`encoder-fearture`：
使用自编码器增加特征。

`data_label_three`:
根据当下时刻的未来t分钟的值进行打标签操作，总共分为三种标签。

`data_label_ts`：
根据当下时刻的未来t分钟内的值进行打标签操作，每一列得到一个t维的0，1，-1向量。


