

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from my_layer import GraphConvolution
from utils import *
import time
import os


#---------------------------------设置参数-------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NB_EPOCH = 1000                  # 迭代次数
PATIENCE = 10                   # 提前停止参数
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
data_path = "D:/科研/python代码/炼丹手册/GCN/datasets/cora/"
save_path = "D:/科研/python代码/炼丹手册/GCN/save_models/gcn.h5"
#-----------------------------------------------------------------------------


#--------------------------------加载数据集-------------------------------------
X, A, y = load_data(data_path)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
#-----------------------------------------------------------------------------


#--------------------------------预处理-------------------------------------
X /= X.sum(1).reshape(-1, 1)     # 特征矩阵归一化
A = preprocess_adj(A)            # 邻接矩阵添加自环
A = normalize_adj(A)             # 邻接矩阵归一化
#-----------------------------------------------------------------------------


#---------------------------------搭建模型-------------------------------------
F = Input(shape=(X.shape[1],))
G = Input(shape=(None,), sparse=True)
H = Dropout(0.5)(F)
H = GraphConvolution(16, activation='relu', kernel_regularizer=l2(5e-4))([H, G])
H = Dropout(0.5)(H)
Y = GraphConvolution(7, activation='softmax')([H, G])
gcn = Model(inputs=[F, G], outputs=Y)

gcn.compile(
    loss='categorical_crossentropy', 
    optimizer=Adam(learning_rate=0.01), 
    metrics=['acc']
    )
gcn.summary()
#-----------------------------------------------------------------------------  


#--------------------------------训练和保存-------------------------------------
wait, best_val_loss = 0, 99999
for epoch in range(1, NB_EPOCH + 1):
    t = time.time()

    gcn.fit(
        x=[X, A], 
        y=y_train, 
        sample_weight=train_mask,
        batch_size=A.shape[0], 
        epochs=1, 
        shuffle=False, 
        verbose=0
        )

    # 预测模型在整个数据集上的输出
    preds = gcn.predict([X, A], batch_size=A.shape[0])

    # 模型在验证集上的损失和准确率
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])

    print(
        "Epoch: {:04d}".format(epoch),
        "train_loss= {:.4f}".format(train_val_loss[0]),     # 在训练集上的损失
        "train_acc= {:.4f}".format(train_val_acc[0]),       # 在训练集上的准确率
        "val_loss= {:.4f}".format(train_val_loss[1]),       # 在验证集上的损失
        "val_acc= {:.4f}".format(train_val_acc[1]),         # 在验证集上的准确率
        "time= {:.4f}".format(time.time() - t)              # 本次迭代的运行时间
        )  

    # 设置early stop
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# 测试和保存
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
gcn.save_weights(save_path)
#-----------------------------------------------------------------------------