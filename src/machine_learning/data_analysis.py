import io
import itertools

import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
import numpy as np


def show_history(history, is_accuracy=False):
    """
    展示训练过程
    :param is_accuracy: 是否使用了准确率
    :param history: 训练历史
    :return: 无
    """

    if is_accuracy:
        # 绘制训练 & 验证的准确率值
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_confusion_matrix(y_true: [], y_pred: [], class_names: []):
    """
    创建一个混淆矩阵，直观地表示错误分类的图像
    :param y_true: 真实值
    :param y_pred: 预测值
    :param class_names: 类别名称列表
    :return: 无
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print(cm)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def to_2d_show(vectors, labels, early_exaggeration=12):
    """
    将高维度的样本矩阵转换为2维观察
    :param vectors: 二维矩阵，一行表示一个样本，一列表示一个特征;
    :param labels: 标签列表
    :param early_exaggeration: 样本点显示的距离
    :return: 无
    """
    random_state = 40
    verbose = 1
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=random_state, verbose=verbose, method='exact',
                         early_exaggeration=early_exaggeration)
    vectors_tsne = tsne.fit_transform(vectors)
    print("[complete!] Org data dimension is {}. Embedded data dimension is {}".format(vectors.shape[-1],
                                                                                       vectors_tsne.shape[-1]))
    # 绘图
    # 归一化
    x_min, x_max = vectors_tsne.min(0), vectors_tsne.max(0)
    X_norm = (vectors_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


def to_3d_show(vectors, labels, early_exaggeration=1):
    """
    将高维度的样本矩阵转换为3维观察
    :param vectors: 二维矩阵，一行表示一个样本，一列表示一个特征;
    :param labels: 标签列表
    :param early_exaggeration: 样本点显示的距离
    :return: 无
    """
    random_state = 40
    verbose = 1
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=random_state, verbose=verbose, method='exact',
                         early_exaggeration=early_exaggeration)
    vectors_tsne = tsne.fit_transform(vectors)
    print("[complete!] Org data dimension is {}. Embedded data dimension is {}".format(vectors.shape[-1],
                                                                                       vectors_tsne.shape[-1]))
    # 绘图
    # 归一化
    x_min, x_max = vectors_tsne.min(0), vectors_tsne.max(0)
    X_norm = (vectors_tsne - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(X_norm.shape[0]):
        ax.text(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], str(labels[i]), color=plt.cm.Set1(labels[i]),
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    iris = datasets.load_iris()

    to_2d_show(iris.data, iris.target)
    to_3d_show(iris.data, iris.target)
