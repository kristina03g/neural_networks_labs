from builtins import range
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # получаем количество классов и размер обучающей выборки
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # проходим по всем элементам обучающей выборки
    for i in range(num_train):
        scores = X[i].dot(W) # вычисляем оценки для каждого класса
        correct_class_score = scores[y[i]] # получаем оценку правильного класса

        # вычисляем сумму экспонент для softmax
        sum_j = 0.0
        for j in range(num_classes):
            sum_j += np.exp(scores[j])

        # рассчитываем градиент и потери
        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j]) * X[i]) / sum_j
            if (j == y[i]):
                dW[:, y[i]] -= X[i]

        # обновляем потери
        loss += -correct_class_score + np.log(sum_j)

    # усредняем потери и градиент
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # получаем количество классов и размер обучающей выборки
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W) # вычисляем оценки для всех обучающих примеров
    correct_class_scores = scores[range(num_train), y].reshape((num_train, 1)) # выделяем оценки правильных классов
    sum_j = np.sum(np.exp(scores), axis=1).reshape((num_train, 1)) # вычисляем сумму экспонент для softmax

    loss = np.sum(-1 * correct_class_scores + np.log(sum_j)) / num_train + 0.5 * reg * np.sum(W * W) # вычисляем потери

    # создаем матрицу для правильных классов
    correct_matrix = np.zeros(scores.shape)
    correct_matrix[range(num_train), y] = 1

    dW = X.T.dot(np.exp(scores) / sum_j) - X.T.dot(correct_matrix) # вычисляем градиент
    dW = dW / num_train + W * reg # усредняем градиент и добавляем регуляризацию

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
