import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#process data
pre_data = np.load('F:/LZB_pre_data/LZB_P2l_6positions_24points.npy')
row_num, col_num, sample_num = np.shape(pre_data)
data_num = row_num*col_num
data = np.zeros([data_num, sample_num])
for i in range(row_num):
        data[(i*col_num):(i*col_num+col_num)] = pre_data[i,:,:]     #三维数据转二维矩阵，采样点数 x 道数

max_val = np.max(data)
min_val = np.min(data)
data = (data-min_val)/(max_val-min_val)     #归一化数据
################################################################################################################################
# G(z)
def generator(x):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('G_w0', [x.get_shape()[1], 64], initializer=w_init)
    b0 = tf.get_variable('G_b0', [64], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # 2nd hidden layer
    w1 = tf.get_variable('G_w1', [h0.get_shape()[1], 128], initializer=w_init)
    b1 = tf.get_variable('G_b1', [128], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3rd hidden layer
    w2 = tf.get_variable('G_w2', [h1.get_shape()[1], 256], initializer=w_init)
    b2 = tf.get_variable('G_b2', [256], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output hidden layer
    w3 = tf.get_variable('G_w3', [h2.get_shape()[1], 144], initializer=w_init)
    b3 = tf.get_variable('G_b3', [144], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h2, w3) + b3)

    return o

# D(x)
def discriminator(x, drop_out):

    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('D_w0', [x.get_shape()[1], 256], initializer=w_init)
    b0 = tf.get_variable('D_b0', [256], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.nn.dropout(h0, drop_out)

    # 2nd hidden layer
    w1 = tf.get_variable('D_w1', [h0.get_shape()[1], 128], initializer=w_init)
    b1 = tf.get_variable('D_b1', [128], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, drop_out)

    # 3rd hidden layer
    w2 = tf.get_variable('D_w2', [h1.get_shape()[1], 64], initializer=w_init)
    b2 = tf.get_variable('D_b2', [64], initializer=b_init)
    feat = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.dropout(h2, drop_out)

    # output layer
    w3 = tf.get_variable('D_w3', [h2.get_shape()[1], 1], initializer=w_init)
    b3 = tf.get_variable('D_b3', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h2, w3) + b3)

    return o, feat

# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 100

# networks : generator
with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(None, 50))
    G_z = generator(z)

# networks : discriminator
with tf.variable_scope('D') as scope:
    drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
    x = tf.placeholder(tf.float32, shape=(None, 144))
    D_real, Feat = discriminator(x, drop_out)
    scope.reuse_variables()
    D_fake, _ = discriminator(G_z, drop_out)


# loss for each network
eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

# trainable variables for each network
t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

# optimizer for each network
D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
start_time = time.time()
G_losses = []
D_losses = []
epoch_start_time = time.time()
for iter in range(data_num // batch_size):
    # update discriminator
    x_ = data[iter*batch_size:(iter+1)*batch_size]
    z_ = np.random.normal(0, 1, (batch_size, 50))

    loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})
    D_losses.append(loss_d_)

    # update generator
    z_ = np.random.normal(0, 1, (batch_size, 50))
    loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
    G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    if iter % 100 == 0:
        print('[%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((iter + 1), per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!")

############################################################################################################################################
features = sess.run(Feat, feed_dict={x : data, drop_out: 1})
np.save('features_GAN_55wX64.npy', features)

##############################################################################################################################################
n_clusters = 6
pred_labels_kmeans = np.array([], dtype=np.int16).reshape(0,)
minibatch_size=10
print("Learning the clusters.")
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(features)
print("Extracting features from val set and predicting from it.")
for ii in range(data_num // minibatch_size):
    X = data[ii*minibatch_size:(ii+1)*minibatch_size] #shape(batchsize,144)
    d_features = sess.run(Feat, feed_dict={x : X, drop_out: 1})
    batch_pred_labels_kmeans = kmeans.predict(d_features)
    pred_labels_kmeans = np.concatenate((pred_labels_kmeans, batch_pred_labels_kmeans))

result = np.reshape(pred_labels_kmeans,[row_num, col_num])
plt.imshow(result)
plt.show()

sess.close()
