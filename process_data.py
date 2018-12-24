import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


pre_data = np.load('F:/LZB_pre_data/6positions_24points.npy')
row_num, col_num, sample_num = np.shape(pre_data)#(950,550,144)
data_num = row_num*col_num
data = np.zeros([data_num, sample_num])
for i in range(row_num):
        trace = pre_data[i,:,:]
        data[(i*col_num):(i*col_num+col_num)] = pre_data[i,:,:]     #三维数据转二维矩阵，采样点数 x 道数

#异常值：-7.458588736694416e+28
max_val = np.max(data)
tmp = data.min(1)
min_val = -1
for k in range(data_num):
    if tmp[k] < min_val and tmp[k] >-70000.0:
        min_val = tmp[k]
#min_val = np.min(data)
data = (data-min_val)/(max_val-min_val)     #归一化数据

S=[]
for i in range(data.shape[0]):
        S.append(np.reshape(data[i,:],[24,6],))
S = np.asarray(S)
S=S[:, :,: ,np.newaxis]  #（522500,24,6,1）
print(S.shape)
###############################################################################################################################


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 256, [2, 1], strides=(1, 1), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 128, [2, 3], strides=(3, 3), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 64, [4, 2], strides=(2, 1), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # output layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 1, [4, 3], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv4)

        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 64, [4, 3], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 128, [4, 2], strides=(2, 1), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 256, [2, 3], strides=(3, 3), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # output layer
        conv4 = tf.layers.conv2d(lrelu3, 1, [2, 1], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv4)

        return o, conv4, conv3

#fixed_z_ = np.random.normal(0, 1, (25, 1, 1, 100))
# def show_result(num_epoch, show = False, save = False, path = 'result.png'):
#     test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})
#
#     size_figure_grid = 5
#     fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
#     for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)
#
#     for k in range(size_figure_grid*size_figure_grid):
#         i = k // size_figure_grid
#         j = k % size_figure_grid
#         ax[i, j].cla()
#         ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')
#
#     label = 'Epoch {0}'.format(num_epoch)
#     fig.text(0.5, 0.04, label, ha='center')
#
#     if save:
#         plt.savefig(path)
#
#     if show:
#         plt.show()
#     else:
#         plt.close()

# def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
#     x = range(len(hist['D_losses']))
#
#     y1 = hist['D_losses']
#     y2 = hist['G_losses']
#
#     plt.plot(x, y1, label='D_loss')
#     plt.plot(x, y2, label='G_loss')
#
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#
#     plt.legend(loc=4)
#     plt.grid(True)
#     plt.tight_layout()
#
#     if save:
#         plt.savefig(path)
#
#     if show:
#         plt.show()
#     else:
#         plt.close()

# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 10

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 24, 6, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 50))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits, D_real_feature= discriminator(x, isTrain)
D_fake, D_fake_logits, _ = discriminator(G_z, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

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
print('training start!')
start_time = time.time()

G_losses = []
D_losses = []
epoch_start_time = time.time()
saver=tf.train.Saver(max_to_keep=1)
for iter in range(S.shape[0] // batch_size):
#for iter in range(10):
    # update discriminator
    x_ = S[iter*batch_size:(iter+1)*batch_size]
    z_ = np.random.normal(0, 1, (batch_size, 1, 1, 50))

    loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
    D_losses.append(loss_d_)

    # update generator
    z_ = np.random.normal(0, 1, (batch_size, 1, 1, 50))
    loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
    G_losses.append(loss_g_)

    if iter % 209 == 0:
        saver.save(sess, 'ckpt/seiemic_GAN.ckpt', global_step=iter + 1)
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((iter + 1), per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")

Flat_feat = np.zeros([data_num, 512])
# n_clusters = 8
for i in range(data_num):

    X = S[i:(i+1)]    #不降维切块
    print('X=',X)
    Feat = sess.run(D_real_feature, {x: X, isTrain: False})
    print('Feat=',Feat)
    Flat_feat[i, :]=Feat.flatten()
    #Tmp_feat = Feat.flatten()
    #Flat_feat.append(Feat.flatten())
    #Flat_feat = np.asarray(Flat_feat)
    #Flat_feat = np.concatenate((Flat_feat, Feat.reshape(512,1)))
    #print(Flat_feat[:,ii])
features = Flat_feat
np.save('features2.npy', features)
# print('Flat_feat=',Flat_feat.shape)
# pred_labels_kmeans = np.array([], dtype=np.int16).reshape(0,)
# minibatch_size=10
# print("Learning the clusters.")
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(Flat_feat)
# print("Extracting features from val set and predicting from it.")
# # for ii in range(data_num // minibatch_size):
# #     sample = Flat_feat[ii*minibatch_size:(ii+1)*minibatch_size]
# #     print('sample=',sample.shape)
#     #d_features = sess.run(Feat, feed_dict={x : X, drop_out: 0.3})
# pred_labels_kmeans = kmeans.predict(Flat_feat)
# #pred_labels_kmeans = np.concatenate((pred_labels_kmeans, batch_pred_labels_kmeans))

k = 5
#Feat = sess.run(D_real_feature, {x: S, isTrain: False})
#encode = sess.run(Feat, feed_dict={x:data,drop_out: 0.3})    #把整块数据输入训练好的模型中
#features = np.reshape(Feat,(522500,-1))

# centroides = tf.Variable(tf.slice(tf.random_shuffle(features),[0,0],[k,-1]))
# expanded_features = tf.expand_dims(features, 0)
# expanded_centroides = tf.expand_dims(centroides, 1)
# assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_features, expanded_centroides)), 2), 0)
# means = tf.concat([tf.reduce_mean(tf.gather(features, tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])), 1) for c in range(k)], 0)
#
# update_centroides = tf.assign(centroides, means)  #将means值赋给centroides
#
# y = tf.placeholder('float')
#
# init_op = tf.initialize_all_variables()
# sess.run(init_op)
#
# for step in range(150):
#     _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
#     if step % 10 == 0:
#         print('step %d, new centroides is'%step, centroid_values)
#
# result = np.reshape(assignment_values,[row_num, col_num])
# plt.imshow(result)
# plt.show()

# result = np.reshape(pred_labels_kmeans,[row_num, col_num])
# plt.imshow(result)
# plt.show()


sess.close()
# seismicGAN
