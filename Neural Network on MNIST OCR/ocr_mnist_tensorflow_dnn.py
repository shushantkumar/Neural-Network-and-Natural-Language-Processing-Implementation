import tensorflow as tf
#data from mnist
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 1000
n_nodes_hl3 = 1500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
#x = tf.placeholder('float')  this will also work only it will take any format not as 1 * 784
y = tf.placeholder('float')

def neural_network_model(data):

    #tf.Variable means the weights are tensorflow variable
    #assigning random weights initially  
    #(input_data * weight) + biases  ---- 

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    #(input_data * weight) + biases
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)     #passing through activation function

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):

    #basically assign variables and then tensorflow initializez and resuces cost
    prediction = neural_network_model(x)

    #tf.nn.softmax basically used for normalizing like a = tf.constant(np.array([[.1, .3, .5, .9]])) => [[ 0.16838508  0.205666    0.25120102  0.37474789]] such that sum is 1
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ) #calculate diff between prediction we got with actual labels
    
    #minimizing the cost using adam optimizer, it works similar to gradient descent
    #default learning rate AdamOptimizer takes learning rate 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #cycles feedforward + backwardprop
    hm_epochs = 10
    with tf.Session() as sess:
        #standard to initialize all variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):

                #this initializes entire block of that size to x,y 
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                #c is cost that we r trying to reduce
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})  #feeding data with x and y
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        #tf.argmax returns max value in that tensor
        #so here it checks if predicted and actual are equal
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))    #calculates mean
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

#Training data
train_neural_network(x)


