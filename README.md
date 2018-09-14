from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parametros
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Datos de entrenamiento
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Entrada de Grafico
X = tf.placeholder("float")
Y = tf.placeholder("float")

# pesos del modelo
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# construimos el modelo lineal
pred = tf.add(tf.multiply(X, W), b)

# error medio cuadrado
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# descenso de gtadiente
# Nota, minimize () sabe modificar W y b porque los objetos Variable son entrenables = Verdadero por defecto
optimizer = tf.train.GradientDescentOptimizer (learning_rate) .minimize (costo)

# Inicializamos Variable (i.e. assign their default value)
init = tf.global_variables_initializer()

# comienza entrenamiento
with tf.Session() as sess:

    # ejecute el icializador
    sess.run(init)

    # Ajustar todos los datos de entrenamiento
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Mostrar registros por cada paso de época
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimización finalizada!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Visualización gráfica
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
    
    #RESULTADOS
    Epoch: 0050 cost= 0.121231794 W= 0.36728987 b= -0.04523236
Epoch: 0100 cost= 0.116115995 W= 0.36028993 b= 0.0051248
Epoch: 0150 cost= 0.111591235 W= 0.35370627 b= 0.05248706
Epoch: 0200 cost= 0.107589230 W= 0.34751424 b= 0.0970323
Epoch: 0250 cost= 0.104049616 W= 0.34169045 b= 0.1389283
Epoch: 0300 cost= 0.100919016 W= 0.33621296 b= 0.17833258
Epoch: 0350 cost= 0.098150164 W= 0.33106136 b= 0.21539338
Epoch: 0400 cost= 0.095701307 W= 0.3262161 b= 0.2502499
Epoch: 0450 cost= 0.093535513 W= 0.32165894 b= 0.28303325
Epoch: 0500 cost= 0.091619983 W= 0.31737286 b= 0.31386736
Epoch: 0550 cost= 0.089925952 W= 0.31334165 b= 0.34286717
Epoch: 0600 cost= 0.088427752 W= 0.30955032 b= 0.37014213
Epoch: 0650 cost= 0.087102793 W= 0.30598438 b= 0.39579502
Epoch: 0700 cost= 0.085931025 W= 0.30263057 b= 0.41992193
Epoch: 0750 cost= 0.084894821 W= 0.29947627 b= 0.4426139
Epoch: 0800 cost= 0.083978407 W= 0.29650944 b= 0.4639568
Epoch: 0850 cost= 0.083168015 W= 0.2937191 b= 0.4840307
Epoch: 0900 cost= 0.082451403 W= 0.29109472 b= 0.50291
Epoch: 0950 cost= 0.081817731 W= 0.28862652 b= 0.520666
Epoch: 1000 cost= 0.081257381 W= 0.286305 b= 0.5373666
Optimization Finished!
Training cost= 0.08125738 W= 0.286305 b= 0.5373666

#Gráficas
![graficas](https://user-images.githubusercontent.com/16944756/45560881-ea4f0b80-b80b-11e8-9e7a-827390ee7b39.png)



