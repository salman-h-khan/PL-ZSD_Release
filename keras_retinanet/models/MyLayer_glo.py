from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def sigmoid(self,x):
        "Numerically-stable sigmoid function."
        z = K.exp(-x)
        return 1 / (1 + z)

    def tanh(self,x):
        return K.tanh(x)

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        voc = np.loadtxt('MSCOCO/vocabulary_glo.txt', dtype='float32', delimiter=',')
        vec = np.loadtxt('MSCOCO/word_glo.txt', dtype='float32', delimiter=',')
        vec = vec[:, :65]
        vec = K.variable(vec, dtype='float32')
        voc = K.variable(voc, dtype='float32')
        self.vec = vec
        self.voc = voc
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(int(self.voc.shape[1]), int(self.vec.shape[0])),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        projection = K.dot(K.dot(self.voc, self.kernel), self.vec)
        projection = self.tanh(projection)
        out = K.dot(x, projection)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0],self.output_dim)
