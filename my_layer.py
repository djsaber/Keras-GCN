
from keras import activations, initializers, constraints
from keras import regularizers
from keras.layers import Layer
import keras.backend as K


class GraphConvolution(Layer):
    """Keras自定义层要实现:
        - build()方法
        - call()方法
        - compute_output_shape()方法
    """
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
        ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # 权重初始化
        self.kernel_initializer = initializers.get(kernel_initializer)
        # 偏置初始化
        self.bias_initializer = initializers.get(bias_initializer)
        # 施加在权重上的正则项
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # 施加在偏置向量上的正则项
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # 施加在输出上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # 对主权重矩阵进行约束
        self.kernel_constraint = constraints.get(kernel_constraint)
        # 对偏置向量进行约束
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        super(GraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        """
        计算输出的形状
        如果自定义层更改了输入张量的形状，则应该在这里定义形状变化的逻辑
        让Keras能够自动推断各层的形状
        """
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape

    def build(self, input_shapes):
        """
        定义层中的参数，为层创建可训练的权重
        """
        super(GraphConvolution, self).build(input_shapes)
        # 输入维度
        features_shape = input_shapes[0]
        input_dim = features_shape[1]
        # 权重
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,                            
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
            )
        # 偏置
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,            
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
                )

    def call(self, inputs):
        """
        编写层的功能逻辑
        """
        # 特征矩阵，邻接矩阵
        features, A = inputs 
        # A * X * W
        output = K.dot(K.dot(A, features), self.kernel)
        # A * X * W + b
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        return self.activation(output)