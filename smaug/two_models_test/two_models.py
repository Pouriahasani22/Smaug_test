import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

with sg.Graph(name="my_model", backend="SMV") as graph:
  input_tensor_1 = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data((1, 28, 28, 1)))
  conv_weights_1 = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 32, 1)))
  fc_weights_1 = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, 6272)))

  input_tensor_2 = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data((1, 16, 16, 1)))
  conv_weights_2 = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data((4, 4, 4, 1)))
  fc_weights_2 = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, 256)))


  # Shape of act: [1, 28, 28, 1].
  act_1 = sg.input_data(input_tensor_1,name="input1")
  # After the convolution, shape of act: [1, 32, 28, 28].
  act_1 = sg.nn.convolution(
      act_1, conv_weights_1, stride=[1, 1], padding="same", activation="relu",name="conv1")
  # After the max pooling, shape of act: [1, 32, 14, 14].
  act_1 = sg.nn.max_pool(act_1, pool_size=[2, 2], stride=[2, 2],name="max1")
  # After the matrix multiply, shape of act: [1, 10].
  act_1 = sg.nn.mat_mul(act_1, fc_weights_1,name="mat1")


  # Shape of act: [1, 28, 28, 1].
  act_2 = sg.input_data(input_tensor_2,name="input2")
  # After the convolution, shape of act: [1, 32, 28, 28].
  act_2 = sg.nn.convolution(
      act_2, conv_weights_2, stride=[1, 1], padding="same", activation="relu",name="conv2")
  # After the max pooling, shape of act: [1, 32, 14, 14].
  act_2 = sg.nn.max_pool(act_2, pool_size=[2, 2], stride=[2, 2],name="max2")
  # After the matrix multiply, shape of act: [1, 10].
  act_2 = sg.nn.mat_mul(act_2, fc_weights_2,name="mat2")

graph.write_graph()


