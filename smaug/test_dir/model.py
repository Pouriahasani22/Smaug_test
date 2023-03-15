import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

with sg.Graph(name="my_model", backend="SMV") as graph:
  input_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data((1, 28, 28, 1)))
  conv_weights = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 1)))
  fc_weights = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((10, 6272)))

  # Shape of act: [1, 28, 28, 1].
  act = sg.input_data(input_tensor)
  # After the convolution, shape of act: [1, 32, 28, 28].
  act = sg.nn.convolution(
      act, conv_weights, stride=[1, 1], padding="same", activation="relu")
  # After the max pooling, shape of act: [1, 32, 14, 14].
  act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
  # After the matrix multiply, shape of act: [1, 10].
  act = sg.nn.mat_mul(act, fc_weights)

graph.write_graph()
