# Heavily rely on source at https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/dezero/utils.py
def np_sum_to(x, shape):
  ndim = len(shape)
  lead = x.ndim - ndim
  lead_axis = tuple(range(lead))
  
  axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
  y = x.sum(lead_axis + axis, keepdims=True)
  if lead > 0:
    return y.squeeze(lead_axis)
  return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
  ndim = len(x_shape)
  tupled_axis = axis
  if axis is None:
    tupled_axis = None
  elif not isinstance(axis, tuple):
    tupled_axis = (axis,)
  
  if not (ndim == 0 or tupled_axis is None or keepdims):
    actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
    shape = list(gy.shape)
    for a in sorted(actual_axis):
      shape.insert(a, 1)
  else:
    shape = gy.shape
  
  gy = gy.reshape(shape)  # reshape
  return gy
