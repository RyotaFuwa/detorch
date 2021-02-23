import detorch.functional as F


class _Reduction:
  @staticmethod
  def get_reduction_from_str(name: str):
    table = {
      'mean': F.mean,
      'sum': F.sum,
      'none': lambda x: x,
    }
    if name not in table:
      raise KeyError('no reduction found with the name')
    return table[name]
