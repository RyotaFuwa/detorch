class Heap:
  def __init__(self, array=None, key=None, deepcopy=False):
    if array and deepcopy:
      self.array = array[:]
    else:
      self.array = array if array is not None else []
    self.key = key if key is not None else lambda x: x
    self.heapify()
  
  def __len__(self):
    return len(self.array)
  
  def peek(self):
    return self.array[0]
  
  def push(self, node):
    self.array.append(node)
    self._shiftUp(len(self.array) - 1)
  
  def pop(self):
    self.array[0], self.array[-1] = self.array[-1], self.array[0]
    priority = self.array.pop()
    self._shiftDown(0)
    return priority
  
  def heapify(self):
    middle = len(self.array) // 2
    for i in reversed(range(middle)):
      self._shiftDown(i)
  
  def _shiftUp(self, idx):
    if idx == 0:
      return
    if self.key(self.array[idx]) < self.key(self.array[(idx - 1) // 2]):
      self._swap(idx, (idx - 1) // 2)
      self._shiftUp((idx - 1) // 2)
  
  def _shiftDown(self, idx):
    left = 2 * idx + 1
    right = 2 * idx + 2
    if left > len(self.array) - 1:
      return
    elif right > len(self.array) - 1:
      candidate = left
    else:
      candidate = self._min(left, right)
    
    if self._compare(candidate, idx):
      self._swap(candidate, idx)
      self._shiftDown(candidate)
  
  def _swap(self, i, j):
    if 0 <= i < len(self.array) and 0 <= j < len(self.array):
      self.array[i], self.array[j] = self.array[j], self.array[i]
    else:
      raise IndexError()
  
  def _compare(self, i, j):
    if 0 <= i < len(self.array) and 0 <= j < len(self.array):
      return self.key(self.array[i]) < self.key(self.array[j])
    else:
      raise IndexError()
  
  def _min(self, i, j):
    return i if self._compare(i, j) else j
