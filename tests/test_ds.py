import unittest

from ds import *


class HeapTest(unittest.TestCase):
  def test_heap_init(self):
    array = [0, 2, 9, -3, 50, 14, 9, 7, 14, 28, 54, -39, 932, 0, -1, 2, 72]
    
    heap = Heap(array, deepcopy=True)
    self.assertTrue(heap.peek() == -39)
    heap.push(-59)
    self.assertTrue(heap.peek() == -59)
    heap.pop()
    
    sortedArrayByHeap = []
    for _ in range(len(heap)):
      sortedArrayByHeap.append(heap.pop())
    self.assertTrue(sorted(array) == sortedArrayByHeap)
  
  def test_heap_key(self):
    
    def sort_by_heap(array, key):
      heap = Heap(key=key)
      for num in array:
        heap.push(num)
      sortedArray = []
      for _ in range(len(heap)):
        sortedArray.append(heap.pop())
      return sortedArray
    
    key = lambda x: -x
    array = [10, 3, -93, 10932, 14, 14, 28, 39, -4, 4]
    sortedArray = list(reversed(sorted(array)))
    sortedByHeap = sort_by_heap(array, key)
    print(sortedArray, sortedByHeap)
    self.assertTrue(sortedArray == sortedByHeap)
