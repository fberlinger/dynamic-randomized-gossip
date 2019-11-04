import math

class Heap():
    """A zero-indexed min heap that keeps track of vertex positions in the heap
    
    Attributes:
        heap (list): Verteces ordered by minimum value
        pos (int): Heap position of a given vertex
        size (int): Heap size (zero-indexed)
    """

    def __init__(self, N):
        """Summary
        """
        self.heap = [] # list of list containing vertex/value pairs
        self.size = -1
        self.inf = 1000000
        self.pos = [self.inf]*N # heap position of given vertex

    def min_heapify(self, N):
        """Min heapifies the heap, i.e. promotes vertex at position N downwards until it is smaller than its children
        
        Args:
            N (int): Heap position from which to heapify
        """
        # swap with smallest child until smallest
        #print(self.heap)
        #print('\n')

        if 2*N+1 <= self.size and self.heap[2*N+1][1] < self.heap[N][1]:
            smallest = 2*N+1
        else:
            smallest = N
        if 2*N+2 <= self.size and self.heap[2*N+2][1] < self.heap[smallest][1]:
            smallest = 2*N+2

        if smallest != N:
            child = self.heap[smallest]
            self.heap[smallest] = self.heap[N] # child to parent
            self.pos[self.heap[smallest][0]] = smallest # heap position
            self.heap[N] = child # parent to child
            self.pos[self.heap[N][0]] = N # heap position
            self.min_heapify(smallest)

    def insert(self, obj, value, pos):
        """Inserts a new vertex/value pair into the heap as a leaf and promotes it upwards until larger than its parent
        
        Args:
            obj (int): Vertex number
            value (int): Vertex distance
        """
        # add obj/value pair as a leaf, increment heap size
        self.heap.append([obj, value, pos])
        self.size += 1

        # promote new pair upwards until larger than parent (swap!)
        N = self.size
        while N != 0 and self.heap[int(math.floor((N-1)/2))][1] > value:
            parent = self.heap[int(math.floor((N-1)/2))]
            self.heap[int(math.floor((N-1)/2))] = self.heap[N] # parent to child
            self.heap[N] = parent # child to parent
            self.pos[self.heap[N][0]] = N # heap position
            N = int(math.floor((N-1)/2))
        self.pos[obj] = N # heap position

    def delete_min(self):
        """Delete the root (minimum element in the heap). Put the last leaf as a new root and min heapify
        
        Returns:
            TYPE: Description
        """
        # safe min, replace with last leaf, decrement heap size, min_heapify
        minn = self.heap[0]
        self.pos[minn[0]] = self.inf # set min pos to out of heap position
        self.heap[0] = self.heap.pop()
        self.pos[self.heap[0][0]] = 0 # reset leaf heap position
        self.size -= 1
        self.min_heapify(0)
        return minn