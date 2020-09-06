from search import Search

class UniformCost(Search):
    def __init__(self):
        pass

    """
    Strategy: Expand least-cost unexpanded node
    Implementation: fringe = minheap/priority queue ordered by path cost
    Equivalent to breadth-first if step costs are all equal.
    """
