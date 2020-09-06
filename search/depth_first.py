from search import Search

class DepthFirst(Search):
    def __init__(self):
        pass

    """
    Strategy: Expand deepest unexpanded node
    Implementation: fringe = LIFO queue (aka stack), i.e., put successors at front
    """
