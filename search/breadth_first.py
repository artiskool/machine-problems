from search import Search

class BreadthFirst(Search):
    def __init__(self):
        super().__init__()

    """
    Strategy: expand shallowest unexpanded node
    Implementation: the current set of unexpanded nodes, the fringe, is processed as FIFO queue, i.e., new successors go at the end
    """
