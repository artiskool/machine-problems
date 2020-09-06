class Search():

    def __init__(self):
        self.initialState = {}
        self.neighbors = {}
        self.nodes = {}

    # returns a solution or failure
    """
    def generalSearch(self, problem, queueingFunction):
        nodes = self.makeQueue(self.makeNode(self.initialState[problem]))
        while True:
            if nodes is None:
                return False
            node = self.removeFront(nodes)
            if self.goalTest(problem, node):
                return node
            nodes = self.queueingFunction(nodes, self.expand(node, Operators
    """

    """ PSEUDOCODE:
    function GENERAL-SEARCH(problem, QUEUEING-FN) returns a solution, or failure
        nodes = MAKE-QUEUE(MAKE-NODE(INITIAL-STATE[problem]))
        loop do
            if nodes is empty then return failure
            node = REMOVE-FRONT(nodes)
            if GOAL-TEST[problem] applied to STATE(node) succeeds then return node
            nodes = QUEUEING-FN(nodes, EXPAND(node, OPERATORS[problem]))
        end
    """

    def goalTest(self, node, goal):
        return False

    def addNeighbors(self, neighbor1, neighbor2, distance):
        if neighbor1.id not in self.neighbors:
            self.neighbors[neighbor1.id] = []
        self.neighbors[neighbor1.id].append({'id': neighbor2.id, 'distance': distance})
        if neighbor2.id not in self.neighbors:
            self.neighbors[neighbor2.id] = []
        self.neighbors[neighbor2.id].append({'id': neighbor1.id, 'distance': distance})
        if neighbor1.id not in self.nodes:
            self.nodes[neighbor1.id] = neighbor1
        if neighbor2.id not in self.nodes:
            self.nodes[neighbor2.id] = neighbor2
