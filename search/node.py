import copy

class Node(object):

    def __init__(self, id, name, parent = None, cost = 0, distance = 0, nodes = []):
        self.id = id # node id
        self.name = name # name/label of node
        self.parent = parent # parent node object
        self.cost = cost # distance from its parent
        self.distance = distance # straight-line distance to its goal
        self.nodes = nodes # list of immediate node objects
        #if parent is not None:
        #    parent.addNode(self, self.cost)

    def addNode(self, node, cost):
        node.parent = self
        node.cost = cost
        self.nodes.append(node)
        #if not self.hasNode(node):
        #    self.nodes.append(node)

    def hasNode(self, node):
        if node in self.nodes:
            return True
        return False

    def copy(self):
        return copy.deepcopy(self)
