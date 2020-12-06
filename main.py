from cmath import sqrt
from time import sleep
import json
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from model import Perceptron, Adaline


class Machine:
  DEFAULT_SIZE = 20

class Vertex(Machine):
  def __init__(self):
    self.parent = None
    self.children = []
    self.value = None
    self.label = None
    self.distance = 0

  def toJSONChildren(self):
    children = []
    for child in self.children:
      children.append(child.toJSON())
    return children

  def toJSON(self):
    return {
      'parent': self.parent,
      'children': self.toJSONChildren(),
      'value': self.value,
      'label': self.label,
      'distance': self.distance
    }

class Point(Machine):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def toJSON(self):
    return {'x': self.x, 'y': self.y}


class Box(Machine):
  def __init__(self, left, top, right, bottom):
    self.left = left
    self.top = top
    self.right = right
    self.bottom = bottom

  def toJSON(self):
    return {'left': self.left, 'top': self.top, 'right': self.right, 'bottom': self.bottom}


class Edge(Machine):
  def __init__(self, canvas, tag, x1, y1, x2, y2, color, width=3, with_text=True):
    self.point = Point(x1, y1)
    self.point2 = Point(x2, y2)
    self.distance = round(sqrt((x2 - x1)**2 + (y2 - y1)**2).real)
    self.graph = canvas.create_line(self.point.x, self.point.y, self.point2.x, self.point2.y, fill=color, width=width, tags=tag)
    px = (self.point.x + self.point2.x) / 2
    py = (self.point.y + self.point2.y) / 2
    # create label distance
    self.text = canvas.create_text(px, py, text=self.distance, font=('Monaco', 14, 'italic'), fill='red') if with_text else None
    self.canvas = canvas

  def toJSON(self):
    return {'point': self.point.toJSON(), 'point2': self.point2.toJSON()}


class Node(Machine):
  def __init__(self, canvas, tag, x, y, label, color='red', with_text=True):
    self.point = Point(x, y)
    self.point2 = Point(x + self.DEFAULT_SIZE, y + self.DEFAULT_SIZE)
    self.box = Box(self.point.x, self.point.y, self.point.x + self.DEFAULT_SIZE, self.point.y + self.DEFAULT_SIZE)
    self.graph = canvas.create_oval(self.box.left, self.box.top, self.box.right, self.box.bottom, fill=color, tags=tag)
    self.text = canvas.create_text(self.point.x, self.point.y, text=label, font=('Monaco', 10, 'italic')) #if with_text else None
    self.heuristic = None
    self.heuristicValue = None
    self.canvas = canvas

  def toJSON(self):
    return {'point': self.point.toJSON(), 'point2': self.point2.toJSON(), 'box': self.box.toJSON()}


class Form(Machine):
  MENU_FILE_NEW = 1
  MENU_FILE_OPEN = 2
  MENU_FILE_SAVE = 3
  MENU_FILE_EXIT = 4
  MENU_GRAPH_NODE = 5
  MENU_GRAPH_EDGE = 6
  MENU_SEARCH_BREADTH_FIRST = 7
  MENU_SEARCH_DEPTH_FIRST = 8
  MENU_SEARCH_UNIFORM_COST = 9
  MENU_SEARCH_BEST_FIRST = 10
  MENU_SEARCH_A_STAR = 11
  MENU_MAP_COLORING = 12
  MENU_TRAVELING_SALESMAN_PROBLEM = 13
  MENU_ANN = 14

  SORT_QUEUE = 1
  SORT_STACK = 2
  SORT_PRIORITY_QUEUE = 3
  SORT_PRIORITY_QUEUE_BFS = 4
  SORT_PRIORITY_QUEUE_ASS = 5

  def __init__(self, title='Machine Problems', width=1024, height=768):
    self.reset(True)
    self.title = title
    self.width = width
    self.height = height
    self.selectedMenu = None
    self.selectedMainMenu = None
    self.tk = Tk()
    #m_len = self.font.measure('m')
    #self.tk.resizable(False, False)
    self.canvas = Canvas(self.tk, height=self.height, width=self.width)
    #self.canvas2 = Canvas(self.tk, height=self.height/2, width=self.width/2)
    self.tk.title(title)
    self.canvas.pack()
    #self.canvas.pack(expand = YES, fill = BOTH)
    self.canvas.bind('<ButtonRelease-1>', self.createEdge)
    self.loadMapImage()
    # START loading background image and default endpoints
    """
    self.backgroundImage = ImageTk.PhotoImage(Image.open('./data/bantayan_island.png'))
    self.backgroundHeight = self.backgroundImage.height()
    self.backgroundWidth = self.backgroundImage.width()*1.5
    self.canvas.config(height=self.backgroundHeight, width=self.backgroundWidth)
    self.background = self.canvas.create_image(self.backgroundImage.width() / 2, self.backgroundImage.height() / 2, image=self.backgroundImage, anchor=CENTER)
    """
    self.redraw('./data/bantayan_island.json')

    self.summary = Listbox(self.canvas, font=('Monaco', 10))
    #bolded = font.Font(weight='bold') # will use the default font
    #self.label.config(font=bolded)
    self.summary.pack()
    self.canvas.create_window((self.backgroundImage.width()*1.25)+2, self.backgroundImage.height()/2, window=self.summary, height=self.backgroundImage.height(), width=self.backgroundImage.width()*0.5)
    # END loading background image and default endpoints
    self.buildMenu()
    self.visitedNodes = {}
    self.matrix = {}
    self.startNode = None
    self.goalNode = None
    self.ANNMaps = {
      'A': [3, 7, 9, 11, 15, 16, 17, 18, 19, 20, 21, 25, 26, 30, 31, 35],
      'B': [1, 2, 3, 4, 6, 10, 11, 15, 16, 17, 18, 19, 21, 25, 26, 30, 31, 32, 33, 34],
      'C': [2, 3, 4, 6, 10, 11, 16, 21, 26, 30, 32, 33, 34],
      'D': [1, 2, 3, 4, 6, 10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 32, 33, 34],
      'E': [1, 2, 3, 4, 5, 6, 11, 16, 17, 18, 19, 21, 26, 31, 32, 33, 34, 35],
      'F': [1, 2, 3, 4, 5, 6, 11, 16, 17, 18, 19, 21, 26, 31],
      'G': [2, 3, 4, 6, 10, 11, 16, 17, 18, 19, 20, 21, 25, 26, 30, 32, 33, 34],
      'H': [1, 5, 6, 10, 11, 15, 16, 17, 18, 19, 20, 21, 25, 26, 30, 31, 35],
      'I': [1, 2, 3, 4, 5, 8, 13, 18, 23, 28, 31, 32, 33, 34, 35],
      'J': [1, 2, 3, 4, 5, 8, 13, 18, 21, 23, 26, 28, 32, 33],
      'K': [1, 5, 6, 9, 11, 13, 16, 17, 21, 23, 26, 29, 31, 35],
      'L': [1, 6, 11, 16, 21, 26, 31, 32, 33, 34, 35],
      'M': [1, 5, 6, 7, 9, 10, 11, 13, 15, 16, 20, 21, 25, 26, 30, 31, 35],
      'N': [1, 5, 6, 7, 10, 11, 13, 15, 16, 19, 20, 21, 25, 26, 30, 31, 35],
      'O': [2, 3, 4, 6, 10, 11, 15, 16, 20, 21, 25, 26, 30, 32, 33, 34],
      'P': [2, 3, 4, 6, 10, 11, 15, 16, 17, 18, 19, 21, 26, 31],
      'Q': [2, 3, 4, 6, 10, 11, 15, 16, 20, 21, 23, 25, 26, 29, 30, 32, 33, 34, 35],
      'R': [2, 3, 4, 6, 10, 11, 15, 16, 17, 18, 19, 21, 23, 26, 29, 31, 35],
      'S': [2, 3, 4, 6, 10, 11, 17, 18, 19, 25, 26, 30, 32, 33, 34],
      'T': [1, 2, 3, 4, 5, 8, 13, 18, 23, 28, 33],
      'U': [1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26, 30, 32, 33, 34],
      'V': [1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 27, 29, 33],
      'W': [1, 5, 6, 10, 11, 15, 16, 20, 21, 23, 25, 26, 27, 29, 30, 31, 35],
      'X': [1, 5, 6, 10, 12, 14, 18, 22, 24, 26, 30, 31, 35],
      'Y': [1, 5, 6, 10, 12, 13, 14, 15, 20, 25, 26, 30, 32, 33, 34],
      'Z': [1, 2, 3, 4, 5, 10, 14, 18, 22, 26, 31, 32, 33, 34, 35],
    }
    self.ANNVowelConsonants = {
      'v': ['A', 'E', 'I', 'O', 'U'],
      'c': ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
    }
    self.loadJSONANN()

  def logSummary(self, log):
    self.summary.insert(END, log)

  def loadMapImage(self):
    # START loading background image and default endpoints
    self.backgroundImage = ImageTk.PhotoImage(Image.open('./data/bantayan_island.png'))
    self.backgroundHeight = self.backgroundImage.height()
    self.backgroundWidth = self.backgroundImage.width()*1.5
    self.canvas.config(height=self.backgroundHeight, width=self.backgroundWidth)
    self.background = self.canvas.create_image(self.backgroundImage.width() / 2, self.backgroundImage.height() / 2, image=self.backgroundImage, anchor=CENTER)

  def start(self):
    self.tk.mainloop()

  def reset(self, init=False):
    self.nodes = []
    self.edges = []
    self.selections = []
    if init:
      return
    self.clearBackgroundImage()

  def clearBackgroundImage(self):
    if self.backgroundImage is not None:
      self.canvas.config(height=self.backgroundHeight, width=self.backgroundWidth)
    self.backgroundImage = None

  def dumpGraphs(self):
    graphs = {'nodes': [], 'edges': []}
    for node in self.nodes:
      graphs['nodes'].append(node.toJSON())
    for edge in self.edges:
      graphs['edges'].append(edge.toJSON())
    return graphs

  def clearSelections(self):
    for index in self.selections:
      self.canvas.itemconfig('node{}'.format(index), fill='red')
    self.selections = []

  def drawEdge(self):
    node = self.nodes[self.selections[0]]
    node2 = self.nodes[self.selections[1]]
    self.clearSelections()
    center_x1 = node.point.x + ((node.point2.x - node.point.x) / 2)
    center_y1 = node.point.y + ((node.point2.y - node.point.y) / 2)
    center_x2 = node2.point.x + ((node2.point2.x - node2.point.x) / 2)
    center_y2 = node2.point.y + ((node2.point2.y - node2.point.y) / 2)
    # check if edge already exists
    p1 = Point(center_x1, center_y1)
    p2 = Point(center_x2, center_y2)
    edge_exists = False
    for edge in self.edges:
      # check if p1 is in edge.point
      p1p1 = (p1.x == edge.point.x and p1.y == edge.point.y) # check if p1 is in edge.point
      p1p2 = (p1.x == edge.point2.x and p1.y == edge.point2.y) # check if p1 is in edge.point2
      p2p1 = (p2.x == edge.point.x and p2.y == edge.point.y) # check if p2 is in edge.point
      p2p2 = (p2.x == edge.point2.x and p2.y == edge.point2.y) # check if p2 is in edge.point2
      if (p1p1 or p1p2) and (p2p1 or p2p2):
        edge_exists = True
        break
    if edge_exists:
      return # edge already exists
    edge = Edge(self.canvas, 'edge{}'.format(len(self.edges)), center_x1, center_y1, center_x2, center_y2, 'black', with_text=(self.selectedMainMenu != self.MENU_MAP_COLORING))
    self.edges.append(edge)

  def generateAdjacencyMatrix(self):
    self.matrix = {}
    nodes_len = len(self.nodes)
    # initialize matrix
    for row in range(nodes_len):
      self.matrix[row] = {}
      for col in range(nodes_len):
        # check if edge exists between the two nodes
        connected = self.isConnected(self.nodes[row], self.nodes[col], returnEdgeConnected=True)
        linked = True if row != col and connected['connected'] else False
        self.matrix[row][col] = {'linked': linked, 'node1': row, 'node2': col, 'edge': connected['edge']}
    return self.matrix

  # Search by Simple Queue, FIFO
  def UninformedBFS(self):
    traversed = self.traverse(self.SORT_QUEUE)
    self.animate(traversed['list'], traversed['path'])
    return traversed['path']

  # Search by Stack, LIFO
  def UninformedDFS(self):
    traversed = self.traverse(self.SORT_STACK)
    self.animate(traversed['list'], traversed['path'])
    return traversed['path']

  def UninformedUCS(self):
    traversed = self.traverse(self.SORT_PRIORITY_QUEUE)
    self.animate(traversed['list'], traversed['path'])
    return traversed['path']

  def InformedBFS(self):
    traversed = self.traverse(self.SORT_PRIORITY_QUEUE_BFS)
    self.animate(traversed['list'], traversed['path'])
    return traversed['path']

  def InformedASS(self):
    traversed = self.traverse(self.SORT_PRIORITY_QUEUE_ASS)
    self.animate(traversed['list'], traversed['path'])
    return traversed['path']

  def calculateHeuristic(self, node):
    start = self.nodes[node]
    goal = self.nodes[self.goalNode]
    #point = start.point.x
    # calculate using Manhattan Distance
    heuristic = abs(start.point.x - goal.point.x) + abs(start.point.y - goal.point.y)
    return heuristic

  def sortPriority(self, queue, method=None):
    if method is None:
      method = self.SORT_PRIORITY_QUEUE
    distances = {}
    for key in range(len(queue)):
      values = queue[key]
      # compute the distance for all the values
      distance = 0
      previousNode = None
      for value in values:
        if previousNode is None:
          previousNode = value
          continue
        # compute the distance
        if value in self.matrix and previousNode in self.matrix[value]:
          # use the distance
          edge = self.matrix[value][previousNode]['edge']
        else: # it the other way around
          edge = self.matrix[previousNode][value]['edge']
        distance += self.edges[edge].distance
        previousNode = value
      if method == self.SORT_PRIORITY_QUEUE_BFS:
        distances[key] = self.calculateHeuristic(previousNode)
      elif method == self.SORT_PRIORITY_QUEUE_ASS:
        distances[key] = distance + self.calculateHeuristic(previousNode)
      else: # self.SORT_PRIORITY_QUEUE
        distances[key] = distance
    # now based on distances, we're going to sort the values
    sortedDistances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    sortedQueue = []
    for key, value in sortedDistances.items():
      sortedQueue.append(queue[key])
    print('QUEUE: ', queue, ' SORTED DISTANCES: ', sortedDistances, ' SORTED: ', sortedQueue)
    # TODO: Priority queue for BFS
    # TODO: Priority queue for ASS
    return sortedQueue

  def sortQueue(self, method, fringe, queue):
    if method == self.SORT_QUEUE:
      queue += fringe
    elif method == self.SORT_STACK:
      fringe = sorted(fringe, reverse=True)
      queue += fringe
    elif method == self.SORT_PRIORITY_QUEUE:
      queue += fringe
      # do priority sort by distance
      # compute the distance of queue items
      queue = self.sortPriority(queue)
    elif method == self.SORT_PRIORITY_QUEUE_BFS:
      queue += fringe
      queue = self.sortPriority(queue, method)
    elif method == self.SORT_PRIORITY_QUEUE_ASS:
      queue += fringe
      queue = self.sortPriority(queue, method)
    return {'fringe': fringe, 'queue': queue}

  def animation(self, fringe, cols, previousVisitedNode):
    for key in fringe:
      col = key[-1]
      node1 = 'node{}'.format(cols[col]['node1'])
      node2 = 'node{}'.format(cols[col]['node2'])
      edge = 'edge{}'.format(cols[col]['edge'])
      if previousVisitedNode is None:
        previousVisitedNode = {'node1': node1, 'node2': node2, 'edge': edge}
      else: # revert to visited color
        self.canvas.itemconfig(previousVisitedNode['node1'], fill='blue')
        self.canvas.itemconfig(previousVisitedNode['node2'], fill='blue')
        self.canvas.itemconfig(previousVisitedNode['edge'], fill='blue')
        previousVisitedNode = {'node1': node1, 'node2': node2, 'edge': edge}
      self.visitedNodes[node1] = {'from': 'yellow', 'to': 'red'}
      self.visitedNodes[node2] = {'from': 'yellow', 'to': 'red'}
      self.visitedNodes[edge] = {'from': 'yellow', 'to': 'black'}
      self.canvas.itemconfig(node1, fill='yellow')
      self.canvas.itemconfig(node2, fill='yellow')
      self.canvas.itemconfig(edge, fill='yellow')
      self.tk.update()
    return previousVisitedNode

  def showAdjacencyMatrix(self, method=None, showDistance=False):
    nodeLen = len(self.nodes)
    rows = []
    headers = [' ']
    maxStr = 0
    for row in range(nodeLen):
      rowName = str(row+1)
      headers.append(rowName)
      cells = [rowName]
      if len(rowName) > maxStr:
          maxStr = len(rowName)
      for col in range(nodeLen):
        value = '0'
        if self.matrix[row][col]['linked']:
          if showDistance:
            value = str(self.edges[self.matrix[row][col]['edge']].distance)
          else:
            value = '1'
        cells.append(value)
        if len(value) > maxStr:
          maxStr = len(value)
      #rows.append(' | '.join(cells))
      rows.append(cells)
    self.summary.insert(END, '') # create extra space
    self.summary.insert(END, '=== ADJACENCY MATRIX ===')
    # pad values for headers
    for i in range(len(headers)):
      headers[i] = headers[i].ljust(maxStr)
    # pad values for cells
    for i in range(len(rows)):
      for j in range(len(rows[i])):
        rows[i][j] = rows[i][j].ljust(maxStr)
      rows[i] = (' | '.join(rows[i]))
    self.summary.insert(END, ' | '.join(headers))
    for row in rows:
      self.summary.insert(END, row)
    self.summary.insert(END, '') # create extra space
    # Create Heuristic Summary
    if method in [self.SORT_PRIORITY_QUEUE_BFS, self.SORT_PRIORITY_QUEUE_ASS]:
      headers = ['Node', 'Heuristic']
      rows = []
      for row in range(nodeLen):
        header = str(row+1)
        cell = str(self.nodes[row].heuristicValue)
        lenHeader = len(header)
        lenCell = len(cell)
        maxLen = lenCell if lenCell > lenHeader else lenHeader
        format = '{:<' + str(maxLen) + '}'
        rows.append(' | '.join([format.format(header), format.format(cell)]))
      self.summary.insert(END, '') # create extra space
      self.summary.insert(END, '=== HEURISTIC VALUES ===')
      self.summary.insert(END, ' | '.join(headers))
      for row in rows:
        self.summary.insert(END, row)
      self.summary.insert(END, '') # create extra space
    # Create Distance Summary
    if method == self.SORT_PRIORITY_QUEUE_ASS: # create distance
      headers = ['Nodes', 'Distance']
      rows = []
      for row in range(nodeLen):
        for col in range(nodeLen):
          if (row == col):
            continue
          header = '{} -> {}'.format(row+1, col+1)
          edge = self.matrix[row][col]['edge']
          if edge is None:
            continue
          distance = self.edges[edge].distance
          cell = str(distance)
          lenHeader = len(header)
          lenCell = len(cell)
          maxLen = lenCell if lenCell > lenHeader else lenHeader
          format = '{:<' + str(maxLen) + '}'
          rows.append(' | '.join([format.format(header), format.format(cell)]))
      self.summary.insert(END, '') # create extra space
      self.summary.insert(END, '=== DISTANCE VALUES ===')
      self.summary.insert(END, ' | '.join(headers))
      for row in rows:
        self.summary.insert(END, row)
      self.summary.insert(END, '') # create extra space

  def genericTraversal(self, method=None):
    # add/remove heuristic values
    for index in range(len(self.nodes)):
      # clear the heuristic first
      if self.nodes[index].heuristic is not None:
        self.canvas.delete(self.nodes[index].heuristic)
      self.nodes[index].heuristic = None
      node = self.nodes[index]
      if method in [self.SORT_PRIORITY_QUEUE_BFS, self.SORT_PRIORITY_QUEUE_ASS]:
        heuristic = self.calculateHeuristic(index)
        self.nodes[index].heuristicValue = heuristic
        self.nodes[index].heuristic = self.canvas.create_text(node.point.x, node.point.y+30, text='h={}'.format(heuristic), font=('Monaco', 10, 'italic'), fill='blue')
    self.showAdjacencyMatrix(method)
    visited = []
    queue = [[self.startNode]]
    traversed = []
    stack = []
    #matched = None
    # Start traversing
    while queue:
      if method == self.SORT_STACK:
        path = queue.pop() # get the last path from the queue and remove it
      else: # Queue
        path = queue.pop(0) # get the first path from the queue and remove it
      stack.append(path)
      row = path[-1] # get the last row from the path
      if row in visited: # check if path has already been visited
        continue # skip if path already visited
      cols = self.matrix[row] # get all the columns
      # open the fringe
      fringe = []
      for col in cols:
        if not cols[col]['linked']: # skip if matrix is not 1
          continue
        # do the dance
        rows = list(path)
        rows.append(col)
        fringe.append(rows)
      # sort the fringe
      if method in [self.SORT_PRIORITY_QUEUE, self.SORT_PRIORITY_QUEUE_BFS, self.SORT_PRIORITY_QUEUE_ASS]:
        sortedQueue = self.sortQueue(method, fringe, queue)
      else:
        sortedQueue = self.sortQueue(self.SORT_QUEUE, fringe, queue)
      queue = sortedQueue['queue']
      traversed += sortedQueue['fringe']
      visited.append(row) # set this row to visited
    return stack + queue

  def traverseBFS(self):
    nodes = self.genericTraversal(self.SORT_QUEUE)
    path = None
    traversed = []
    for node in nodes:
      if len(node) < 2:
        continue
      traversed.append(node)
      if node[0] == self.startNode and node[-1] == self.goalNode:
        path = node
        break
    return {'path': path, 'list': traversed}

  def traverseDFS(self):
    nodes = self.genericTraversal(self.SORT_STACK)
    path = None
    traversed = []
    for node in nodes:
      if len(node) < 2:
        continue
      traversed.append(node)
      if node[0] == self.startNode and node[-1] == self.goalNode:
        path = node
        break
    return {'path': path, 'list': traversed}

  def traverseUCS(self):
    nodes = self.genericTraversal(self.SORT_PRIORITY_QUEUE)
    path = None
    traversed = []
    for node in nodes:
      if len(node) < 2:
        continue
      traversed.append(node)
      if node[0] == self.startNode and node[-1] == self.goalNode:
        path = node
        break
    return {'path': path, 'list': traversed}

  def traverseGBFS(self):
    nodes = self.genericTraversal(self.SORT_PRIORITY_QUEUE_BFS)
    path = None
    traversed = []
    for node in nodes:
      if len(node) < 2:
        continue
      traversed.append(node)
      if node[0] == self.startNode and node[-1] == self.goalNode:
        path = node
        break
    return {'path': path, 'list': traversed}

  def traverseASS(self):
    nodes = self.genericTraversal(self.SORT_PRIORITY_QUEUE_ASS)
    path = None
    traversed = []
    for node in nodes:
      if len(node) < 2:
        continue
      traversed.append(node)
      if node[0] == self.startNode and node[-1] == self.goalNode:
        path = node
        break
    return {'path': path, 'list': traversed}

  def traverse(self, method=None):
    if self.startNode == self.goalNode:
      return []
    if method is None:
      method = self.SORT_QUEUE
    if method == self.SORT_STACK:
      return self.traverseDFS()
    elif method == self.SORT_PRIORITY_QUEUE:
      return self.traverseUCS()
    elif method == self.SORT_PRIORITY_QUEUE_BFS:
      return self.traverseGBFS()
    elif method == self.SORT_PRIORITY_QUEUE_ASS:
      return self.traverseASS()
    else:
      return self.traverseBFS()

  def animate(self, nodes, path):
    visited = None
    for node in nodes:
      if len(node) < 2:
        continue
      row = node[-2]
      col = node[-1]
      # do the dance
      node1 = 'node{}'.format(self.matrix[row][col]['node1'])
      node2 = 'node{}'.format(self.matrix[row][col]['node2'])
      edge = 'edge{}'.format(self.matrix[row][col]['edge'])
      self.summary.insert(END, 'Visiting {} -> {}'.format(self.matrix[row][col]['node1']+1, self.matrix[row][col]['node2']+1))
      if visited is None:
        visited = {'node1': node1, 'node2': node2, 'edge': edge}
      else: # revert to visited color
        self.canvas.itemconfig(visited['node1'], fill='blue')
        self.canvas.itemconfig(visited['node2'], fill='blue')
        self.canvas.itemconfig(visited['edge'], fill='blue')
        visited = {'node1': node1, 'node2': node2, 'edge': edge}
      self.visitedNodes[node1] = {'from': 'yellow', 'to': 'red'}
      self.visitedNodes[node2] = {'from': 'yellow', 'to': 'red'}
      self.visitedNodes[edge] = {'from': 'yellow', 'to': 'black'}
      self.canvas.itemconfig(node1, fill='yellow')
      self.canvas.itemconfig(node2, fill='yellow')
      self.canvas.itemconfig(edge, fill='yellow')
      self.tk.update()
      sleep(2) # delay
      if node == path:
        break
    self.summary.insert(END, 'PATH: {}'.format(' -> '.join(str(x+1) for x in path)))
    pathCost = 0
    prev = None
    for x in path:
      if prev is None:
        prev = x
        continue
      edge = self.matrix[prev][x]['edge']
      prev = x
      if edge is None:
        continue
      pathCost += self.edges[edge].distance
    self.summary.insert(END, 'PATH COST: {}'.format(pathCost))

  def isConnected(self, node, node2, returnIndex=False, returnEdgeConnected=False):
    connected = False
    edgeConnected = None
    for index, edge in enumerate(self.edges):
      x1 = edge.point.x
      y1 = edge.point.y
      x2 = edge.point2.x
      y2 = edge.point2.y
      # check if left side of the edge is within the first node
      left_node = True if node.point.x < x1 and x1 < node.point2.x and node.point.y < y1 and y1 < node.point2.y else False
      # check if right side of the edge is within the first node
      right_node = True if node.point.x < x2 and x2 < node.point2.x and node.point.y < y2 and y2 < node.point2.y else False
      # check if left side of the edge is within the second node
      left_node2 = True if node2.point.x < x1 and x1 < node2.point2.x and node2.point.y < y1 and y1 < node2.point2.y else False
      # check if right side of the edge is within the second node
      right_node2 = True if node2.point.x < x2 and x2 < node2.point2.x and node2.point.y < y2 and y2 < node2.point2.y else False
      if (left_node or right_node) and (left_node2 or right_node2):
        connected = True if not returnIndex else index
        edgeConnected = index
        break
    return {'connected': connected, 'edge': edgeConnected} if returnEdgeConnected else connected

  def searchGeneric(self, method):
    self.startNode = self.selections[0]
    self.goalNode = self.selections[1]
    self.summary.insert(END, 'Start: {} Goal: {}'.format(self.startNode+1, self.goalNode+1))
    self.generateAdjacencyMatrix()
    print('Matrix: ', self.matrix)
    print('Searching from {} to {}'.format(self.startNode+1, self.goalNode+1))
    self.visitedNodes = {}
    search = getattr(self, method)()
    for index in self.visitedNodes:
      self.canvas.itemconfig(index, fill=self.visitedNodes[index]['to'])
    #print('SEARCH RESULTS: ', search)
    # change edge colors
    if search is not False and search is not None:
      node = None
      for x in search:
        if node is None:
          node = self.nodes[x]
          continue
        index = self.isConnected(node, self.nodes[x], True)
        self.canvas.itemconfig('edge{}'.format(index), fill='green')
        node = self.nodes[x]
    self.clearSelections()

  def searchBreadthFirst(self):
    self.summary.delete(0, END)
    self.summary.insert(END, '*** BREADTH FIRST SEARCH ***')
    return self.searchGeneric('UninformedBFS')

  def searchDepthFirst(self):
    self.summary.delete(0, END)
    self.summary.insert(END, '*** DEPTH FIRST SEARCH ***')
    return self.searchGeneric('UninformedDFS')

  def searchUniformCost(self):
    self.summary.delete(0, END)
    self.summary.insert(END, '*** UNIFORMED COST SEARCH ***')
    return self.searchGeneric('UninformedUCS')

  def searchBestFirst(self):
    self.summary.delete(0, END)
    self.summary.insert(END, '*** BEST FIRST SEARCH ***')
    # NOTE: heuristic never overestimate the cost from node to goal
    # similar to Uniform Cost Search but only deals with heuristic value
    # heuristic value = abs (current_node.x – goal.x) + abs (current_node.y – goal.y) - Manhattan Distance
    return self.searchGeneric('InformedBFS')

  def searchAStar(self):
    self.summary.delete(0, END)
    self.summary.insert(END, '*** A STAR SEARCH ***')
    # NOTE: heuristic never overestimate the cost from node to goal
    # similar to Uniform Cost Search + heuristic value
    # A score = sum of cost of path from start node + heuristic value of node
    return self.searchGeneric('InformedASS')

  def doFileNew(self):
    self.selectedMenu = self.MENU_FILE_NEW = 1
    #self.canvas.delete('all')
    for index, node in enumerate(self.nodes):
      self.canvas.delete('node{}'.format(index))
      if node.heuristic is not None:
        self.canvas.delete(node.heuristic)
      if node.graph is not None:
        self.canvas.delete(node.graph)
      if node.text is not None:
        self.canvas.delete(node.text)
    for index, edge in enumerate(self.edges):
      if edge.graph is not None:
        self.canvas.delete(edge.graph)
      if edge.text is not None:
        self.canvas.delete(edge.text)
      self.canvas.delete('edge{}'.format(index))
    self.reset()

  def doFileOpen(self):
    filename = filedialog.askopenfilename(initialdir='/', title='Select json file', filetypes=(('json files', '*.json'),('all files', '*.*')))
    self.doFileNew()
    self.redraw(filename)
    self.selectedMenu = self.MENU_FILE_OPEN = 2

  def redraw(self, filename):
    contents = open(filename).read(999999999)
    objects = json.loads(contents)
    # draw canvas
    if objects['nodes']:
      for index, node in enumerate(objects['nodes']):
        node = Node(self.canvas, 'node{}'.format(index) , node['point']['x'], node['point']['y'], index + 1, 'red', with_text=(self.selectedMainMenu != self.MENU_MAP_COLORING))
        self.nodes.append(node)
    if objects['edges']:
      for index, edge in enumerate(objects['edges']):
        edge = Edge(self.canvas, 'edge{}'.format(len(self.edges)), edge['point']['x'], edge['point']['y'], edge['point2']['x'], edge['point2']['y'], 'black', with_text=(self.selectedMainMenu != self.MENU_MAP_COLORING))
        self.edges.append(edge)

  def doFileSave(self):
    self.selectedMenu = self.MENU_FILE_SAVE = 4
    filename = filedialog.asksaveasfilename(initialdir='/', title='Select json file', filetypes=(('json files', '*.json'),('all files', '*.*')), initialfile='machine_problem_1', defaultextension='.json')
    if len(filename) > 0:
      contents = json.dumps(self.dumpGraphs(), sort_keys=True, indent=4)
      f = open(filename, 'w')
      f.write(contents)
      f.close()

  def doFileExit(self):
    self.selectedMenu = self.MENU_FILE_EXIT
    self.canvas.quit()
    self.tk.destroy()

  def doGraphNode(self):
    self.selectedMenu = self.MENU_GRAPH_NODE

  def doGraphEdge(self):
    self.selectedMenu = self.MENU_GRAPH_EDGE
    self.clearSelections()

  def doSearchBreadthFirst(self):
    self.selectedMenu = self.MENU_SEARCH_BREADTH_FIRST
    self.clearSelections()

  def doSearchDepthFirst(self):
    self.selectedMenu = self.MENU_SEARCH_DEPTH_FIRST
    self.clearSelections()

  def doSearchUniformCost(self):
    self.selectedMenu = self.MENU_SEARCH_UNIFORM_COST
    self.clearSelections()

  def doSearchBestFirst(self):
    self.selectedMenu = self.MENU_SEARCH_BEST_FIRST
    self.clearSelections()

  def doSearchAStar(self):
    self.selectedMenu = self.MENU_SEARCH_A_STAR
    self.clearSelections()

  def doMapSelect(self):
    self.clearSelections()
    self.selectedMainMenu = self.MENU_MAP_COLORING

  vertices = 0
  graph = []
  colors = ["maroon", "cyan", "gray", "orange", "brown", "purple"]
  printed = [] # storage of summary report
  blinked = []

  def hasConstraint(self, vertex, colors, color):
    for i in range(self.vertices):
      if self.graph[vertex][i] == 0:
        continue # skip, only graph with linked
      if colors[i] == color: # constraint found, color already assigned to one of its neighbors
        return True
    return False

  def backtrackMapColoring(self, m, colors, vertex):
    # if assignment is complete then return assignment
    if vertex == self.vertices:
      return True
    # var <- SELECT-UNASSIGNED-VARIABLE(VARIABLES[csp], assignment, csp)
    # for each value in ORDER-DOMAIN-VALUES(var, assignment, csp) do
    self.canvas.itemconfig('node{}'.format(vertex), fill="yellow")
    self.tk.update()
    for color in range(m):
      sleep(1) # delay
      # check if value is does not have the same color
      if self.hasConstraint(vertex, colors, color):
        continue
      # add { var = value } to assignment
      colors[vertex] = color
      # update node coloring
      for index in range(len(colors)):
        if colors[index] is None:
          continue
        text = 'node{} = {}'.format(index+1, self.colors[colors[index]])
        if text not in self.printed:
          self.printed.append(text)
          self.summary.insert(END, text)
        self.canvas.itemconfig('node{}'.format(index), fill=self.colors[colors[index]])
        self.tk.update()
      # result <- backtrackMapColoring(assignment, csp)
      result = self.backtrackMapColoring(m, colors, vertex + 1)
      # if result != failure then return result
      if result == True:
        return True
      # remove { var = value } from assignment
      colors[vertex] = None
    return False

  def doMapColoring(self, m):
    self.selectedMainMenu = None
    self.summary.delete(0, END)
    self.summary.insert(END, '*** MAP COLORING ***')
    # perform coloring
    self.generateAdjacencyMatrix()
    #print('Matrix: ', self.matrix)
    self.showAdjacencyMatrix()
    nodeLen = len(self.nodes)
    graph = []
    for row in range(nodeLen):
      item = []
      for col in range(nodeLen):
        item.append(1 if self.matrix[row][col]['linked'] else 0)
      graph.append(item)
    self.vertices = len(graph)
    self.graph = graph
    self.printed = []
    self.blinked = []
    colors = [None] * self.vertices
    textColors = []
    for color in range(m):
      textColors.append("{}".format(self.colors[color]))
    self.summary.insert(END, "Assigning {} color(s): [{}]".format(m, ", ".join(textColors)))
    self.summary.insert(END, "")
    result = self.backtrackMapColoring(m, colors, 0)
    self.summary.insert(END, "")
    if result is False:
      self.summary.insert(END, "No solution found!")
      return False
    textColors = []
    for color in set(colors):
      textColors.append("{}".format(self.colors[color]))
    self.summary.insert(END, "Solution: [{}]".format(", ".join(textColors)))
    return True

  def permutation(self, elements):
    if len(elements) == 0:
      return elements
    if len(elements) == 1:
      return [elements]
    dataset = []
    for i in range(len(elements)):
      #element = elements[i]
      newElements = elements[:i] + elements[i+1:]
      for perm in self.permutation(newElements):
        dataset.append([elements[i]] + perm)
    return dataset

  def travellingSalesmanProblem(self, graph, vertex):
    vertices = []
    for i in range(self.vertices):
      if i != vertex:
        vertices.append(i)
    minPath = None
    minPaths = None
    permVertices = self.permutation(vertices)
    for i in permVertices:
      sleep(2)
      print(i)
      weightPath = 0
      # compute current path weight
      k = vertex
      paths = []
      #print('COLORING')
      colorChanges = []
      for j in i:
        if graph[k][j] == 0:
          weightPath = 0
          paths = []
          break
        paths.append((k, j, graph[k][j]))
        colorChanges.append((k, j))
        self.summary.insert(END, "node {} -> node {} = distance {}".format(k+1, j+1, graph[k][j]))
        self.canvas.itemconfig('node{}'.format(k), fill="yellow")
        self.canvas.itemconfig('edge{}'.format(self.matrix[k][j]['edge']), fill="yellow")
        self.tk.update()
        sleep(0.3) # TODO: to speed up, comment this out
        weightPath += graph[k][j]
        k = j
      j = vertex
      weightPath += graph[k][vertex]
      paths.append((k, j, graph[k][vertex]))
      colorChanges.append((k, j))
      self.summary.insert(END, "node {} -> node {} = distance {}".format(k+1, j+1, graph[k][j]))
      self.canvas.itemconfig('node{}'.format(k), fill="yellow")
      self.canvas.itemconfig('edge{}'.format(self.matrix[k][j]['edge']), fill="yellow")
      self.summary.insert(END, "PATH COST = {}".format(weightPath))
      self.tk.update()
      # update minimum
      if minPath is None or weightPath > minPath:
        minPath = weightPath
        minPaths = paths
      #print('REVERTING')
      sleep(2)
      for item in colorChanges:
        k = item[0]
        j = item[1]
        self.canvas.itemconfig('node{}'.format(k), fill="red")
        self.canvas.itemconfig('edge{}'.format(self.matrix[k][j]['edge']), fill="black")
        self.tk.update()
    print("minPaths")
    print(minPaths)
    if len(minPaths) > 0:
      for path in minPaths:
        row = path[0]
        col = path[1]
        self.canvas.itemconfig('node{}'.format(row), fill="green")
        self.canvas.itemconfig('edge{}'.format(self.matrix[row][col]['edge']), fill="green")
        self.tk.update()
      row = 0
      #if self.matrix[row][col]['linked']:
      #  self.canvas.itemconfig('node{}'.format(col), fill="green")
      #  self.canvas.itemconfig('edge{}'.format(self.matrix[row][col]['edge']), fill="green")
      #self.tk.update()
    return minPath

  def doTravelingSalesmanProblem(self):
    self.clearSelections()
    self.selectedMainMenu = self.MENU_TRAVELING_SALESMAN_PROBLEM
    self.summary.delete(0, END)
    self.summary.insert(END, "*** TRAVELING SALESMAN PROBLEM ***")
    self.summary.insert(END, "")
    self.generateAdjacencyMatrix()
    self.showAdjacencyMatrix(showDistance=True)
    # compose the graph data
    nodeLen = len(self.nodes)
    graph = []
    for row in range(nodeLen):
      item = []
      for col in range(nodeLen):
        distance = self.edges[self.matrix[row][col]['edge']].distance if self.matrix[row][col]['linked'] else 0
        #self.summary.insert(END, "{} -> {} = {}".format(row+1, col+1, distance))
        item.append(distance)
      graph.append(item)
    print(graph)
    self.vertices = len(graph)
    min_path = self.travellingSalesmanProblem(graph, 0)
    print(min_path)

  def loadMap(self):
    self.loadMapImage()

  def loadJSONUniformed(self):
    self.doFileNew()
    self.loadMap()
    self.redraw('./data/bantayan_island.json')

  def loadJSONTSP(self):
    self.doFileNew()
    self.loadMap()
    self.redraw('./data/traveling_salesman.json')

  def loadJSONMapColoring(self):
    self.doFileNew()
    self.loadMap()
    self.redraw('./data/map_coloring.json')

  def loadJSONANN(self):
    self.doFileNew()
    self.selectedMainMenu = self.MENU_ANN
    self.alphaButtons = []
    #self.loadMap()
    # create an option menu
    dimension = 24
    x = 5
    y = 5
    alphabets = list(self.ANNMaps.keys())
    for i in range(0, 26):
      x2 = x + dimension
      y2 = y + dimension
      self.alphaButtons.append({'x': x, 'y': y, 'x2': x2, 'y2': y2, 'index': i, 'alpha': alphabets[i]})
      self.canvas.create_rectangle(x, y, x2, y2, outline="black", fill="white", tags='box_{}'.format(i))
      self.canvas.create_text((x+x2)/2, (y+y2)/2, text=alphabets[i], font=('Monaco', 14, 'italic'), fill='black', tags='alpha_{}'.format(i))
      x = x2
      #y = y2
    self.cmaps = []
    self.selectedCmaps = [True] # index 0 is biased
    dimension = 70
    x = 30
    y = 70
    for i in range(1, 36):
      x2 = x + dimension
      y2 = y + dimension
      self.selectedCmaps.append(False)
      self.cmaps.append({'x': x, 'y': y, 'x2': x2, 'y2': y2, 'index': i})
      self.canvas.create_rectangle(x, y, x2, y2, outline="black", fill="white", tags='cmap_{}'.format(i))
      self.canvas.create_text(x+8, y+8, text=i, font=('Monaco', 10, 'italic'), fill='red', tags='calpha_{}'.format(i))
      x = x2
      if i % 5 == 0:
        x = 30
        y += dimension
    self.canvas.create_text((x2+(dimension*2)), y/2, text='', font=('Monaco', 30, 'italic'), fill='blue', tags='classify')
    self.canvas.pack(fill=BOTH, expand=1)
    self.resetANNOutput()
    #self.redraw('./data/ann.json')

  def resetANNOutput(self):
    self.canvas.itemconfig('classify', text='...')

  def trainANN(self):
    ann = Perceptron(self)
    ann.train()

  def classifyANN(self):
    ann = Perceptron(self)
    isVowel = ann.classify()
    self.canvas.itemconfig('classify', text=('Vowel' if isVowel else 'Consonant'))

  def trainAdaline(self):
    ann = Adaline(self)
    ann.train()

  def classifyAdaline(self):
    ann = Adaline(self)
    isVowel = ann.classify()
    self.canvas.itemconfig('classify', text=('Vowel' if isVowel else 'Consonant'))

  def redrawANNMap(self, char):
    if char not in self.ANNMaps:
      return
    for index in range(1, len(self.selectedCmaps)):
      color = 'black' if index in self.ANNMaps[char] else 'white'
      textcolor = 'green' if index in self.ANNMaps[char] else 'red'
      self.canvas.itemconfig('cmap_{}'.format(index), fill=color)
      self.canvas.itemconfig('calpha_{}'.format(index), fill=textcolor)
      self.selectedCmaps[index] = True if index in self.ANNMaps[char] else False

  def buildMenu(self):
    self.menubar = Menu(self.tk)
    loadmenu = Menu(self.menubar, tearoff=0)
    #loadmenu.add_command(label='Map', command=self.loadMap)
    loadmenu.add_command(label='Uninformed/Informed', command=self.loadJSONUniformed)
    loadmenu.add_command(label='Traveling Salesman Problem', command=self.loadJSONTSP)
    #loadmenu.add_command(label='Map Coloring', command=self.loadJSONMapColoring)
    #loadmenu.add_command(label='ANN', command=self.loadJSONANN)
    # File menu
    filemenu = Menu(self.menubar, tearoff=0)
    filemenu.add_command(label='New', command=self.doFileNew)
    filemenu.add_cascade(label='Load', menu=loadmenu)
    filemenu.add_command(label='Open', command=self.doFileOpen)
    filemenu.add_command(label='Save', command=self.doFileSave)
    filemenu.add_command(label='Clear Background', command=self.clearBackgroundImage)
    filemenu.add_separator()
    filemenu.add_command(label='Exit', command=self.doFileExit)
    self.menubar.add_cascade(label='File', menu=filemenu)
    # Graph menu
    graphmenu = Menu(self.menubar, tearoff=0)
    graphmenu.add_command(label='Draw Node', command=self.doGraphNode)
    graphmenu.add_command(label='Draw Edge', command=self.doGraphEdge)
    self.menubar.add_cascade(label='Graph', menu=graphmenu)
    # Search menu
    # Uninformed Search
    uninformed_search = Menu(self.menubar)
    uninformed_search.add_command(label='Breadth First Search', command=self.doSearchBreadthFirst)
    uninformed_search.add_command(label='Depth First Search', command=self.doSearchDepthFirst)
    uninformed_search.add_command(label='Uniform Cost Search', command=self.doSearchUniformCost)
    # Informed Search
    informed_search = Menu(self.menubar)
    informed_search.add_command(label='Best First Search', command=self.doSearchBestFirst)
    informed_search.add_command(label='A Star Search', command=self.doSearchAStar)
    # Combine in one menu
    searchmenu = Menu(self.menubar, tearoff=0)
    searchmenu.add_cascade(label='Uninformed', menu=uninformed_search)
    searchmenu.add_cascade(label='Informed', menu=informed_search)
    searchmenu.add_command(label='Traveling Salesman Problem', command=self.doTravelingSalesmanProblem)
    self.menubar.add_cascade(label='Search', menu=searchmenu)
    # map coloring
    mapmenu = Menu(self.menubar)
    mapmenu.add_command(label='Load', command=self.loadJSONMapColoring)
    mapmenu.add_command(label='Select', command=self.doMapSelect)
    #mapmenu.add_command(label='Run', command=self.doMapColoring)
    maprunmenu = Menu(self.menubar, tearoff=0)
    maprunmenu.add_command(label='1 color', command=lambda: self.doMapColoring(1))
    maprunmenu.add_command(label='2 colors', command=lambda: self.doMapColoring(2))
    maprunmenu.add_command(label='3 colors', command=lambda: self.doMapColoring(3))
    maprunmenu.add_command(label='4 colors', command=lambda: self.doMapColoring(4))
    maprunmenu.add_command(label='5 colors', command=lambda: self.doMapColoring(5))
    maprunmenu.add_command(label='6 colors', command=lambda: self.doMapColoring(6))
    mapmenu.add_cascade(label='Run', menu=maprunmenu)
    self.menubar.add_cascade(label='Map Coloring', menu=mapmenu)
    annmenu = Menu(self.menubar)
    annmenu.add_command(label='Load', command=self.loadJSONANN)
    annmenu.add_command(label='Reset Output', command=self.resetANNOutput)
    annmenu.add_command(label='Train', command=self.trainANN)
    annmenu.add_command(label='Classify', command=self.classifyANN)
    self.menubar.add_cascade(label='Perceptron', menu=annmenu)
    adamenu = Menu(self.menubar)
    adamenu.add_command(label='Load', command=self.loadJSONANN)
    adamenu.add_command(label='Reset Output', command=self.resetANNOutput)
    adamenu.add_command(label='Train', command=self.trainAdaline)
    adamenu.add_command(label='Classify', command=self.classifyAdaline)
    self.menubar.add_cascade(label='Adaline', menu=adamenu)
    self.tk.config(menu=self.menubar)

  def createEdge(self, event):
    x, y = event.x, event.y
    if self.selectedMainMenu == self.MENU_ANN:
      for coords in self.alphaButtons:
        if coords['x'] < x and x < coords['x2'] and coords['y'] < y and y < coords['y2']:
          self.redrawANNMap(coords['alpha'])
          return
      for coords in self.cmaps:
        if coords['x'] < x and x < coords['x2'] and coords['y'] < y and y < coords['y2']:
          color = 'black' if self.canvas.itemcget('cmap_{}'.format(coords['index']), 'fill') == 'white' else 'white'
          textcolor = 'green' if self.canvas.itemcget('calpha_{}'.format(coords['index']), 'fill') == 'red' else 'red'
          self.canvas.itemconfig('cmap_{}'.format(coords['index']), fill=color)
          self.canvas.itemconfig('calpha_{}'.format(coords['index']), fill=textcolor)
          self.selectedCmaps[coords['index']] = True if color == 'black' else False
          #print(self.selectedCmaps)
          return
      return
    # check for out of bounds
    size = self.DEFAULT_SIZE / 2
    x = self.width - size if x > (self.width - size) else x
    x = size + 5 if x < size + 5 else x
    y = self.height - size if y > (self.height - size) else y
    y = size + 5 if y < size + 5 else y
    if self.selectedMenu == self.MENU_GRAPH_NODE:
      node = Node(self.canvas, 'node{}'.format(len(self.nodes)) , x-size, y-size, len(self.nodes) + 1, 'red', with_text=(self.selectedMainMenu != self.MENU_MAP_COLORING))
      self.nodes.append(node)
    elif self.selectedMenu == self.MENU_GRAPH_EDGE:
      # locate if clicked boundary within a node and turn it to green, otherwise do nothing
      for index, node in enumerate(self.nodes):
        if node.point.x < x and x < node.point2.x and node.point.y < y and y < node.point2.y:
          self.selections.append(index)
          self.canvas.itemconfig('node{}'.format(index), fill='green')
          if len(self.selections) >= 2:
            self.drawEdge()
    elif self.selectedMenu == self.MENU_SEARCH_BREADTH_FIRST \
      or self.selectedMenu == self.MENU_SEARCH_DEPTH_FIRST \
      or self.selectedMenu == self.MENU_SEARCH_UNIFORM_COST \
      or self.selectedMenu == self.MENU_SEARCH_BEST_FIRST \
      or self.selectedMenu == self.MENU_SEARCH_A_STAR:
      # locate if clicked boundary within a node and turn it to green, otherwise do nothing
      for index, node in enumerate(self.nodes):
        if node.point.x < x and x < node.point2.x and node.point.y < y and y < node.point2.y:
          self.selections.append(index)
          self.canvas.itemconfig('node{}'.format(index), fill='green')
          if len(self.selections) >= 2:
            if self.selectedMenu == self.MENU_SEARCH_BREADTH_FIRST:
              self.searchBreadthFirst()
            elif self.selectedMenu == self.MENU_SEARCH_DEPTH_FIRST:
              self.searchDepthFirst()
            elif self.selectedMenu == self.MENU_SEARCH_UNIFORM_COST:
              self.searchUniformCost()
            elif self.selectedMenu == self.MENU_SEARCH_BEST_FIRST:
              self.searchBestFirst()
            elif self.selectedMenu == self.MENU_SEARCH_A_STAR:
              self.searchAStar()
          else:
            # change all edge colors
            for index, in enumerate(self.edges):
              self.canvas.itemconfig('edge{}'.format(index), fill='black')

Form().start()
