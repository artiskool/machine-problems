from cmath import sqrt
from time import sleep
import json
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


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
  def __init__(self, canvas, tag, x1, y1, x2, y2, color, width=3):
    self.point = Point(x1, y1)
    self.point2 = Point(x2, y2)
    self.distance = round(sqrt((x2 - x1)**2 + (y2 - y1)**2).real)
    self.graph = canvas.create_line(self.point.x, self.point.y, self.point2.x, self.point2.y, fill=color, width=width, tags=tag)
    px = (self.point.x + self.point2.x) / 2
    py = (self.point.y + self.point2.y) / 2
    self.text = canvas.create_text(px, py, text=self.distance, font='Verdana 14 italic', fill='red') # create label distance
    self.canvas = canvas

  def toJSON(self):
    return {'point': self.point.toJSON(), 'point2': self.point2.toJSON()}


class Node(Machine):
  def __init__(self, canvas, tag, x, y, label, color='red'):
    self.point = Point(x, y)
    self.point2 = Point(x + self.DEFAULT_SIZE, y + self.DEFAULT_SIZE)
    self.box = Box(self.point.x, self.point.y, self.point.x + self.DEFAULT_SIZE, self.point.y + self.DEFAULT_SIZE)
    self.graph = canvas.create_oval(self.box.left, self.box.top, self.box.right, self.box.bottom, fill=color, tags=tag)
    self.text = canvas.create_text(self.point.x, self.point.y, text=label, font='Verdana 10 italic')
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

  SORT_QUEUE = 1
  SORT_STACK = 2
  SORT_PRIORITY_QUEUE = 3

  def __init__(self, title='Machine Problem #1', width=1024, height=768):
    self.reset()
    self.width = width
    self.height = height
    self.selectedMenu = None
    self.tk = Tk()
    self.canvas = Canvas(self.tk, height=self.height, width=self.width)
    #self.canvas.grid(row=0, column=0)
    self.tk.title(title)
    self.canvas.pack()
    self.canvas.bind('<ButtonRelease-1>', self.createEdge)
    # START loading background image and default endpoints
    self.backgroundImage = ImageTk.PhotoImage(Image.open('./data/bantayan-island.png'))
    self.canvas.config(height=self.backgroundImage.height(), width=self.backgroundImage.width())
    self.background = self.canvas.create_image(self.backgroundImage.width() / 2, self.backgroundImage.height() / 2, image=self.backgroundImage, anchor=CENTER)
    self.redraw('./data/bantayan-island.json')
    # END loading background image and default endpoints
    self.buildMenu()
    self.visitedNodes = {}
    self.matrix = {}

  def start(self):
    self.tk.mainloop()

  def reset(self):
    self.nodes = []
    self.edges = []
    self.selections = []

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
    center_x1 = node.point.x + ((node.point2.x - node.point.x) / 2);
    center_y1 = node.point.y + ((node.point2.y - node.point.y) / 2);
    center_x2 = node2.point.x + ((node2.point2.x - node2.point.x) / 2);
    center_y2 = node2.point.y + ((node2.point2.y - node2.point.y) / 2);
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
    edge = Edge(self.canvas, 'edge{}'.format(len(self.edges)), center_x1, center_y1, center_x2, center_y2, 'black')
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
  def UninformedBFS(self, start, goal):
    traversed = self.traverse(start, goal, self.SORT_QUEUE)
    self.animate(traversed['list'], traversed['path'])
    return traversed['path']

  # Search by Stack, LIFO
  def UninformedDFS(self, start, goal):
    traversed = self.traverse(start, goal, self.SORT_STACK)
    self.animate(traversed['list'], traversed['path'])
    return traversed['path']

  def InformedBFS(self, start, goal):
    pass

  def InformedASS(self, start, goal):
    pass

  def sortPriority(self, queue):
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
      distances[key] = distance
    # now based on distances, we're going to sort the values
    sortedDistances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    sortedQueue = []
    for key, value in sortedDistances.items():
      sortedQueue.append(queue[key])
    print('QUEUE: ', queue, ' SORTED DISTANCES: ', sortedDistances, ' SORTED: ', sortedQueue)
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
    return {'fringe': fringe, 'queue': queue}

  def animate(self, fringe, cols, previousVisitedNode):
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

  def genericTraversal(self, start, method=None):
    visited = []
    queue = [[start]]
    #traversed = [[start]]
    traversed = []
    stack = []
    matched = None
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
      if method == self.SORT_PRIORITY_QUEUE:
        sortedQueue = self.sortQueue(method, fringe, queue)
      else:
        sortedQueue = self.sortQueue(self.SORT_QUEUE, fringe, queue)
      queue = sortedQueue['queue']
      traversed += sortedQueue['fringe']
      visited.append(row) # set this row to visited
    return stack + queue

  def traverseBFS(self, start, goal):
    nodes = self.genericTraversal(start)
    path = None
    traversed = []
    for node in nodes:
      if len(node) < 2:
        continue
      traversed.append(node)
      if node[0] == start and node[-1] == goal:
        path = node
        break
    return {'path': path, 'list': traversed}

  def traverseDFS(self, start, goal):
    nodes = self.genericTraversal(start, self.SORT_STACK)
    path = None
    traversed = []
    for node in nodes:
      if len(node) < 2:
        continue
      traversed.append(node)
      if node[0] == start and node[-1] == goal:
        path = node
        break
    return {'path': path, 'list': traversed}

  def traverseUCS(self, start, goal):
    nodes = self.genericTraversal(start, self.SORT_PRIORITY_QUEUE)
    path = None
    traversed = []
    for node in nodes:
      if len(node) < 2:
        continue
      traversed.append(node)
      if node[0] == start and node[-1] == goal:
        path = node
        break
    return {'path': path, 'list': traversed}

  def traverse(self, start, goal, method=None):
    if start == goal:
      return []
    if method is None:
      method = self.SORT_QUEUE
    if method == self.SORT_STACK:
      return self.traverseDFS(start, goal)
    elif method == self.SORT_PRIORITY_QUEUE:
      return self.traverseUCS(start, goal)
    else:
      return self.traverseBFS(start, goal)

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

  def UninformedUCS(self, start, goal):
    traversed = self.traverse(start, goal, self.SORT_PRIORITY_QUEUE)
    self.animate(traversed['list'], traversed['path'])
    return traversed['path']

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
    start = self.selections[0]
    goal = self.selections[1]
    self.generateAdjacencyMatrix()
    print('Matrix: ', self.matrix)
    print('Searching from {} to {}'.format(start+1, goal+1))
    self.visitedNodes = {}
    search = getattr(self, method)(start, goal)
    for index in self.visitedNodes:
      self.canvas.itemconfig(index, fill=self.visitedNodes[index]['to'])
    #print('SEARCH RESULTS: ', search)
    # change edge colors
    if search is not False:
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
    return self.searchGeneric('UninformedBFS')

  def searchDepthFirst(self):
    return self.searchGeneric('UninformedDFS')

  def searchUniformCost(self):
    return self.searchGeneric('UninformedUCS')

  def searchBestFirst(self):
    # NOTE: heuristic never overestimate the cost from node to goal
    # similar to Uniform Cost Search but only deals with heuristic value
    # heuristic value = abs (current_node.x – goal.x) + abs (current_node.y – goal.y) - Manhattan Distance
    return self.searchGeneric('InformedBFS')

  def searchAStar(self):
    # NOTE: heuristic never overestimate the cost from node to goal
    # similar to Uniform Cost Search + heuristic value
    # A score = sum of cost of path from start node + heuristic value of node
    return self.searchGeneric('InformedASS')

  def doFileNew(self):
    self.selectedMenu = self.MENU_FILE_NEW = 1
    self.canvas.delete('all')
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
        node = Node(self.canvas, 'node{}'.format(index) , node['point']['x'], node['point']['y'], index + 1, 'red')
        self.nodes.append(node)
    if objects['edges']:
      for index, edge in enumerate(objects['edges']):
        edge = Edge(self.canvas, 'edge{}'.format(len(self.edges)), edge['point']['x'], edge['point']['y'], edge['point2']['x'], edge['point2']['y'], 'black')
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

  def buildMenu(self):
    self.menubar = Menu(self.tk)
    # File menu
    filemenu = Menu(self.menubar, tearoff=0)
    filemenu.add_command(label='New', command=self.doFileNew)
    filemenu.add_command(label='Open', command=self.doFileOpen)
    filemenu.add_command(label='Save', command=self.doFileSave)
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
    self.menubar.add_cascade(label='Search', menu=searchmenu)
    self.tk.config(menu=self.menubar)

  def createEdge(self, event):
    x, y = event.x, event.y
    # check for out of bounds
    size = self.DEFAULT_SIZE / 2
    x = self.width - size if x > (self.width - size) else x
    x = size + 5 if x < size + 5 else x
    y = self.height - size if y > (self.height - size) else y
    y = size + 5 if y < size + 5 else y
    if self.selectedMenu == self.MENU_GRAPH_NODE:
      node = Node(self.canvas, 'node{}'.format(len(self.nodes)) , x-size, y-size, len(self.nodes) + 1, 'red')
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
            for index, edge in enumerate(self.edges):
              self.canvas.itemconfig('edge{}'.format(index), fill='black')

Form().start()
