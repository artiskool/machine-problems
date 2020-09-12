from cmath import sqrt
from time import sleep
import json
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


class Machine:
    DEFAULT_SIZE = 20

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


class Line(Machine):
    def __init__(self, canvas, tag, x1, y1, x2, y2, color, width=3):
        self.point = Point(x1, y1)
        self.point2 = Point(x2, y2)
        self.distance = round(sqrt((x2 - x1)**2 + (y2 - y1)**2).real)
        self.shape = canvas.create_line(self.point.x, self.point.y, self.point2.x, self.point2.y, fill=color, width=width, tags=tag)
        px = (self.point.x + self.point2.x) / 2
        py = (self.point.y + self.point2.y) / 2
        self.text = canvas.create_text(px, py, text=self.distance, font='Verdana 14 italic', fill='red') # create label distance
        self.canvas = canvas

    def toJSON(self):
        return {'point': self.point.toJSON(), 'point2': self.point2.toJSON()}


class Circle(Machine):
    def __init__(self, canvas, tag, x, y, label, color='red'):
        self.point = Point(x, y)
        self.point2 = Point(x + self.DEFAULT_SIZE, y + self.DEFAULT_SIZE)
        self.box = Box(self.point.x, self.point.y, self.point.x + self.DEFAULT_SIZE, self.point.y + self.DEFAULT_SIZE)
        self.shape = canvas.create_oval(self.box.left, self.box.top, self.box.right, self.box.bottom, fill=color, tags=tag)
        self.text = canvas.create_text(self.point.x, self.point.y, text=label, font='Verdana 10 italic')
        self.canvas = canvas

    def toJSON(self):
        return {'point': self.point.toJSON(), 'point2': self.point2.toJSON(), 'box': self.box.toJSON()}


class Form(Machine):
    MENU_FILE_NEW = 1
    MENU_FILE_OPEN = 2
    MENU_FILE_SAVE = 4
    MENU_FILE_EXIT = 8
    MENU_SHAPE_CIRCLE = 16
    MENU_SHAPE_LINE = 32
    MENU_SEARCH_BLIND = 64
    MENU_SEARCH_INFORMED = 128

    def __init__(self, title='Machine Problem #1', width=800, height=600):
        self.reset()
        self.width = width
        self.height = height
        self.selectedMenu = None
        self.tk = Tk()
        self.canvas = Canvas(self.tk, height=self.height, width=self.width)
        #self.canvas.grid(row=0, column=0)
        self.tk.title(title)
        self.canvas.pack()
        self.canvas.bind('<ButtonRelease-1>', self.createLine)
        # START loading background image and default endpoints
        self.backgroundImage = ImageTk.PhotoImage(Image.open('./data/bantayan-island.png'))
        self.canvas.config(height=self.backgroundImage.height(), width=self.backgroundImage.width())
        self.background = self.canvas.create_image(self.backgroundImage.width() / 2, self.backgroundImage.height() / 2, image=self.backgroundImage, anchor=CENTER)
        self.redraw('./data/bantayan-island.json')
        # END loading background image and default endpoints
        self.buildMenu()
        self.visitedNodes = {}

    def start(self):
        self.tk.mainloop()

    def reset(self):
        self.circles = []
        self.lines = []
        self.selections = []

    def dumpShapes(self):
        shapes = {'circles': [], 'lines': []}
        for circle in self.circles:
            shapes['circles'].append(circle.toJSON())
        for line in self.lines:
            shapes['lines'].append(line.toJSON())
        return shapes

    def clearSelections(self):
        for index in self.selections:
            self.canvas.itemconfig('circle{}'.format(index), fill='red')
        self.selections = []

    def drawLine(self):
        circle = self.circles[self.selections[0]]
        circle2 = self.circles[self.selections[1]]
        self.clearSelections()
        center_x1 = circle.point.x + ((circle.point2.x - circle.point.x) / 2);
        center_y1 = circle.point.y + ((circle.point2.y - circle.point.y) / 2);
        center_x2 = circle2.point.x + ((circle2.point2.x - circle2.point.x) / 2);
        center_y2 = circle2.point.y + ((circle2.point2.y - circle2.point.y) / 2);
        # check if line already exists
        p1 = Point(center_x1, center_y1)
        p2 = Point(center_x2, center_y2)
        line_exists = False
        for line in self.lines:
            # check if p1 is in line.point
            p1p1 = (p1.x == line.point.x and p1.y == line.point.y) # check if p1 is in line.point
            p1p2 = (p1.x == line.point2.x and p1.y == line.point2.y) # check if p1 is in line.point2
            p2p1 = (p2.x == line.point.x and p2.y == line.point.y) # check if p2 is in line.point
            p2p2 = (p2.x == line.point2.x and p2.y == line.point2.y) # check if p2 is in line.point2
            if (p1p1 or p1p2) and (p2p1 or p2p2):
                line_exists = True
                break
        if line_exists:
            return # line already exists
        line = Line(self.canvas, 'line{}'.format(len(self.lines)), center_x1, center_y1, center_x2, center_y2, 'black')
        self.lines.append(line)

    def generateAdjacencyMatrix(self):
        matrix = {}
        circles_len = len(self.circles)
        # initialize matrix
        for row in range(circles_len):
            matrix[row] = {}
            for col in range(circles_len):
                # check if line exists between the two circles
                connected = self.isConnected(self.circles[row], self.circles[col], returnLineConnected=True)
                linked = True if row != col and connected['connected'] else False
                matrix[row][col] = {'linked': linked, 'circle1': row, 'circle2': col, 'line': connected['line']}
        return matrix

    def BFSShortestPath(self, matrix, start, goal):
        if start == goal:
            return []
        visited = []
        queue = [[start]]
        while queue:
            path = queue.pop(0) # get the first path from the queue and remove it
            row = path[-1] # get the last row from the path
            if row in visited: # check if path has already been visited
                # TODO: probably animate visited nodes???
                continue # skip if path already visited
            cols = matrix[row] # get all the columns
            for col in cols:
                if not cols[col]['linked']: # skip if matrix is not 1
                    continue
                # do the dance
                circle1 = 'circle{}'.format(cols[col]['circle1'])
                circle2 = 'circle{}'.format(cols[col]['circle2'])
                line = 'line{}'.format(cols[col]['line'])
                self.visitedNodes[circle1] = {'from': 'blue', 'to': 'red'}
                self.visitedNodes[circle2] = {'from': 'blue', 'to': 'red'}
                self.visitedNodes[line] = {'from': 'blue', 'to': 'black'}
                self.canvas.itemconfig(circle1, fill='blue')
                self.canvas.itemconfig(circle2, fill='blue')
                self.canvas.itemconfig(line, fill='blue')
                self.tk.update()
                rows = list(path)
                rows.append(col)
                queue.append(rows)
                sleep(2) # delay
                if col == goal: # found it
                    return rows
            visited.append(row) # set this row to visited
        return False

    def isConnected(self, circle, circle2, returnIndex=False, returnLineConnected=False):
        connected = False
        lineConnected = None
        for index, line in enumerate(self.lines):
            x1 = line.point.x
            y1 = line.point.y
            x2 = line.point2.x
            y2 = line.point2.y
            # check if left side of the line is within the first circle
            left_circle = True if circle.point.x < x1 and x1 < circle.point2.x and circle.point.y < y1 and y1 < circle.point2.y else False
            # check if right side of the line is within the first circle
            right_circle = True if circle.point.x < x2 and x2 < circle.point2.x and circle.point.y < y2 and y2 < circle.point2.y else False
            # check if left side of the line is within the second circle
            left_circle2 = True if circle2.point.x < x1 and x1 < circle2.point2.x and circle2.point.y < y1 and y1 < circle2.point2.y else False
            # check if right side of the line is within the second circle
            right_circle2 = True if circle2.point.x < x2 and x2 < circle2.point2.x and circle2.point.y < y2 and y2 < circle2.point2.y else False
            if (left_circle or right_circle) and (left_circle2 or right_circle2):
                connected = True if not returnIndex else index
                lineConnected = index
                break
        return {'connected': connected, 'line': lineConnected} if returnLineConnected else connected

    def searchBlind(self):
        start = self.selections[0]
        goal = self.selections[1]
        matrix = self.generateAdjacencyMatrix()
        print('Matrix: ', matrix)
        print('Searching from {} to {}'.format(start+1, goal+1))
        self.visitedNodes = {}
        search = self.BFSShortestPath(matrix, start, goal)
        for index in self.visitedNodes:
            self.canvas.itemconfig(index, fill=self.visitedNodes[index]['to'])
        print('SEARCH RESULTS: ', search)
        # change line colors
        circle = None
        for x in search:
            if circle is None:
                circle = self.circles[x]
                continue
            index = self.isConnected(circle, self.circles[x], True)
            self.canvas.itemconfig('line{}'.format(index), fill='green')
            circle = self.circles[x]
        self.clearSelections()

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
        if objects['circles']:
            for index, circle in enumerate(objects['circles']):
                circle = Circle(self.canvas, 'circle{}'.format(index) , circle['point']['x'], circle['point']['y'], index + 1, 'red')
                self.circles.append(circle)
        if objects['lines']:
            for index, line in enumerate(objects['lines']):
                line = Line(self.canvas, 'line{}'.format(len(self.lines)), line['point']['x'], line['point']['y'], line['point2']['x'], line['point2']['y'], 'black')
                self.lines.append(line)

    def doFileSave(self):
        self.selectedMenu = self.MENU_FILE_SAVE = 4
        filename = filedialog.asksaveasfilename(initialdir='/', title='Select json file', filetypes=(('json files', '*.json'),('all files', '*.*')), initialfile='machine_problem_1', defaultextension='.json')
        if len(filename) > 0:
            contents = json.dumps(self.dumpShapes(), sort_keys=True, indent=4)
            f = open(filename, 'w')
            f.write(contents)
            f.close()

    def doFileExit(self):
        self.selectedMenu = self.MENU_FILE_EXIT = 8
        self.canvas.quit()
        self.tk.destroy()

    def doShapeCircle(self):
        self.selectedMenu = self.MENU_SHAPE_CIRCLE = 16

    def doShapeLine(self):
        self.selectedMenu = self.MENU_SHAPE_LINE = 32
        self.clearSelections()

    def doSearchBlind(self):
        self.selectedMenu = self.MENU_SEARCH_BLIND = 64
        #matrix = self.generateAdjacencyMatrix()
        self.clearSelections()

    def doSearchInformed(self):
        self.selectedMenu = self.MENU_SEARCH_INFORMED
        self.clearSelections()

    def doNothing(self):
        self.selectedMenu = None

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
        # Shape menu
        shapemenu = Menu(self.menubar, tearoff=0)
        shapemenu.add_command(label='Draw Point', command=self.doShapeCircle)
        shapemenu.add_command(label='Draw Line', command=self.doShapeLine)
        self.menubar.add_cascade(label='Shape', menu=shapemenu)
        # Search menu
        searchmenu = Menu(self.menubar, tearoff=0)
        searchmenu.add_command(label='Blind', command=self.doSearchBlind)
        searchmenu.add_command(label='Informed', command=self.doSearchInformed)
        self.menubar.add_cascade(label='Search', menu=searchmenu)
        self.tk.config(menu=self.menubar)

    def createLine(self, event):
        x, y = event.x, event.y
        # check for out of bounds
        size = self.DEFAULT_SIZE / 2
        x = self.width - size if x > (self.width - size) else x
        x = size + 5 if x < size + 5 else x
        y = self.height - size if y > (self.height - size) else y
        y = size + 5 if y < size + 5 else y
        if self.selectedMenu == self.MENU_SHAPE_CIRCLE:
            circle = Circle(self.canvas, 'circle{}'.format(len(self.circles)) , x-size, y-size, len(self.circles) + 1, 'red')
            self.circles.append(circle)
        elif self.selectedMenu == self.MENU_SHAPE_LINE:
            # locate if clicked boundary within a circle and turn it to green, otherwise do nothing
            for index, circle in enumerate(self.circles):
                if circle.point.x < x and x < circle.point2.x and circle.point.y < y and y < circle.point2.y:
                    self.selections.append(index)
                    self.canvas.itemconfig('circle{}'.format(index), fill='green')
                    if len(self.selections) >= 2:
                        self.drawLine()
        elif self.selectedMenu == self.MENU_SEARCH_BLIND:
            # locate if clicked boundary within a circle and turn it to green, otherwise do nothing
            for index, circle in enumerate(self.circles):
                if circle.point.x < x and x < circle.point2.x and circle.point.y < y and y < circle.point2.y:
                    self.selections.append(index)
                    self.canvas.itemconfig('circle{}'.format(index), fill='green')
                    if len(self.selections) >= 2:
                        self.searchBlind()
                    else:
                        # change all line colors
                        for index, line in enumerate(self.lines):
                            self.canvas.itemconfig('line{}'.format(index), fill='black')


Form().start()
