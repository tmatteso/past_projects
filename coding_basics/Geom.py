#  File: Geom.py

#  Description: Read a file and generate objects according from it, then run a series of tests on the objects and print out the results

#  Student Name: Tomas Matteson

#  Student UT EID: tlm3347

#  Partner Name: Mohammad Abdelrazzaq

#  Partner UT EID: mja2753

#  Course Name: CS 313E

#  Unique Number: 86325

#  Date Created: 6/19/18

#  Date Last Modified: 6/21/18


# need to import math for its hypot method
import math


class Point(object):
    # constructor
    # x and y are floats
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    # get distance
    # other is a Point object
    def dist(self, other):
        distance = math.hypot(self.x - other.x, self.y - other.y)
        return distance

    # get a string representation of a Point object
    # takes no arguments
    # returns a string
    def __str__(self):
        return '(' + str(self.x) + ", " + str(self.y) + ")"

    # test for equality
    # other is a Point object
    # returns a Boolean
    def __eq__(self, other):
        tol = 1.0e-16
        return ((abs(self.x - other.x) < tol) and (abs(self.y - other.y) < tol))


class Circle(object):
    # constructor
    # x, y, and radius are floats
    def __init__(self, radius=1, x=0, y=0):
        self.radius = radius
        self.center = Point(x, y)

    # compute cirumference
    def circumference(self):
        return (2.0 * math.pi * self.radius)

    # compute area
    def area(self):
        return (math.pi * self.radius * self.radius)

    # determine if point is strictly inside circle
    def point_inside(self, p):
        if (self.center.dist(p) < self.radius):
            return True
        else:
            return False

    # determine if a circle is strictly inside this circle
    def circle_inside(self, c):
        distance = self.center.dist(c.center)
        if (distance + c.radius) < self.radius:
            return True
        else:
            return False

    # determine if a circle c intersects this circle (non-zero area of overlap)
    # the only argument c is a Circle object
    # returns a boolean
    def does_intersect(self, other):
        distance = self.center.dist(other.center)
        if (self.radius + other.radius >= distance):
            return True
        else:
            return False

    # determine the smallest circle that circumscribes a rectangle
    # the circle goes through all the vertices of the rectangle
    # the only argument, r, is a rectangle object
    def circle_circumscribes(self, other):
        new_circle = Circle()
        new_circle.radius = other.ul.dist(other.lr) / 2
        new_circle_x = ((other.lr.x - other.ul.x) / 2) + other.ul.x
        new_circle_y = other.ul.y - ((other.ul.y - other.lr.y) / 2)
        new_circle.center = Point(new_circle_x, new_circle_y)
        return new_circle

    # string representation of a circle
    # takes no arguments and returns a string
    def __str__(self):
        return str(self.radius)+';'+str(self.center)

    # test for equality of radius
    # the only argument, other, is a circle
    # returns a boolean
    def __eq__(self, other):
        tol = 1.0e-16
        return (abs(self.radius - other.radius) < tol)



class Rectangle(object):
    # constructor
    def __init__(self, ul_x=0, ul_y=1, lr_x=1, lr_y=0):
        if ((ul_x < lr_x) and (ul_y > lr_y)):
            self.ul = Point(ul_x, ul_y)
            self.lr = Point(lr_x, lr_y)
        else:
            self.ul = Point(0, 1)
            self.lr = Point(1, 0)

    # determine length of Rectangle (distance along the x axis)
    # takes no arguments, returns a float
    def length(self):
        length = self.lr.x - self.ul.x
        return (length)

    # determine width of Rectangle (distance along the y axis)
    # takes no arguments, returns a float
    def width(self):
        width = self.ul.y - self.lr.y
        return (width)

    # determine the perimeter
    # takes no arguments, returns a float
    def perimeter(self):
        length = self.lr.x - self.ul.x
        width = self.ul.y - self.lr.y
        perimeter = (2 * length) + (2 * width)
        return (perimeter)

    # determine the area
    # takes no arguments, returns a float
    def area(self):
        length = self.lr.x - self.ul.x
        width = self.ul.y - self.lr.y
        area = length * width
        return (area)

    # determine if a point is strictly inside the Rectangle
    # takes a point object p as an argument, returns a boolean
    def point_inside(self, p):
        if ((self.ul.x - self.lr.x) > (self.ul.x - p.x)) and (self.ul.y - self.lr.y) > (self.ul.y - p.y):
            return True
        else:
            return False

    # determine if another Rectangle is strictly inside this Rectangle
    # takes a rectangle object r as an argument, returns a boolean
    # should return False if self and r are equal
    def rectangle_inside(self, r):
        if (self.ul.x > r.ul.x) and (self.ul.y > r.ul.y) and (r.lr.x > self.lr.x) and (r.lr.y > self.lr.y):
            return True
        else:
            return False

    # determine if two Rectangles overlap (non-zero area of overlap)
    # takes a rectangle object r as an argument returns a boolean
    def does_intersect(self, other):
        if (self.ul.x < other.lr.x and self.lr.x > other.ul.x):
            if (other.lr.y < self.ul.y and other.ul.y > self.lr.y ):
                return True
            else:
                return False

        else:
            return False

    # determine the smallest rectangle that circumscribes a circle
    # sides of the rectangle are tangents to circle c
    # takes a circle object c as input and returns a rectangle object
    def rect_circumscribe(self, c):
        new_rect = Rectangle()
        new_rect.ul = Point(c.center.x - c.radius, c.center.y + c.radius)
        new_rect.lr = Point(c.center.x + c.radius, c.center.y - c.radius)
        return new_rect

    # give string representation of a rectangle
    # takes no arguments, returns a string
    def __str__(self):
        return str(self.ul) + ';' + str(self.lr)

    # determine if two rectangles have the same length and width
    # takes a rectangle other as argument and returns a boolean
    def __eq__(self, other):
        tol = 1.0e-16
        return (abs(self.length() - other.length()) < tol) and (abs(self.width() - other.width()) < tol)


def main():
    # open the file
    file = open('geom.txt', 'r')
    lines = file.read().splitlines()
    # take the correct elements from the first line and apply them to make a Point called P
    first_line_elements = lines[0].split()
    p_co_ord = first_line_elements[0:2]
    p = []
    for i in p_co_ord:
        p.append(float(i))
    P = Point(p[0], p[1])
    # take the correct elements from the second line and apply them to make a Point called Q
    second_line_elements = lines[1].split()
    q_co_ord = second_line_elements[0:2]
    q = []
    for i in q_co_ord:
        q.append(float(i))
    Q = Point(q[0], q[1])
    # take the correct elements from the third line and apply them to make a Circle called C
    third_line_elements = lines[2].split()
    c_co_ords = third_line_elements[0:2]
    c_point = []
    for i in c_co_ords:
        c_point.append(float(i))
    c_radius = float(third_line_elements[2])
    C = Circle(c_radius, c_point[0], c_point[1])
    # take the correct elements from the fourth line and apply them to make a Circle called D
    fourth_line_elements = lines[3].split()
    d_co_ords = fourth_line_elements[0:2]
    d_point = []
    for i in d_co_ords:
        d_point.append(float(i))
    d_center = Point(d_point[0], d_point[1])
    d_radius = float(fourth_line_elements[2])
    D = Circle(d_radius, d_point[0], d_point[1])
    # take the correct elements from the fifth line and apply them to make a Rectangle called G
    fifth_line_elements = lines[4].split()
    g_ul = fifth_line_elements[0:2]
    G_ul = []
    for i in g_ul:
        G_ul.append(float(i))
    g_lr = fifth_line_elements[2:4]
    G_lr = []
    for i in g_lr:
        G_lr.append(float(i))

    G = Rectangle(G_ul[0], G_ul[1], G_lr[0], G_lr[1])
    # take the correct elements from the sixth line and apply them to make a Rectangle called H
    sixth_line_elements = lines[5].split()
    h_ul = sixth_line_elements[0:2]
    H_ul = []
    for i in h_ul:
        H_ul.append(float(i))
    h_lr = sixth_line_elements[2:4]
    H_lr = []
    for i in h_lr:
        H_lr.append(float(i))

    H = Rectangle(H_ul[0], H_ul[1], H_lr[0], H_lr[1])
    # close the file
    file.close()
    # Printing formatted as indicated in the Assignment
    print("Coordinates of P: " + str(P))
    print("Coordinates of Q: " + str(Q))
    print("Distance between P and Q: %f" % P.dist(Q))
    print("Circle C: " + str(C))
    print("Circle D: " + str(D))
    print("Circumference of C: %f" % C.circumference())
    print("Area of D: %f" % D.area())
    if (C.point_inside(P)):
        print("P is inside C")
    else:
        print("P is not inside C")

    if (D.circle_inside(C)):
        print("C is inside D")
    else:
        print("C is not inside D")
    if C.does_intersect(D):
        print("C does intersect D")
    else:
        print("C does not intersect D")
    if D == C:
        print("C is equal to D")
    else:
        print("C is not equal to D")
    print("Rectangle G: " + str(G))
    print("Rectangle H: " + str(H))
    print("Length of G: " + str(G.length()))
    print("Width of H: " + str(H.width()))
    print("Perimeter of G: " + str(G.perimeter()))
    print("Area of H: " + str(H.area()))
    if (G.point_inside(P)):
        print("P is inside G")
    else:
        print("P is not inside G")
    if (H.rectangle_inside(G)):
        print("G is inside H")
    else:
        print("G is not inside H")
    if G.does_intersect(H):
        print("G does overlap H")
    else:
        print("G does not overlap H")
    print("Circle that circumscribes G: " + str(C.circle_circumscribes(G)))
    print("Rectangle that circumscribes D: " + str(G.rect_circumscribe(D)))
    if G == H:
        print("Rectangle G is equal to H")
    else:
        print("Rectangle G is not equal to H")

if __name__ == "__main__":
  main()
