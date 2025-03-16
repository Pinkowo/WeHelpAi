import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def vector(self):
        return [self.p2.x - self.p1.x, self.p2.y - self.p1.y]

    def slope(self):
        return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)

    def parallel(self, line):
        return self.slope() == line.slope()

    def perpendicular(self, line):
        return self.slope() == -1 / line.slope()


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius

    def distance(self, other):
        p = [self.center.x, self.center.y]
        q = [other.center.x, other.center.y]
        return math.dist(p, q)

    def intersect(self, other):
        return self.distance(other) < self.radius + other.radius


class Polygon:
    def __init__(self, points):
        self.points = points

    def perimeter(self):
        points = self.points
        length = len(points)
        perimeter = 0

        for i in range(length):
            perimeter += math.dist(
                [points[i].x, points[i].y],
                [
                    points[(i + 1) % length].x,
                    points[(i + 1) % length].y,
                ],
            )
        return perimeter


###################################################

# Line A
plA1 = Point(2, 4)
plA2 = Point(-6, 1)
lA = Line(plA1, plA2)

# Line B
plB1 = Point(2, 2)
plB2 = Point(-6, -1)
lB = Line(plB1, plB2)

# Line C
plC1 = Point(-1, 6)
plC2 = Point(-4, -4)
lC = Line(plC1, plC2)

# Circle A
pcA = Point(6, 3)
cA = Circle(pcA, 2)

# Circle B
pcB = Point(8, 1)
cB = Circle(pcB, 1)

# Polygon A
ppA1 = Point(2, 0)
ppA2 = Point(5, -1)
ppA3 = Point(4, -4)
ppA4 = Point(-1, -2)
pA = Polygon([ppA1, ppA2, ppA3, ppA4])

###################################################

print(f"Are Line A and Line B parallel? {lA.parallel(lB)}")
print(f"Are Line C and Line A perpendicular? {lA.perpendicular(lC)}")
print(f"Print the area of Circle A. {cA.area()}")
print(f"Do Circle A and Circle B intersect? {cA.intersect(cB)}")
print(f"Print the perimeter of Polygon A. {pA.perimeter()}")
