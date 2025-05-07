import random
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CircleSet(Dataset):
    def __init__(self, minX, minY, maxX, maxY, minRadius, maxRadius, numCircles, numPoints):
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.oneHot = numCircles > 0
        self.numCircles = numCircles if numCircles > 0 else numCircles * -1
        self.circles: list[tuple[float, float, float]] = []
        self.num_classes = self.numCircles
        if numCircles < 0:
            # place non‐overlapping circles
            max_attempts = 100000
            attempts = 0
            while len(self.circles) < self.numCircles and attempts < max_attempts:
                r = random.uniform(minRadius, maxRadius)
                x = random.uniform(minX + r, maxX - r)
                y = random.uniform(minY + r, maxY - r)
                # check against all existing circles
                if all((x-cx)**2 + (y-cy)**2 >= (r+cr)**2 
                       for cx, cy, cr in self.circles):
                    self.circles.append((x, y, r))
                attempts += 1
            if len(self.circles) < self.numCircles:
                self.numCircles = len(self.circles)
                print(f"Warning: only {self.numCircles} circles could be placed without overlap.")
        
        else:
            # get random x,y,radius
            for i in range(self.numCircles):
                radius = random.uniform(minRadius, maxRadius)
                x = random.uniform(minX + radius, maxX - radius)
                y = random.uniform(minY + radius, maxY - radius)
                self.circles.append((x, y, radius))
        self.numPoints = numPoints
        self.points: list[tuple[float, float]] = []
        self.x= []
        self.y= []
        # get random points
        if self.oneHot:
            for i in range(numPoints):
                x = random.uniform(minX, maxX)
                y = random.uniform(minY, maxY)
                self.points.append((x, y))
        else:
            while len(self.points) < numPoints:
                x = random.uniform(minX, maxX)
                y = random.uniform(minY, maxY)
                # check if point is inside any circle
                if any((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2 for cx, cy, radius in self.circles):
                    self.points.append((x, y))
        for point in self.points:
            x, y = point
            self.x.append([x, y])
            circles = self.circles_of_point(point)
            if self.oneHot:
                one_hot = [0] * self.numCircles
                for circle in circles:
                    one_hot[circle] = 1
                self.y.append(one_hot)
            else:
                # if the point is inside a circle, set the label to the index of the circle
                if len(circles) > 0:
                    self.y.append(circles[0])
                else:
                    self.y.append(-1)
            
        self.x_tensor = torch.tensor(self.x, dtype=torch.float32, device=device)
        self.y_tensor = torch.tensor(self.y, dtype=torch.float32, device=device) if self.oneHot else torch.tensor(self.y, dtype=torch.long, device=device)
        


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]   
    
    def circles_of_point(self, point):
        x, y = point
        circles = []
        for i in range(self.numCircles):
            circle = self.circles[i]
            circleX, circleY, radius = circle
            if (x - circleX) ** 2 + (y - circleY) ** 2 <= radius ** 2:
                circles.append(i)
        return circles

    def print(self):
        for x,y in self:
            if self.oneHot:
                print(f"point: {x}, label: {y}")
            else:
                print(f"point: X:{x[0]:.2f} Y:{x[1]:.2f}, circle nr: {y}")
                
                
    
    def display(self):
        fig, ax = plt.subplots()
        # draw each circle
        for (x, y, radius) in self.circles:
            circle = Circle((x, y), radius, fill=False, edgecolor='blue')
            ax.add_patch(circle)
        # draw points
        if self.points:
            x_points, y_points = zip(*self.points)
            ax.scatter(x_points, y_points, color='red', zorder=5)
        # set limits with a margin
        ax.set_xlim(self.minX - self.maxRadius, self.maxX + self.maxRadius)
        ax.set_ylim(self.minY - self.maxRadius, self.maxY + self.maxRadius)
        ax.set_aspect('equal')
        plt.show()
