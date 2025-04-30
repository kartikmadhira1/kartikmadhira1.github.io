import numpy as np
import random



# random points 
points = random.randint(0,10, (100, 3))


def ransac(points, N, threshold):

    for _ in range(N):

        
