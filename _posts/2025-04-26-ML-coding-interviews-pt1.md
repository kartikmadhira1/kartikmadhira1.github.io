---
title:  "Computer Vision/Machine Learning coding interviews Part-1"
mathjax: true
layout: post
categories: media
---

Over the past 7 years, I have had the chance to give multiple interviews for the computer vision and machine learning roles. This post is to delve down into the common questions asked in these interviews. These are in general 45-60 minute interviews asked with a real world problem in mind. 



# Question - You are given a sample of n 3D point cloud points. Write a program to cluster the points points into clusterd objects.

One way to think of this question is to ask - how are these points built? Is there a special pattern about them? Clustering can be the easy part - We build a k-means cluster or do a euclidean cluster. But for that we will need an efficient way to access the neighbours for each of the points. How do we do that?

Point clouds are stored in special data structures called the [Kd-trees.](https://en.wikipedia.org/wiki/K-d_tree#:~:text=k%2Dd%20trees%20are%20a,Creating%20point%20clouds.) Essentially, instead of taking a point and trying to search for the closest point with a complexity of O(n^2) for a particular axis, we are trying to limit the search by carefully forming a tree of points, storing the points by splitting from a particular sorted axis. Think of this as forming a binary tree but we switch between axis:

1. At depth 0, split by x axis
2. At depth 1, split by y axis
3. At depth 2, split by z axis
3. At depth 4, split by x axis
..
n. At depth%3, split by [0=x-axis, 1=y-axis, 2=z-axis]


{% highlight python %}

points = [(2,7,3), (3,6,2), (6,12,1), (9,1,7), (13,15,6), (17,15,13)]



# for the 1st point we will sort it on x-axis, take the median as the parent which is (9,1,7)
# next, left subtree will be chosen for values between x <=9 and the right with x>9 -> 
# (2,7,3), (3,6,2), (6,12,1) for x <=9 (left subtree) and (13,15,6), (17,15,13) for x>9 (right subtree)
# Between (2,7,3), (3,6,2), (6,12,1) on left, the parent chosen will be median of y-axis sorted = 6, 7, 12 which is 6
# hence left subtree chosen in (2, 7, 3)
# continue building the same way.

{% endhighlight %}




For building the Kd-tree:


{% highlight python %}
# randomly generate a list of 3d points, 3d points in tuple

points = [(random.randint(0, 20), random.randint(0, 20), random.randint(0, 20)) for _ in range(10)]

class KdNode:
    def __init__(self, xyz_tuple):
        self.x = xyz_tuple[0]
        self.y = xyz_tuple[1]
        self.z = xyz_tuple[2]
        self.left = None
        self.right = None


def build_kdtree(points, depth=0):

    if not points:
        return
    sorted_list = sorted(points, key=lambda x: x[depth%3])
    
    middle_idx = len(points)//2
    median_val = points[middle_idx]

    node = KdNode(median_val)

    node.left = build_kdtree(points[:middle_idx], depth+1)
    node.right = build_kdtree(points[middle_idx+1:], depth+1)

    return node

{% endhighlight %}

The overall time complexity of building a KD-tree is O(n log n).


Clustering:

{% highlight python %}
def get_euclid_dist(root, query_point):

    return math.sqrt((root.x - query_point.x)**2 + (root.y - query_point.y)**2 + (root.z - query_point.z)**2)



def cluster(root, query_point, visit_map, cluster_id, depth, radius=2):
    if root in visit_map:
        cluster(root.left, query_point, visit_map, cluster_id, depth+1)
        cluster(root.right, query_point, visit_map, cluster_id, depth+1)
    else:
        dist = get_euclid_dist(root, query_point)
        visit_map.add(root.tuple)
        if  dist < radius:
            root.cluster_id = cluster_id
    
    axis = depth%3
    diff = query_point.tuple[axis] - root.tuple[axis]

    # search on left if the initial crossing is on left
    if diff <=0:
        cluster(root.left, query_point, visit_map, cluster_id, depth+1, radius)
    else:
        cluster(root.right, query_point, visit_map, cluster_id, depth+1, radius)

    
    # if diff is less than the radius search on the other side
    if abs(diff) <= radius:
        if diff <=0:
            cluster(root.right, query_point, visit_map, cluster_id, depth+1, radius)
        else:
            cluster(root.left, query_point, visit_map, cluster_id, depth+1, radius)



cluster_id = 0
def euclid_cluster(root, points):

    for each_point in points:

        if each_point in visit_map:
            continue
        else:
            query_point = KdNode(each_point)
            cluster(root, each_point, visit_map, cluster_id)
            cluster_id+=1


{% endhighlight %}


In my opinion, either of these are definitely a 45-60 minute interview questions, not both.



# Question: Given a set of 3D points with random gaussian noise, fit a plane.

This question screams of RANSAC. At this point, if you have a CV background you must have encountered RANSAC in some form. This is a pretty robust outlier rejection method. The last I had implemented one was back in 2018, 7 years back and I got asked this a few months ago with a slight catch - fit a plane instead of a line. 



{% highlight python %}

points = [(2,7,3), (3,6,2), (6,12,1), (9,1,7), (13,15,6), (17,15,13)]

def ransac_pseudo(points, threshold):

    
    best_fit_plane = None
    max_points = -1e18
   
    for n iterations:
        #get 3 random points and fit a plabe
        rand_3_points = random(points)
        curr_plane = fit_plane(rand_3_points)
        
        # check the dist to the plane for each point,
        #if less than threshold, include in set
        
        agreed_points = 0
        for each_point in points:
            dist = dist_to_plane(each_point, curr_plane)
            if dist < threshold:
                agreed_points +=1

        if agreed_points > max_points:
            best_fit_plane = curr_plane
            max_points = agreed_points 


{% endhighlight %}

For a primer of plane equations and basics, you can see them [here.](https://github.com/kartikmadhira1/kartikmadhira1.github.io/blob/master/_data/post1/Ransac.pdf)