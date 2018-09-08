import numpy as np
from os import system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def getEdges(xpos,ypos,zpos,size):
    size=size/2
    points = np.array([[ -1*size +xpos, -1*size +ypos, -1*size +zpos],
                        [ 1*size +xpos, -1*size +ypos, -1*size +zpos ],
                        [ 1*size +xpos,  1*size +ypos, -1*size +zpos],
                        [-1*size +xpos,  1*size +ypos, -1*size +zpos],
                        [-1*size +xpos, -1*size +ypos,  1*size +zpos],
                        [ 1*size +xpos, -1*size +ypos,  1*size +zpos ],
                        [ 1*size +xpos,  1*size +ypos,  1*size +zpos],
                        [-1*size +xpos,  1*size +ypos,  1*size +zpos]])
    edges = [
            [points[0], points[1], points[2], points[3]],  #back 0-right -1- up -2 - left 3 -down -0
            [points[4], points[5], points[6], points[7]],  #front 4-right -5- up -6 - left 7 -down -4
            [points[0], points[4], points[7], points[3]], #left
            [points[1], points[5], points[6], points[2]], #right
            [points[3], points[7], points[6], points[2]],  #top
            [points[0], points[4], points[5], points[1]]   #bottom
            ]



   
    return edges

def getcubeFaces(xpos,ypos,zpos,size):
    size=size/2
    points = np.array([[ -1*size +xpos, -1*size +ypos, -1*size +zpos],
                        [ 1*size +xpos, -1*size +ypos, -1*size +zpos ],
                        [ 1*size +xpos,  1*size +ypos, -1*size +zpos],
                        [-1*size +xpos,  1*size +ypos, -1*size +zpos],
                        [-1*size +xpos, -1*size +ypos,  1*size +zpos],
                        [ 1*size +xpos, -1*size +ypos,  1*size +zpos ],
                        [ 1*size +xpos,  1*size +ypos,  1*size +zpos],
                        [-1*size +xpos,  1*size +ypos,  1*size +zpos]])
    edges = [
            [points[0], points[1], points[2], points[3]],  #back 0-right -1- up -2 - left 3 -down -0
            [points[4], points[5], points[6], points[7]],  #front 4-right -5- up -6 - left 7 -down -4
            [points[0], points[4], points[7], points[3]], #left
            [points[1], points[5], points[6], points[2]], #right
            [points[3], points[7], points[6], points[2]],  #top
            [points[0], points[4], points[5], points[1]]   #bottom
            ]



    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    return faces

    

xpos=4
ypos=0
zpos=4
size=2

system('cls')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

faces = getcubeFaces(xpos,ypos,zpos,size)
faces.set_facecolor((1,0,0,0.5))
ax.add_collection3d(faces)

ax.scatter3D(xpos,ypos,zpos,marker=".",facecolors='black', edgecolors='none',s=100) 

xpos=0
ypos=0
zpos=0
size=2


faces = getcubeFaces(xpos,ypos,zpos,size)
faces.set_facecolor((0,0,1,0.5))
ax.add_collection3d(faces)

ax.scatter3D(xpos,ypos,zpos,marker=".",facecolors='black', edgecolors='none',s=100) 


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

