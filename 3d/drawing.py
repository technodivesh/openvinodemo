import cv2
import numpy as np

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

def drawPoints(img, points, color=(0, 255, 0)):
    for i,point in enumerate(points):
        # cv2.putText(img, i, org, font,  
        #            fontScale, color, thickness, cv2.LINE_AA) 
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)

def drawCross(img, params, center=(100, 100), scale=30.0):
    R = cv2.Rodrigues(params[1:4])[0]

    points = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    points = np.dot(points, R.T)
    points2D = points[:, :2]

    points2D = (points2D * scale + center).astype(np.int32)
    
    cv2.line(img, (center[0], center[1]), (points2D[0, 0], points2D[0, 1]), (255, 0, 0), 3)
    cv2.line(img, (center[0], center[1]), (points2D[1, 0], points2D[1, 1]), (0, 255, 0), 3)
    cv2.line(img, (center[0], center[1]), (points2D[2, 0], points2D[2, 1]), (0, 0, 255), 3)

def drawMesh(img, shape, mesh, color=(255, 0, 0)):
    for triangle in mesh:
        point1 = shape[triangle[0]].astype(np.int32)
        point2 = shape[triangle[1]].astype(np.int32)
        point3 = shape[triangle[2]].astype(np.int32)

        cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]), (255, 0, 0), 1)
        cv2.line(img, (point2[0], point2[1]), (point3[0], point3[1]), (255, 0, 0), 1)
        cv2.line(img, (point3[0], point3[1]), (point1[0], point1[1]), (255, 0, 0), 1)

def drawProjectedShape(img, x, projection, mesh, params, lockedTranslation=False):
    localParams = np.copy(params)

    if lockedTranslation:
        localParams[4] = 100
        localParams[5] = 200

    projectedShape = projection.fun(x, localParams)

    drawPoints(img, projectedShape.T, (0, 0, 255))
    drawMesh(img, projectedShape.T, mesh)
    drawCross(img, params)
