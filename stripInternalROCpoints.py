#!/usr/bin/env python
# coding: utf-8

def stripInternalROCpoints(fpr, tpr):
    # strips internal points on verticals, horizontals or diagonals, leaving only the endpoints
    # of each line segment.
    import numpy as np
    ep          = 1 * (10 ** -4)
    currentX    = -99
    currentY    = -99
    newfpr      = []
    newtpr      = []
    firstPoint  = True
    secondPoint = False
    for nextX, nextY in zip(fpr, tpr):
        pass
        if not firstPoint and not secondPoint:  # the first point cannot be an internal point
            # is it an internal point? check 3 ways:
            internal = False
            # 1. last point and next point are horizontal
            if lastY == currentY and currentY == nextY:
                internal = True

            # 2. last point and next point are vertical
            elif lastX == currentX and currentX == nextX:
                internal = True

            # 3. last point and next point are in a diagonal line
            else:
                if currentX == lastX:    # not diagonal
                    internal = False
                elif nextX == currentX:  # not diagonal
                    internal = False
                else:
                    slope1  = (currentY-lastY)/(currentX-lastX)
                    slope2  = (nextY-currentY)/(nextX-currentX)
                    fuzzyEQ = lambda a, b, ep: (np.abs(a - b) < ep)
                    if fuzzyEQ(slope1, slope2, ep):
                        internal = True
                    #endif
                #endif
            #endif

            if not internal:  # if not internal, then add it to newfpr, newtpr
                newfpr.extend([currentX])
                newtpr.extend([currentY])
            lastX    = currentX
            lastY    = currentY
            currentX = nextX
            currentY = nextY
        else:
            if secondPoint:
                secondPoint = False  #assuming we do not need this below here
                newfpr.extend([currentX])
                newtpr.extend([currentY])
                lastX       = currentX
                lastY       = currentY
                currentX    = nextX
                currentY    = nextY
            if firstPoint:
                firstPoint  = False  #assuming we do not need this below here
                secondPoint = True
                currentX    = nextX
                currentY    = nextY
        #endif
    #endfor

    # add the last point
    newfpr.extend([currentX])
    newtpr.extend([currentY])
    return newfpr, newtpr
#enddef
