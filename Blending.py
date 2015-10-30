import cv2

def gaussian_pyramid(img):
    G  = img.copy()
    gp = [G]

    for i in xrange(3):
        G = cv2.pyrDown(G)
        gp.append(G)

    return gp


def laplacian_pyramid(img):
    gp = gaussian_pyramid(img)
    lp = [gp[3]]

    for i in xrange(3, 0, -1):
        GE = cv2.pyrUp(gp[i])
        L = cv2.subtract(gp[i-1], GE)
        lp.append(L)

    return lp


def img_blending(left, right):
    LS = []
    
    left_rows, left_cols, left_dept = left.shape
    right_rows, right_cols, right_dept = left.shape

    min_rows = min(left_rows, right_rows)
    min_cols = min(left_cols, right_cols)

    if (min_rows % 32 != 0):
        min_rows -= (min_rows % 32)
    if (min_cols % 32 != 0):
        min_cols -= (min_cols % 32)

    # blending left, mid
    LA = laplacian_pyramid(left[0:min_rows, 0:min_cols])
    LB = laplacian_pyramid(right[0:min_rows, 0:min_cols])
    for la, lb in zip(LA, LB):
        rows,cols,dpt = la.shape
        ls = la + lb
        LS.append(ls)

    return LS