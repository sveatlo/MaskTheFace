import numpy as np

def rect_to_bb(rect):
    x1 = rect.left()
    x2 = rect.right()
    y1 = rect.top()
    y2 = rect.bottom()
    return (x1, x2, y2, x1)


# from https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py#L44
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def line_intersection(line1, line2):
    # mid = int(len(line1) / 2)
    start = 0
    end = -1
    line1 = ([line1[start][0], line1[start][1]], [line1[end][0], line1[end][1]])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    x = []
    y = []
    flag = False

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return flag, x, y

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    segment_minX = min(line2[0][0], line2[1][0])
    segment_maxX = max(line2[0][0], line2[1][0])

    segment_minY = min(line2[0][1], line2[1][1])
    segment_maxY = max(line2[0][1], line2[1][1])

    if (
        segment_maxX + 1 >= x >= segment_minX - 1
        and segment_maxY + 1 >= y >= segment_minY - 1
    ):
        flag = True

    return flag, x, y

def remove_suffix(s: str, suffix: str):
    if not s.endswith(suffix):
        return s

    return s[:-len(suffix)]

