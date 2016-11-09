"""
Author: Ng Jun Wei

File Handler module

Handles the writing and reading of contents from file.
"""


def read_from_file(path):
    # type: (string) -> (int, int)[][]
    """
    Purpose:
        Reads in a list of list of points from a file at specified path.
    Parameters:
        path - the specified file path
    Returns:
        A list of list of tuples containing two integers that represents a point.
    """
    points_list = []
    with open(path, 'r') as f:
        contents = f.readlines()
        for line in contents:
            points = line.split()

            ret = []
            for p in points:
                coord = p.split(',')
                ret.append((float(coord[0]), float(coord[1])))
            points_list.append(ret)

    return points_list


def write_to_file(path, points_list):
    # type: (string, (int, int)[][]) -> void
    """
    Purpose:
        Write a list of list of points to a file at specified path.
    Parameters:
        path - the specified file path
        points_list - the list of list of points
    Return:
        void
    """
    with open(path, 'a+') as f:
        for points in points_list:
            for i in range(len(points)):
                f.write(str(points[i][0]) + ',' + str(points[i][1]))
                if i < len(points) - 1:
                    f.write('\t')
            f.write('\n')


def write_single_list_to_file(path, points):
    # type: (string, (int, int)[]) -> void
    """
    Purpose:
        Write a list of points to the end of file at specified path.
    Parameters:
        path - the specified file path
        points - the list of points
    Return:
        void
    """
    with open(path, 'a+') as f:
        for i in range(len(points)):
            f.write(str(points[i][0]) + ',' + str(points[i][1]))
            if i < len(points) - 1:
                f.write('\t')
        f.write('\n')
