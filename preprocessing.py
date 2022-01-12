import os
from random import randrange, random, uniform

import plydata
import pandas
import numpy as np
import plyfile
import matplotlib.pyplot as plt


def read_ply_xyzrgbnormal(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert (os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        # vertices[:, 3] = plydata['vertex'].data['red']
        # vertices[:, 4] = plydata['vertex'].data['green']
        # vertices[:, 5] = plydata['vertex'].data['blue']

        # compute normals
        # xyz = np.array([[x, y, z] for x, y, z in plydata["vertex"].data])
        # face = np.array([f[0] for f in plydata["face"].data])
        # nxnynz = compute_normal(xyz, face)
        # vertices[:, 6:] = nxnynz
        # vertices[:, 3:] = nxnynz

    return vertices


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[faces[:, 0]] += n
    normals[faces[:, 1]] += n
    normals[faces[:, 2]] += n
    normalize_v3(normals)

    return normals


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= (lens + 1e-8)
    arr[:, 1] /= (lens + 1e-8)
    arr[:, 2] /= (lens + 1e-8)
    return arr


def slim(vert):
    result = np.array(vert[0])
    for i in vert:
        if random() < .40:
            result = np.vstack([result, i])
    return result


def add_noise(original, percentage_of_noise, variety):
    # minimum = min(original[:, 0])
    # maximum = max(original[:, 0])
    minimum = 0
    maximum = 0.14
    result = np.array(original[0])
    for i in original:
        # result = np.vstack([result, i])
        if random() < percentage_of_noise:
            # result = np.vstack([result, np.random.uniform(low=minimum, high=maximum, size=(3,))])
            result = np.vstack(
                [result, np.array([i[0] + uniform(-variety, variety), i[1] + uniform(-variety, variety),
                                   i[2] + uniform(-variety, variety)])])
        else:
            result = np.vstack([result, i])
    return result

    ""
    # return original + np.random.normal(0, .1, original.shape)


# path = os.path('../data/bunny/data/bun000.ply')
# print(path)

vertices = read_ply_xyzrgbnormal('data/bunny/data/ear_back.ply')
# vertices = np.vstack([vertices,read_ply_xyzrgbnormal('data/bunny/data/top2.ply')])
# vertices = np.vstack([vertices,read_ply_xyzrgbnormal('data/bunny/data/top3.ply')])
vertices = slim(vertices)
print(vertices)
print(np.random.uniform(low=0.5, high=1, size=(3,)))
print(vertices.shape)
# o = add_noise(vertices, .001)
# print(o.shape)
# for i in vertices:
#     print(i)
# N = 200
# x = 5 * np.random.rand(N)
# y = 5 * np.random.rand(N)
# z = f(x, y)
# m = polyfit2d(x,y,z)
# nx, ny = 40, 40
# X, Y = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))
# # Z = polyval2d(X, Y, m)
# plt.show()


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, color="blue")
# # ax.contour3D(X, Y, Z, 50, cmap='RdGy')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()
# vertices2 = add_noise(vertices, .3, 0.1)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], s=1, color="blue")
# # ax.contour3D(X, Y, Z, 50, cmap='RdGy')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()
vertices2 = add_noise(vertices, .5, 0.005)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], s=1, color='blue')
# ax.contour3D(X, Y, Z, 50, cmap='RdGy')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
