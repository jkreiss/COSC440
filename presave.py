import tensorflow as tf
import numpy as np
#
#
# def get_flattened_position(scaled_values, depth):
#     """
#     Take in a 3-dimensional position and map it onto multiple 3-D grids, where each grid is the original grid divided
#     by a factor of 2
#
#     For example, in a 4x4x4 grid, there are 3 multi-resolutions -- 1x1x1, 2x2x2, 4x4x4
#     We will ignore the 1x1x1 dimension because all points would have position 0.
#     If the query point came in as 0.5, 0.25, 0.125 for this grid, the positional encoder should produce two values:
#     1 (position of the point in a 2x2x2 grid which has 8 positions total)
#     6 (position of the point in a 4x4x4 grid which has 64 positions total)
#
#     Input: tensor of float32, shape [N, 3] of N points in 3-D space normalized such that all points are [0,1)
#     Output: tensor of int32, shape [N, 1] of N flattened positions
#     """
#
#     grid_size = 2 ** (depth - 1)
#     scaled_values = tf.floor(scaled_values * grid_size)
#     x, y, z = scaled_values[:, 0], scaled_values[:, 1], scaled_values[:, 2]
#     flattened = x + (y * grid_size) + (z * (grid_size ** 2))
#
#     return tf.cast(flattened, tf.int32)
#
#
# print(get_flattened_position(tf.convert_to_tensor([[0.5, 0.25, 0.125], [0.5, 0.25, 0.125]]), 3))

# def ray_attenuation(attenuations, distances, magnitudes, near, far):
#     """
#     Computes the sum of attenuation for each sampled set of attenuations given the distances
#      of the points along the source to detector axis and magnitudes of the vectors.
#
#     A basic algorithm for this is to find the difference of distances and multiply the attenuations each by their distance
#     to get a weighted sum. Slightly more correct (and what my reference implementation does) is to use the magnitude
#     of the ray, compute the total distance from near to far along that ray using the magnitude, interpolate the attenuations
#     between points, and then create a weighted sum including the first and last points attenuation as an assumed value
#     for the region not covered by the distance between the first and last point sampled.
#
#     You can implement the simpler algorithm for this as with the given geometry it doesn't make much difference.
#
#     :param attenuations: tensor of floats [n_rays, n_points] where n_rays is the number of rays per image used and n_points is the number of points per ray used
#     :param distances: tensor of floats [n_points] which is the distance along each ray in the source to detector axis
#     :param magnitudes: tensor of floats [n_rays] which is the magnitude of each directional ray
#     :param near: float, the closest distance to region of interest
#     :param far: float, the farthest distance to region of interest
#     :return: tensor of floats [n_rays] which is the attenuation value for each ray
#     """
#     # todo fill this in
#     differences = distances[1:] - distances[:-1]
#     mids = 0.5 * (attenuations[:, :-1] + attenuations[:, 1:])
#     weighted_sum = tf.reduce_sum(mids * differences, axis=1)
#
#     return weighted_sum * magnitudes
#
#
#
# attenuations = tf.convert_to_tensor(np.array([[0.5, 0.3, 0.1]]), dtype=tf.float32)
# distances = tf.convert_to_tensor(np.array([0.9, 1.0, 1.1]), dtype=tf.float32)
# magnitudes = tf.convert_to_tensor(np.array([[2.0]]), dtype=tf.float32) # not realistic, just for the test
# near = np.float64(0.9)
# far = np.float64(1.1)
# result = ray_attenuation(attenuations, distances, magnitudes, near, far)
# # You should get close to exact values here
# print("ray_attenuation output:")
# print(result)

"""tf.Tensor([[0.12000003]], shape=(1, 1), dtype=float32)
get_flattened_position output:
tf.Tensor([7 7], shape=(2,), dtype=int32)
tf.Tensor([42 42], shape=(2,), dtype=int32)
tf.Tensor([292 292], shape=(2,), dtype=int32)
tf.Tensor([2184 2184], shape=(2,), dtype=int32)
tf.Tensor([16912 16912], shape=(2,), dtype=int32)
tf.Tensor([133152 133153], shape=(2,), dtype=int32)
tf.Tensor([1056832 1056834], shape=(2,), dtype=int32)
tf.Tensor([8421504 8421509], shape=(2,), dtype=int32)
"""
def rays_to_points(rays, n_points, near, far):
    """
    Computes the sample points for the given rays.
    First, samples a uniform distribution to produce a set of scalars for n_points.
    Second, multiplies those scalars by the distance between far-near to produce a vector multiplier that is within the region of interest.
    Third, multiplies each ray's directional vector by the vector multiplier and adds it to the ray origin to produce a point.

    :param rays: tensor of float32 [N, 6] for N rays defined by 6 values: A 3D point for the origin of the ray and a 3D vector of the direction of the ray
    :param n_points: int, the number of points to sample along each ray
    :param near: float, the closest distance to scale the ray by
    :param far: float, the farthest distance to scale the ray by
    :return: a two element tuple: (points, scalars) where
      points is [N, n_points, 3] for N rays with n_points each of 3D points
      scalars is [n_points] for the scalar values used to multiply the direction ray vector
        (as these are randomly generated they need to be returned for later use)
    """
    # todo fill this in
    scalars = tf.random.uniform([n_points])
    scalars *= far - near
    origins = tf.cast(rays[:, 0:3], tf.float32)
    directions = tf.cast(rays[:, 3:6], tf.float32)
    scalars_expanded = tf.reshape(scalars, [1, n_points, 1])  # [1, n_points, 1]
    directions_expanded = tf.expand_dims(directions, axis=1)  # [N, 1, 3]
    origins_expanded = tf.expand_dims(origins, axis=1)  # [N, 1, 3]

    # Multiply directions by scalar distances, then add to origins
    points = origins_expanded + scalars_expanded * directions_expanded  # [N, n_points, 3]

    return points, scalars
    points = directions * scalars + origins
    return points, scalars

    return None

rays = tf.convert_to_tensor(np.array([[1.,0.,0.,-1.,0.1,0.1]]), dtype=tf.float64)  # not realistic just for the test
near = np.float64(0.9)
far = np.float64(1.1)
# n_points = np.int32(10)
# small test
points, scalars = rays_to_points(rays, 10, near, far)
#NOTE: there is randomness in the ray generation so you won't get the exact values shown
print("rays_to_points output:")
print(points)
print(scalars)
'''ays_to_points output:
tf.Tensor(
[[[ 0.09659314  0.09034069  0.09034069]
  [ 0.08707216  0.09129278  0.09129278]
  [ 0.0536455   0.09463545  0.09463545]
  [ 0.04227487  0.09577251  0.09577251]
  [ 0.01069508  0.09893049  0.09893049]
  [-0.00671153  0.10067115  0.10067115]
  [-0.03321439  0.10332144  0.10332144]
  [-0.06313688  0.10631369  0.10631369]
  [-0.07580807  0.10758081  0.10758081]
  [-0.0987646   0.10987646  0.10987646]]], shape=(1, 10, 3), dtype=float64)'''
