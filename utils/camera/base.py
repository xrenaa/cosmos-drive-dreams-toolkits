# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
import cv2
from utils.pcd_utils import interpolate_polyline_to_points
from typing import List, Union
from abc import abstractmethod

DEPTH_MAX = 122.5

def make_sure_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def make_sure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

class CameraBase:
    def __init__(self):
        pass

    @abstractmethod
    def ray2pixel_torch(self, rays: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def ray2pixel_np(self, rays: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _get_rays(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_rays_from_cached(self) -> torch.Tensor:
        """
        This provides 2x speed up for f-theta semantic buffer generation.
        """
        if not hasattr(self, 'rays_cached'):
            self.rays_cached = self._get_rays()
        return self.rays_cached
    
    def transform_points_torch(self, points: torch.Tensor, tfm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (M, 3)
            tfm: (4, 4)
        Returns:
            points_transformed: (M, 3)
        """
        transformed_points = tfm[:3, :3] @ points.T + tfm[:3, 3].unsqueeze(-1)
        return transformed_points.T
    
    def transform_points_np(self, points: np.ndarray, tfm: np.ndarray) -> np.ndarray:
        """
        Args:
            points: (M, 3)
            tfm: (4, 4)
        Returns:
            points_transformed: (M, 3)
        """
        transformed_points = tfm[:3, :3] @ points.T + tfm[:3, 3].reshape(-1, 1)
        return transformed_points.T
    
    def _get_rays_posed(self, camera_poses: torch.Tensor):
        """
        Args:
            camera_poses: (N, 4, 4)
        Returns:
            ray_o: (N, H, W, 3), camera origin
            ray_d: (N, H, W, 3), camera rays
        """
        rays_in_cam = self._get_rays_from_cached() # shape (H, W, 3)
        rays_d_in_world = torch.einsum('bij,hwj->bhwi', camera_poses[:, :3, :3], rays_in_cam) # shape (N, H, W, 3)
        rays_o_in_world = camera_poses[:, :3, 3].unsqueeze(-2).unsqueeze(-2).expand_as(rays_d_in_world) # shape (N, H, W, 3)
        
        return rays_o_in_world, rays_d_in_world

    def _clip_polyline_to_image_plane(self, points_in_cam: np.ndarray) -> np.ndarray:
        """
        Args:
            points_in_cam: np.ndarray
                shape: (M, 3), a polyline, they are connected.
        Returns:
            points: np.ndarray
                shape: (M', 3), a polyline, but we clip the points to positive z if the points are behind the camera.
        """        
        depth = points_in_cam[:, 2]
        # go through all the edges of the polyline. 
        eps = 1e-1
        cam_coords_cliped = []
        for i in range(len(points_in_cam) - 1):
            pt1 = points_in_cam[i]
            pt2 = points_in_cam[i+1]

            if depth[i] >= 0 and depth[i+1] >= 0:
                cam_coords_cliped.append(pt1)
            elif depth[i] < 0 and depth[i+1] < 0:
                continue
            else:
                # clip the line to the image boundary
                if depth[i] >= 0:
                    # add the first point
                    cam_coords_cliped.append(pt1)

                    # calculate the intersection point and add it
                    t = (- pt2[2]) / (pt1[2] - pt2[2]) + eps
                    inter_pt = pt2 + t * (pt1 - pt2)
                    cam_coords_cliped.append(inter_pt)
                else:
                    # calculate the intersection point and add it
                    t = (- pt1[2]) / (pt2[2] - pt1[2]) + eps
                    inter_pt = pt1 + t * (pt2 - pt1)
                    cam_coords_cliped.append(inter_pt)

        # handle the last point, if its depth > 0 and not already added
        if depth[-1] >= 0:
            cam_coords_cliped.append(points_in_cam[-1])

        cam_coords_cliped = np.stack(cam_coords_cliped, axis=0) # shape (M', 3)
        
        return cam_coords_cliped
        
    """
    Drawing related functions
    """
    def draw_points(
            self, 
            camera_poses: Union[torch.Tensor, np.ndarray], 
            points: Union[torch.Tensor, np.ndarray], 
            colors: Union[torch.Tensor, np.ndarray, None] = None, 
            radius: int = 1,
            fast_impl_when_radius_gt_1: bool = True
        ) -> np.ndarray:
        """
        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4) 
            points: torch.Tensor or np.ndarray
                shape: (M, 3) 
            colors: torch.Tensor or np.ndarray or None
                shape: (M, 3) in uint8
            radius: int, 
                radius of the point
            fast_impl_when_radius_gt_1: bool, 
                if True, use cv2.circle to draw the point when radius > 1
        Returns:
            canvas: np.ndarray
                shape: (N, H, W, 3) or (H, W, 3)
                dtype: np.uint8
        """
        draw_images = []
        camera_poses = make_sure_numpy(camera_poses)
        points = make_sure_numpy(points)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]

        if colors is not None:
            colors = make_sure_numpy(colors)
        else:
            colors = np.tile([[255, 0, 0]], (points.shape[0], 1))

        for camera_to_world in camera_poses:
            points_in_cam = self.transform_points_np(points, np.linalg.inv(camera_to_world))
            uv_coords = self.ray2pixel_np(points_in_cam)
            depth = points_in_cam[:, 2]
            valid_depth_mask = depth > 0

            u_round = np.round(uv_coords[:, 0]).astype(np.int32)
            v_round = np.round(uv_coords[:, 1]).astype(np.int32)

            valid_uv_mask = (u_round >= 0) & (u_round < self.width) & (v_round >= 0) & (v_round < self.height)
            valid_mask = valid_depth_mask & valid_uv_mask

            u_valid = u_round[valid_mask]
            v_valid = v_round[valid_mask]
            z_valid = depth[valid_mask]
            colors_valid = colors[valid_mask]

            sorted_indices = np.argsort(z_valid, axis=0)[::-1]
            u_valid = u_valid[sorted_indices]
            v_valid = v_valid[sorted_indices]
            colors_valid = colors_valid[sorted_indices]

            if radius > 1 and fast_impl_when_radius_gt_1 is False:
                canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                for u, v, color in zip(u_valid, v_valid, colors_valid):
                    cv2.circle(canvas, (u.item(), v.item()), radius, color.tolist(), -1)
                canvas = np.array(canvas, dtype=np.uint8)

            # radius = 1 or we want fast impl
            else: 
                canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                canvas[v_valid, u_valid] = colors_valid # fill from the farthest point to the nearest point

                # use fast impl when radius > 1
                if radius > 1:
                    canvas_accum = np.zeros_like(canvas)
                    i_shifts = np.arange(-radius//2, radius//2+1)
                    j_shifts = np.arange(-radius//2, radius//2+1)
                    for i in i_shifts:
                        for j in j_shifts:
                            # use torch.roll to shift the canvas 
                            canvas_shifted = np.roll(canvas, shift=(i, j), axis=(0, 1))
                            canvas_accum = np.maximum(canvas_accum, canvas_shifted)
                    canvas = canvas_accum

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if draw_images.shape[0] == 1:
            draw_images = draw_images[0]

        return draw_images

    def draw_line_depth(
            self, 
            camera_poses: Union[torch.Tensor, np.ndarray], 
            polylines: List, 
            radius: int = 8,
            colors: np.ndarray = None,
            segment_interval: float = 0,
        ) -> np.ndarray:
        """
        draw lines on the image, and the drawed pixel value is related to the depth of the points.
        The polyline can be out of boundary, use cv2.clipLine to clip the line to the image boundary, or abandon the line.
        Then use cv2.line to draw the line.

        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            polylines: list of list of points, 
                each point is in 3D (x, y, z)
            radius: int, 
                radius of the drawn circle
            colors: np.ndarray, 
                shape: (3, ), dtype: np.uint8
            segment_interval: float, 
                if > 0, the polyline is segmented into segments with the interval

        Returns:
            draw_images: np.ndarray
                shape: (N, H, W, 3) or (H, W, 3)
                dtype: np.uint8
        """
        draw_images = []
        camera_poses = make_sure_numpy(camera_poses)

        if colors is None:
            colors = np.array([255, 255, 255])

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]
        
        for camera_to_world in camera_poses:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            H, W = canvas.shape[:2]

            for polyline in polylines:
                if len(polyline) < 2:
                    continue
                if isinstance(polyline, list):
                    polyline = np.array(polyline)

                if segment_interval > 0:
                    polyline = interpolate_polyline_to_points(polyline, segment_interval)
                
                points_in_cam = self.transform_points_np(polyline, np.linalg.inv(camera_to_world))
                uv_coords = self.ray2pixel_np(points_in_cam)
                depth = points_in_cam[:, 2]

                u_round = np.round(uv_coords[:, 0]).astype(np.int32)
                v_round = np.round(uv_coords[:, 1]).astype(np.int32)
                valid_uv_mask = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)

                # filter out the polyline if all points are out of the image boundary
                if (~valid_uv_mask).all():
                    continue

                # if depth all greater than DEPTH_MAX, skip
                if depth.min() > DEPTH_MAX:
                    continue

                for i in range(len(u_round) - 1):
                    if depth[i] < 0 and depth[i+1] < 0:
                        continue
                    
                    if depth[i] * depth[i+1] < 0:
                        # if the two points are on different sides of the camera, we first clip the 3d point in the back to the camera plane + epsilon
                        # and then reproject it to the image plane, calculate the uv coordinate
                        pt1 = points_in_cam[i]
                        pt2 = points_in_cam[i+1]

                        # make sure pt1 is in front of the camera, pt2 is behind the camera
                        if depth[i] < 0:
                            pt1, pt2 = pt2, pt1

                        # clip the line to the image boundary
                        eps = 2e-1
                        t = (- pt2[2]) / (pt1[2] - pt2[2]) + eps
                        pt2 = t * pt1 + (1 - t) * pt2

                        # project the point to the image plane
                        pt1_norm = pt1[:3] / pt1[2]
                        pt2_norm = pt2[:3] / pt2[2]

                        pixel1 = self.ray2pixel_np(pt1_norm)[0] 
                        pixel2 = self.ray2pixel_np(pt2_norm)[0]
                    else:
                        pixel1 = np.array([u_round[i], v_round[i]])
                        pixel2 = np.array([u_round[i+1], v_round[i+1]])

                    try:
                        clipped, pixel1, pixel2 = \
                            cv2.clipLine((0, 0, W, H), pixel1.astype(np.int32), pixel2.astype(np.int32))
                    except:
                        breakpoint()

                    depth_mean = (depth[i] + depth[i+1]) / 2
                    depth_mean = np.clip(depth_mean, 0, DEPTH_MAX)
                    fill_weight = (2 * (DEPTH_MAX - depth_mean)) / 255
                    fill_value = (fill_weight * colors).astype(np.uint8).tolist()

                    cv2.line(canvas, tuple(pixel1), tuple(pixel2), fill_value, radius)

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if draw_images.shape[0] == 1 and len(draw_images.shape) == 3:
            draw_images = draw_images[0]
        
        return draw_images


    def draw_hull_depth(
            self, 
            camera_poses: Union[torch.Tensor, np.ndarray], 
            hulls: List,
            colors: np.ndarray = None
        ) -> torch.Tensor:
        """
        draw hulls on the image, and the drawed pixel value is related to the depth of the points.
        The hull can be out of boundary, use cv2.clipLine to clip the line to the image boundary, or abandon the line.
        Then use cv2.line to draw the line.

        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            hulls: list of list of points, 
                each point is in 3D (x, y, z)
            colors: np.ndarray, 
                shape: (3, ), dtype: np.uint8

        Returns:
            draw_images: (N, H, W, 3) or (H, W, 3), image with hulls drawn
        """
        draw_images = []
        camera_poses = make_sure_numpy(camera_poses)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]
        
        for camera_to_world in camera_poses:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            H, W = canvas.shape[:2]

            for hull in hulls:
                if len(hull) < 3:
                    continue
                
                points_in_cam = self.transform_points_np(hull, np.linalg.inv(camera_to_world))
                uv_coords = self.ray2pixel_np(points_in_cam).astype(np.int32)
                depth = points_in_cam[:, 2]
                valid_depth_mask = depth > 0

                u_round = uv_coords[:, 0]
                v_round = uv_coords[:, 1]
                valid_uv_mask = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)
                valid_mask = valid_depth_mask & valid_uv_mask

                # filter out the polyline if all points are out of the image boundary
                if not valid_mask.any():
                    continue

                # if depth all greater than DEPTH_MAX, skip
                if depth.min() > DEPTH_MAX:
                    continue

                # project again with clipped points
                points_in_cam_clipped = self._clip_polyline_to_image_plane(points_in_cam)
                uv_coords = self.ray2pixel_np(points_in_cam_clipped).astype(np.int32)
                depth_mean = points_in_cam_clipped[:, 2].mean()

                # cv2.clipLine to limit the points to the image boundary
                uv_coords_line_clipped = []
                for i in range(len(uv_coords) - 1):
                    pixel1 = uv_coords[i]
                    pixel2 = uv_coords[i+1]
                    clipped, pixel1, pixel2 = cv2.clipLine((0, 0, W, H), pixel1, pixel2)
                    uv_coords_line_clipped.append(pixel1)
                    uv_coords_line_clipped.append(pixel2)

                hull_in_uv = np.array(uv_coords_line_clipped).astype(np.int32)

                # create convex hull and draw on the image
                convex_hull_in_uv = cv2.convexHull(hull_in_uv.astype(np.int32)) # shape (N, 1, 2)
                fill_weight = (2 * (DEPTH_MAX - depth_mean)) / 255
                fill_value = (fill_weight * colors).astype(np.uint8).tolist()
                cv2.fillPoly(canvas, [convex_hull_in_uv], fill_value)

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if draw_images.shape[0] == 1 and len(draw_images.shape) == 3:
            draw_images = draw_images[0]

        return draw_images