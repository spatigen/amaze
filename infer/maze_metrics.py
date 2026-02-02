# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Reward Functions for Maze Generation GRPO Training

This module implements three types of reward functions for evaluating maze generation:

原文：
1. 输入 IA：转化成二进制的original_image IB：m_original_img IC：解空间的白色区域（像素点是1，其他部分都是0），也就是mask_img O：模型输出generated_image
目标：
1. 第一个reward：O和IC相乘（也就是mask）提取出O的background，再转成binary，和IA（binary形式）计算l1 loss
2. 第二个reward：查看IC（解空间）范围内是否存在O的solution path并用二进制表示，判断solution是否合法：1. 是否连通（不和黑色区域交叉） 2. 起点和终点在定义的起点终点白色区域内

修改后的逻辑：
1. 输入 original_image：转化成二进制的原始图像 marked_original_img：带标记的原始图像 solution_space_mask：解空间的白色区域（像素点是1，其他部分都是0），也就是从数据集的mask_img得到的解空间 generated_image：模型输出的生成图像
目标：
1. 第一个reward（背景保持）：使用background_mask（1 - solution_space_mask）提取generated_image和original_image的背景（墙壁）区域，转成binary进行比较，计算l1 loss。确保模型正确保持了墙壁结构。
2. 第二个reward（路径质量）：查看solution_space_mask（解空间）范围内是否存在generated_image的solution path并用二进制表示，判断solution是否合法：1. 是否连通（不和黑色区域交叉） 2. 起点和终点在定义的起点终点白色区域内
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Union, Tuple
import skimage.metrics
from scipy import ndimage
from skimage.morphology import skeletonize

# 由 infer_bagel 根据 config.is_circle 设置（config/maze.py 或 infer_auto.sh 传入）
IS_CIRCLE = False
IS_ISCIRCLE = False  # 与 IS_CIRCLE 同义，两处引用保持一致

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        IoU score (0-1, higher is better)
    """
    if mask1.shape != mask2.shape:
        return 0.0

    intersection = np.sum((mask1 == 1) & (mask2 == 1))
    union = np.sum((mask1 == 1) | (mask2 == 1))

    return intersection / union if union > 0 else 0.0



def load_image_as_tensor(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Load an image and convert to tensor.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)

    Returns:
        Image tensor of shape (C, H, W)
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
        image_array = np.array(image).astype(np.float32) / 255.0
        # Convert from HWC to CHW
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        return image_tensor
    except Exception as e:
        #print(f"Error loading image {image_path}: {e}")
        # Return a black image as fallback
        return torch.zeros(3, target_size[0], target_size[1])


def pil_image_to_tensor(pil_image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Convert PIL image directly to tensor.

    Args:
        pil_image: PIL Image object
        target_size: Target size for resizing (height, width)

    Returns:
        Image tensor of shape (C, H, W)
    """
    try:
        if pil_image is None:
            return torch.zeros(3, target_size[0], target_size[1])

        # Convert to RGB if not already
        image = pil_image.convert('RGB')

        # Resize to target size
        image = image.resize((target_size[1], target_size[0]))  # PIL uses (width, height)

        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        # Convert from HWC to CHW
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        return image_tensor
    except Exception as e:
        #print(f"Error converting PIL image to tensor: {e}")
        # Return a black image as fallback
        return torch.zeros(3, target_size[0], target_size[1])


def tensor_to_binary_maze(image_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Convert image tensor to binary maze representation.

    Args:
        image_tensor: Image tensor of shape (C, H, W) or (H, W)
        threshold: Threshold for binarization

    Returns:
        Binary maze array where 1 = wall, 0 = path
    """
    if image_tensor.dim() == 3:
        # Convert RGB to grayscale
        gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
    else:
        gray = image_tensor

    # Binarize: assuming dark pixels (< threshold) are walls (1), light pixels are paths (0)
    binary = (gray < threshold).float().numpy()
    return binary.astype(np.uint8)


def extract_white_regions(image_tensor: torch.Tensor, white_threshold: float = 0.8) -> np.ndarray:
    """
    Extract white regions (traversable paths) from image tensor.

    Args:
        image_tensor: Image tensor of shape (C, H, W)
        white_threshold: Threshold for considering a pixel as white

    Returns:
        Binary mask where 1 = white region (traversable path), 0 = non-white
    """
    if image_tensor.dim() == 3:
        # Convert RGB to grayscale
        gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
    else:
        gray = image_tensor

    # Extract white regions: pixels with high brightness
    white_mask = (gray > white_threshold).float().numpy().astype(np.uint8)
    return white_mask


def extract_blue_path(image_tensor: torch.Tensor,
                     blue_hue_range: Tuple[int, int] = (100, 130),
                     saturation_threshold: int = 50,
                     value_threshold: int = 50) -> np.ndarray:
    """
    Extract blue path pixels from solution image using HSV color space.

    Args:
        image_tensor: Image tensor of shape (C, H, W)
        blue_hue_range: HSV hue range for blue color (0-179 in OpenCV)
        saturation_threshold: Minimum saturation for color detection (0-255)
        value_threshold: Minimum value (brightness) for color detection (0-255)

    Returns:
        Binary mask where 1 = blue path pixel, 0 = non-blue
    """
    try:
        # Convert tensor to numpy array (H, W, C)
#        if image_tensor.device != torch.device('cpu'):
#            image_array = image_tensor.detach().cpu().numpy()
#        else:
#            image_array = image_tensor.numpy()
#
#        image_array = image_array.transpose(1, 2, 0)  # CHW -> HWC
#        image_array = (image_array * 255).astype(np.uint8)  # Convert to 0-255 range

        # Convert RGB to HSV for better color detection
        hsv = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2HSV)

        # Create blue mask based on HSV thresholds
        lower_blue = np.array([blue_hue_range[0], saturation_threshold, value_threshold])
        upper_blue = np.array([blue_hue_range[1], 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Convert to binary mask (0 and 1)
        blue_mask = (blue_mask > 0).astype(np.uint8)

        return blue_mask

    except Exception as e:
        #print(f"Error extracting blue path: {e}")
        # Return empty mask on error
        raise e


def extract_solution_space_from_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract solution space directly from mask PNG where 255 represents solution space and 0 represents other areas.

    Args:
        mask_tensor: Mask image tensor (C, H, W) where 255 = solution space, 0 = other

    Returns:
        Binary mask where 1 = solution space, 0 = other areas
    """
    try:
        # Convert tensor to numpy array
        # 先转换为float32以支持bfloat16等数据类型
        if mask_tensor.device != torch.device('cpu'):
            mask_array = mask_tensor.detach().float().cpu().numpy()
        else:
            mask_array = mask_tensor.float().numpy()

        # Convert from CHW to HWC if needed
        if mask_array.ndim == 3 and mask_array.shape[0] == 3:
            mask_array = mask_array.transpose(1, 2, 0)  # CHW -> HWC
        elif mask_array.ndim == 3 and mask_array.shape[0] == 1:
            mask_array = mask_array.squeeze(0)  # Remove channel dimension if grayscale

        # Convert to grayscale if RGB
        if mask_array.ndim == 3:
            # Convert RGB to grayscale
            gray_mask = 0.299 * mask_array[:,:,0] + 0.587 * mask_array[:,:,1] + 0.114 * mask_array[:,:,2]
        else:
            gray_mask = mask_array

        # Convert to 0-255 range if in 0-1 range
        if gray_mask.max() <= 1.0:
            gray_mask = gray_mask * 255.0

        # Create binary mask: 255 -> 1 (solution space), other values -> 0
        solution_mask = (gray_mask >= 127.5).astype(np.uint8)  # Use threshold of 127.5 to handle potential artifacts

        # #print(f"从mask PNG提取的解空间像素数: {np.sum(solution_mask)}")
        # #print(f"mask的尺寸: {solution_mask.shape}")

        return solution_mask

    except Exception as e:
        #print(f"Error extracting solution space from mask: {e}")
        import traceback
        traceback.print_exc()
        if mask_tensor.dim() == 3:
            return np.zeros((mask_tensor.shape[1], mask_tensor.shape[2]), dtype=np.uint8)
        else:
            return np.zeros(mask_tensor.shape, dtype=np.uint8)


def extract_red_markers(image_tensor: torch.Tensor,
                       red_hue_ranges: List[Tuple[int, int]] = [(0, 10), (160, 179)],
                       saturation_threshold: int = 100,
                       value_threshold: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract red start point (solid circle) and end point (X mark) from image.

    Args:
        image_tensor: Image tensor of shape (C, H, W)
        red_hue_ranges: HSV hue ranges for red color (red wraps around in HSV)
        saturation_threshold: Minimum saturation for color detection
        value_threshold: Minimum value (brightness) for color detection

    Returns:
        Tuple of (red_mask, start_mask, end_mask) where masks are binary arrays
    """
    try:
        # Convert tensor to numpy array
        # 先转换为float32以支持bfloat16等数据类型
        if image_tensor.device != torch.device('cpu'):
            image_array = image_tensor.detach().float().cpu().numpy()
        else:
            image_array = image_tensor.float().numpy()

        image_array = image_array.transpose(1, 2, 0)  # CHW -> HWC
        # 确保值范围在[0,1]，然后转换为uint8
        image_array = np.clip(image_array, 0.0, 1.0)
        image_array = (image_array * 255).astype(np.uint8)

        # Convert RGB to HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # 调试：检查HSV值的分布
        h_values = hsv[:,:,0]
        s_values = hsv[:,:,1]
        v_values = hsv[:,:,2]
        
        # Debug: 检查是否有红色像素（通过RGB直接检查）
        red_pixels_rgb = np.sum((image_array[:,:,0] > 200) & (image_array[:,:,1] < 100) & (image_array[:,:,2] < 100))
        # #print(f"      RGB红色像素数（R>200, G<100, B<100）: {red_pixels_rgb}")

        # Create red mask (red color wraps around in HSV space)
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for hue_range in red_hue_ranges:
            lower_red = np.array([hue_range[0], saturation_threshold, value_threshold])
            upper_red = np.array([hue_range[1], 255, 255])
            mask_part = cv2.inRange(hsv, lower_red, upper_red)
            red_mask = cv2.bitwise_or(red_mask, mask_part)
        
        # Debug: 输出红色像素统计
        red_pixel_count = np.sum(red_mask > 0)
        # if red_pixel_count == 0:
        #     #print(f"      extract_red_markers调试: 未检测到红色像素")
        #     #print(f"      HSV范围: H=[{h_values.min()}, {h_values.max()}], S=[{s_values.min()}, {s_values.max()}], V=[{v_values.min()}, {v_values.max()}]")
        #     #print(f"      红色HSV阈值: H在[{red_hue_ranges[0][0]}-{red_hue_ranges[0][1]}]或[{red_hue_ranges[1][0]}-{red_hue_ranges[1][1]}], S>={saturation_threshold}, V>={value_threshold}")
        #     #print(f"      RGB图像值范围: R=[{image_array[:,:,0].min()}, {image_array[:,:,0].max()}], G=[{image_array[:,:,1].min()}, {image_array[:,:,1].max()}], B=[{image_array[:,:,2].min()}, {image_array[:,:,2].max()}]")
        # else:
            # #print(f"      extract_red_markers: 检测到{red_pixel_count}个红色像素")

        # Find connected components to identify start and end points
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        start_mask = np.zeros_like(red_mask)
        end_mask = np.zeros_like(red_mask)

        if len(contours) >= 2:
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # First contour is start point (solid circle)
            cv2.fillPoly(start_mask, [contours[0]], 255)
            
            # Second contour is end point (X mark)
            if len(contours) > 1:
                cv2.fillPoly(end_mask, [contours[1]], 255)
        elif len(contours) == 1:
            # Only one red region found, assume it's start point
            cv2.fillPoly(start_mask, [contours[0]], 255)

        red_mask = (red_mask > 0).astype(np.uint8)
        start_mask = (start_mask > 0).astype(np.uint8)
        end_mask = (end_mask > 0).astype(np.uint8)

        return red_mask, start_mask, end_mask

    except Exception as e:
        #print(f"Error extracting red markers: {e}")
        if image_tensor.dim() == 3:
            empty_mask = np.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=np.uint8)
        else:
            empty_mask = np.zeros(image_tensor.shape, dtype=np.uint8)
        return empty_mask, empty_mask, empty_mask



def extract_maze_structure(binary_maze: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract structural information from binary maze.

    Args:
        binary_maze: Binary maze array

    Returns:
        Dictionary containing maze structure information
    """
    # Find walls and paths
    walls = binary_maze
    paths = 1 - binary_maze

    # Find connected components of paths
    labeled_paths, num_path_components = ndimage.label(paths)

    # Extract skeleton of path network
    path_skeleton = skeletonize(paths.astype(bool)).astype(np.uint8)

    return {
        'walls': walls,
        'paths': paths,
        'labeled_paths': labeled_paths,
        'num_path_components': num_path_components,
        'path_skeleton': path_skeleton
    }


def find_maze_solution(binary_maze: np.ndarray, start_pos: Tuple[int, int] = None,
                      end_pos: Tuple[int, int] = None) -> np.ndarray:
    """
    Find solution path in maze using A* algorithm (simplified version).

    Args:
        binary_maze: Binary maze where 0 = path, 1 = wall
        start_pos: Starting position (row, col)
        end_pos: Ending position (row, col)

    Returns:
        Binary array where 1 indicates solution path
    """
    height, width = binary_maze.shape

    # Auto-detect start and end if not provided
    if start_pos is None:
        # Find top-left corner path
        path_positions = np.where(binary_maze == 0)
        if len(path_positions[0]) > 0:
            start_pos = (path_positions[0][0], path_positions[1][0])
        else:
            start_pos = (0, 0)

    if end_pos is None:
        # Find bottom-right corner path
        path_positions = np.where(binary_maze == 0)
        if len(path_positions[0]) > 0:
            end_pos = (path_positions[0][-1], path_positions[1][-1])
        else:
            end_pos = (height-1, width-1)

    # Simple BFS pathfinding
    from collections import deque

    queue = deque([(start_pos, [start_pos])])
    visited = set([start_pos])

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        (row, col), path = queue.popleft()

        if (row, col) == end_pos:
            # Found solution, create solution array
            solution = np.zeros_like(binary_maze)
            for r, c in path:
                solution[r, c] = 1
            return solution

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if (0 <= new_row < height and 0 <= new_col < width and
                binary_maze[new_row, new_col] == 0 and  # Not a wall
                (new_row, new_col) not in visited):

                visited.add((new_row, new_col))
                queue.append(((new_row, new_col), path + [(new_row, new_col)]))

    # No solution found
    return np.zeros_like(binary_maze)


class MazeRewardFunction:
    """Main reward function class for maze generation evaluation."""

    def __init__(self,
                 weights: Dict[str, float] = None):
        """
        Initialize the maze reward function.

        Args:
            weights: Weights for different reward components
        """
        # Default weights for the reward components
        self.weights = weights or {
            # 'image_similarity': 0.4,    # Weight for image similarity reward
            'solution_space': 0.5,           # Weight for solution space reward
            'path_quality': 0.5,             # Weight for path quality reward
            'gt_cell_coverage': 0.0,         # Weight for GT cell coverage reward (第三个reward)
            'background_violation': 0.0      # Weight for background violation reward (第四个reward)
        }

    def _convert_to_tensor(self, image_input: Union[str, Image.Image, torch.Tensor], target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """
        Convert various image formats to tensor.
        
        Args:
            image_input: Image in various formats (str path, PIL Image, or tensor)
            target_size: Target size for resizing (height, width)
            
        Returns:
            Image tensor of shape (C, H, W)
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return torch.zeros(3, target_size[0], target_size[1])
            return load_image_as_tensor(image_input, target_size)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input)
            #return pil_image_to_tensor(image_input, target_size)
        elif isinstance(image_input, torch.Tensor):
            # Direct tensor input - ensure correct shape and size
            if image_input.dim() == 4:  # (B, C, H, W)
                tensor = image_input.squeeze(0)  # Remove batch dimension
            elif image_input.dim() == 3:  # (C, H, W)
                tensor = image_input
            else:
                #print(f"Unsupported tensor shape: {image_input.shape}")
                return torch.zeros(3, target_size[0], target_size[1])
            
            # Resize if needed
            if tensor.shape[1:] != target_size:
                tensor = F.interpolate(
                    tensor.unsqueeze(0), 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            return tensor
        else:
            #print(f"Unsupported image type: {type(image_input)}")
            return torch.zeros(3, target_size[0], target_size[1])

    def get_reference_images(self, prompt: str, metadata: Dict) -> Tuple[Union[str, Image.Image, torch.Tensor], Union[str, Image.Image, torch.Tensor], Union[str, Image.Image, torch.Tensor], Union[str, Image.Image, torch.Tensor]]:
        """
        Get the reference maze, solution images, and solution mask from metadata.
        
        For parquet-based datasets, images are stored directly in metadata as PIL Images.

        Args:
            prompt: Text prompt used for generation
            metadata: Metadata containing PIL images (from parquet dataset)

        Returns:
            Tuple of (original_image, maze_image, solution_image, solution_mask) - PIL Images or None
        """
        # Get images directly from metadata (parquet format)
        ori_image = None
        solution_image = None
        maze_image = None
        solution_mask = None

        # Check for direct images in metadata (PIL or tensor)
        if 'original_img' in metadata and metadata['original_img'] is not None:
            ori_image = metadata['original_img']

        if 'sol_img' in metadata and metadata['sol_img'] is not None:
            solution_image = metadata['sol_img']

        if 'm_original_img' in metadata and metadata['m_original_img'] is not None:
            maze_image = metadata['m_original_img']

        # Check for solution space mask
        if 'mask_img' in metadata and metadata['mask_img'] is not None:
            solution_mask = metadata['mask_img']

        # Return images from metadata (should always be available for parquet datasets)
        return ori_image, maze_image, solution_image, solution_mask

    def compute_solution_space_reward(self, generated_image: torch.Tensor,
                                        reference_maze: Union[str, Image.Image, torch.Tensor],
                                        reference_solution: Union[str, Image.Image, torch.Tensor],
                                        solution_mask: Union[str, Image.Image, torch.Tensor] = None) -> Tuple[float, float]:
            """
            第一个reward（修改版）：计算解空间内（Mask=1）和解空间外（Mask=0）的MSE。

            Args:
                generated_image: 模型输出的生成图像 (C, H, W)
                reference_maze: 原始图像
                reference_solution: (未使用)
                solution_mask: 解空间mask PNG（255=解空间，0=其他）HWC

            Returns:
                Tuple[float, float]: (mse_inside, mse_outside)
            """
            #print("========")
            if reference_maze is None or solution_mask is None:
                print("错误: reference_maze和solution_mask都必须提供")
                return 0.0, 0.0
            
            try:
                # 获取generated_image的空间尺寸
                if generated_image.dim() == 4:
                    generated_image = generated_image.squeeze(0)
                #print(reference_maze.shape)
                # target_h, target_w = reference_maze.shape[0], reference_maze.shape[1]
                generated_image = torch.tensor(np.array(generated_image))
                Image.fromarray(generated_image.numpy().transpose(1, 2, 0).astype(np.uint8)).save("gener.jpg") 
                #generated_image = generated_image.resize(target_h, target_w)
                # print("generated_image shape: ", generated_image.shape)
                # 1. 转换reference_maze为tensor
                reference_tensor =torch.tensor(np.array(reference_solution).transpose(2, 0, 1)) # self._convert_to_tensor(reference_maze, target_size=(target_h, target_w))
                Image.fromarray(reference_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)).save("reference.jpg") 
                # print("reference_tensor shape: ", reference_tensor.shape)
                #if generated_image.device != reference_tensor.device:
                 #   reference_tensor = reference_tensor.to(generated_image.device)
                #print(mask_tensor.shape)
                # 2. 处理mask HWC
                mask_tensor = torch.tensor(np.array(solution_mask).transpose(2,0,1))#self._convert_to_tensor(solution_mask, target_size=(target_h, target_w))
                Image.fromarray(mask_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)).save("mask.jpg") 
                # print("mask_tensor shape: ", mask_tensor.shape)
                #if generated_image.device != mask_tensor.device:
                #    mask_tensor = mask_tensor.to(generated_image.device)

                solution_space_mask = extract_solution_space_from_mask(mask_tensor)
                solution_space_tensor = torch.from_numpy(solution_space_mask.astype(np.float32)).to(generated_image.device)
                
                # 确保mask维度正确 (C, H, W)
                #if solution_space_tensor.dim() == 2:
                 #   solution_space_tensor = solution_space_tensor.unsqueeze(0).repeat(generated_image.shape[0], 1, 1)
                #elif solution_space_tensor.shape[0] == 1:
                 #   solution_space_tensor = solution_space_tensor.repeat(generated_image.shape[0], 1, 1)

                # 3. 计算均方差 (Squared Difference)
                gen = generated_image.float()
                ref = reference_tensor.float()
                diff_sq = (generated_image / 255.0 - reference_tensor / 255.0) ** 2 
                #print(diff_sq)
                # 4. 计算解空间内的MSE (Mask == 1)
                # 添加 epsilon 防止除零
                mask_sum = torch.sum(solution_space_tensor)
                if mask_sum > 0:
                    mse_inside = torch.sum(diff_sq * solution_space_tensor) / mask_sum
                else:
                    mse_inside = torch.tensor(0.0, device=generated_image.device)

                # 5. 计算解空间外的MSE (Mask == 0, 即背景墙壁)
                inverse_mask = 1.0 - solution_space_tensor
                inverse_mask_sum = torch.sum(inverse_mask)
                
                if inverse_mask_sum > 0:
                    mse_outside = torch.sum(diff_sq * inverse_mask) / inverse_mask_sum
                else:
                    mse_outside = torch.tensor(0.0, device=generated_image.device)

                # print(f"  第一个reward (MSE): Inside={mse_inside.item():.6f}, Outside={mse_outside.item():.6f}")

                return mse_inside.item(), mse_outside.item()

            except Exception as e:
                print(f"Error computing MSE reward: {e}")
                #import traceback
                raise e
                #traceback.print_exc()
               # return 0.0, 0.0


    def compute_path_quality_reward(self, generated_image: torch.Tensor,
                                  reference_solution: Union[str, Image.Image, torch.Tensor],
                                  reference_maze: Union[str, Image.Image, torch.Tensor] = None,
                                  solution_mask: Union[str, Image.Image, torch.Tensor] = None) -> float:
        """
        第二个reward：将mask和generated img转成二进制后相乘，在白色区域内处理路径质量。

        Args:
            generated_image: 模型输出的生成图像 (C, H, W)
            reference_solution: 参考解图像，用于提取起终点位置
            reference_maze: 原始图像，用于验证路径不与黑色区域交叉（可选）
            solution_mask: 解空间mask PNG（255=解空间，0=其他），必须提供

        Returns:
            Path validity reward (0-1, higher is better)
        """
        return 0.0
        if solution_mask is None:
            print("错误: solution_mask必须提供")
            return 0.0

        try:
            # 1. 转换generated_image为二进制图像
            generated_tensor_resized = torch.nn.functional.interpolate(
                generated_image.unsqueeze(0), size=(256, 256), mode='bilinear'
            ).squeeze(0)

            # 转换为灰度并转成numpy
            if generated_tensor_resized.shape[0] == 3:  # RGB
                generated_gray = torch.mean(generated_tensor_resized, dim=0)
            else:
                generated_gray = generated_tensor_resized[0]

            # 转换为float32以支持bfloat16等数据类型
            generated_np = generated_gray.float().cpu().numpy()
            # 转换为二进制（白色=1，黑色=0）
            generated_binary = (generated_np > 0.5).astype(np.float32)

            # 2. 转换solution_mask为二进制图像
            mask_tensor = self._convert_to_tensor(solution_mask, target_size=(256, 256))
            # 转换为float32以支持bfloat16等数据类型
            mask_np = mask_tensor.float().cpu().numpy()

            # 如果是多通道，取第一个通道
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]
            elif len(mask_np.shape) == 2:
                mask_np = mask_np

            # 转换为二进制（白色=1，黑色=0）
            mask_binary = (mask_np > 0.5).astype(np.float32)

            # #print(f"第二个reward：将mask和generated_img转成二进制后相乘")
            # #print(f"第二个reward调试信息:")
            # #print(f"  generated_image二进制白色像素数: {np.sum(generated_binary)}")
            # #print(f"  solution_mask二进制白色像素数: {np.sum(mask_binary)}")

            # 3. 二进制相乘：只保留在解空间（白色区域）内的generated图像部分
            tmp_binary = generated_binary * mask_binary
            result_binary = (tmp_binary>0.5) ^ (mask_binary>0.5)

            # #print(f"  generated_image和solution_mask相乘后白色像素数: {np.sum(result_binary)}")

            # 4. 转换回图片格式（0或1 -> 0或255）
            result_img_array = (result_binary * 255).astype(np.uint8)
            result_img = Image.fromarray(result_img_array)

            # 保存调试图像（基本的二进制处理结果）
            # try:
            #     os.makedirs("test_images", exist_ok=True)

            #     # 保存二进制转换结果
            #     # generated_binary_img = Image.fromarray((generated_binary * 255).astype(np.uint8))
            #     # generated_binary_img.save("test_images/generated_binary.png")

            #     # mask_binary_img = Image.fromarray((mask_binary * 255).astype(np.uint8))
            #     # mask_binary_img.save("test_images/mask_binary.png")

            #     result_img.save("test_images/result_binary_multiply.png")

            #     print("已保存二进制处理调试图像:")
            #     # print("  - test_images/generated_binary.png: generated图像的二进制版本")
            #     # print("  - test_images/mask_binary.png: mask的二进制版本")
            #     print("  - test_images/result_binary_multiply.png: 相乘结果")

            # except Exception as e:
            #     #print(f"保存基本调试图像时出错: {e}")

            # 5. 在白色区域内做处理
            # 步骤1：提取generated image和reference solution中的起点和终点位置，并检测重合度
            end_iou = 0
            try:
                _, gen_start_mask, gen_end_mask = extract_red_markers(generated_tensor_resized)
                _, ref_start_mask, ref_end_mask = extract_red_markers(reference_solution)

                # 检查generated image中是否找到起终点
                if np.sum(gen_start_mask) == 0 or np.sum(gen_end_mask) == 0:
                    print("  ✗ 未能在generated image中找到起点或终点红色标记；")
                    # return 0.0

                # 检查reference solution中是否找到起终点
                if np.sum(ref_start_mask) == 0 or np.sum(ref_end_mask) == 0:
                    print("  ✗ 未能在reference solution中找到起点或终点红色标记；")
                    return 0.0

                # 计算起点和终点的重合度（IoU）
                start_overlap = compute_iou(gen_start_mask, ref_start_mask)
                end_overlap = compute_iou(gen_end_mask, ref_end_mask)

                # #print(f"  起点重合度: {start_overlap:.4f}, 终点重合度: {end_overlap:.4f}")

                # 设置重合度阈值，如果重合度太低则认为位置不匹配
                # overlap_threshold = 0.1  # 可以根据需要调整
                end_iou = start_overlap*0.5+end_overlap*0.5
                # if start_overlap < overlap_threshold or end_overlap < overlap_threshold:
                    # #print(f"  ✗ 起终点位置重合度不足（阈值: {overlap_threshold}）")
                    # return 0.0

            except Exception as e:
                #print(f"  提取起终点或计算重合度时出错: {e}")
                return 0.0

            # 步骤2：用BFS判断路径连通性（使用generated image的起终点）
            connectivity_score = 0.0
            try:
                # 首先进行膨胀处理
                from scipy.ndimage import binary_dilation
                structure = np.ones((3, 3), dtype=bool)
                expanded_start_mask = binary_dilation(gen_start_mask.astype(bool), structure=structure, iterations=2).astype(np.float32)
                expanded_end_mask = binary_dilation(gen_end_mask.astype(bool), structure=structure, iterations=2).astype(np.float32)

                # #print(f"  膨胀处理: 起点{np.sum(gen_start_mask)}→{np.sum(expanded_start_mask)}, 终点{np.sum(gen_end_mask)}→{np.sum(expanded_end_mask)}")

                # # 关键修复：将膨胀后的起终点区域添加到路径中
                # #print(f"  修复膨胀问题：将膨胀区域添加到实际路径图像中")
                result_binary_with_expanded = (1-result_binary).copy()

                # 将膨胀后的起终点区域强制设为黑色路径（0值）
                result_binary_with_expanded[expanded_start_mask > 0] = 0  # 黑色路径
                result_binary_with_expanded[expanded_end_mask > 0] = 0    # 黑色路径

                # 在result_binary中，白色=1，黑色=0
                # 我们需要检查黑色路径（0值）的连通性
                inverted_result = 1-result_binary_with_expanded  # 反转：黑色路径变为1，白色背景变为0

                if np.sum(inverted_result) > 0:  # 如果有黑色路径
                    # #print(f"  反转后路径像素数: {np.sum(inverted_result)}")
                    # #print(f"  原始result_binary中白色像素数: {np.sum(result_binary)}")
                    # #print(f"  原始result_binary中黑色像素数: {np.sum(1 - result_binary)}")

                    # 保存修复后的路径图像
                    # try:
                    #     # 保存原始result_binary
                    #     original_result_img = Image.fromarray((result_binary * 255).astype(np.uint8))
                    #     original_result_img.save("test_images/original_result_binary.png")

                    #     # 保存修复后的result_binary_with_expanded
                    #     expanded_result_img = Image.fromarray((result_binary_with_expanded * 255).astype(np.uint8))
                    #     expanded_result_img.save("test_images/expanded_result_binary.png")

                    #     # 保存反转后用于BFS的图像
                    #     # inverted_img = Image.fromarray((inverted_result * 255).astype(np.uint8))
                    #     # inverted_img.save("test_images/inverted_result_for_bfs.png")

                    #     #print(f"    已保存修复后的路径图像:")
                    #     #print(f"      - test_images/original_result_binary.png: 原始路径")
                    #     #print(f"      - test_images/expanded_result_binary.png: 添加膨胀后的路径")
                    #     # #print(f"      - test_images/inverted_result_for_bfs.png: BFS使用的路径图像")
                    # except Exception as e:
                    #     #print(f"    保存修复后路径图像失败: {e}")

                    # 转换reference_solution为tensor用于提取起终点
                    reference_solution_tensor = self._convert_to_tensor(reference_solution, target_size=(256, 256))
                    if generated_image.device != reference_solution_tensor.device:
                        reference_solution_tensor = reference_solution_tensor.to(generated_image.device)

                    # 使用新的连通性检查策略：随机点BFS找端点
                    connectivity_score = self._check_connectivity_with_random_point(
                        inverted_result, expanded_start_mask, expanded_end_mask, reference_solution_tensor
                    )
                    # #print(f"  第二个reward路径连通性分数: {connectivity_score:.4f}")

                    # 如果路径不连通，直接返回0
                    if connectivity_score == 0.0:
                        print("  路径不连通")
                        # return 0.0
                else:
                    print("  未发现黑色路径,0")
                    return 0.0

            except Exception as e:
                #print(f"  检查连通性时出错: {e}")
                return 0.0

            # 6. 计算最终reward
            # 连通性已确认为非0（否则已经返回）
            final_reward = connectivity_score

            #print(f"  最终第二个reward: {final_reward:.4f}")

            return max(0.0, min(1.0, final_reward))

        except Exception as e:
            #print(f"Error computing path validity reward: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _check_path_connectivity_binary(self, path_binary: np.ndarray,
                                      start_mask: np.ndarray,
                                      end_mask: np.ndarray) -> float:
        """
        检查从起点到终点的路径连通性（使用BFS算法）

        Args:
            path_binary: 二进制路径图像（1=路径，0=背景）
            start_mask: 起点区域mask
            end_mask: 终点区域mask

        Returns:
            连通性分数 (0-1)
        """
        try:
            from collections import deque

            # #print(f"    BFS连通性检查调试信息:")
            # #print(f"      路径图像尺寸: {path_binary.shape}")
            # #print(f"      路径像素总数: {np.sum(path_binary > 0)}")
            # #print(f"      起点mask像素数: {np.sum(start_mask > 0)}")
            # #print(f"      终点mask像素数: {np.sum(end_mask > 0)}")

            # 获取起点和终点坐标
            start_coords = np.where(start_mask > 0)
            end_coords = np.where(end_mask > 0)

            if len(start_coords[0]) == 0 or len(end_coords[0]) == 0:
                # #print(f"      ✗ 起点或终点区域为空")
                return 0.0

            # 选择起点区域的中心点作为起始位置
            start_y, start_x = int(np.mean(start_coords[0])), int(np.mean(start_coords[1]))
            # #print(f"      起点中心坐标: ({start_y}, {start_x})")
            # #print(f"      起点位置路径值: {path_binary[start_y, start_x]}")

            # 检查起点是否在路径上
            if path_binary[start_y, start_x] == 0:
                # #print(f"      ⚠ 起点中心不在路径上，寻找最近的路径点")
                # 在起点区域内寻找路径点
                start_region_path = start_mask * path_binary
                if np.sum(start_region_path) == 0:
                    # #print(f"      ✗ 起点区域内没有路径像素")
                    return 0.0
                else:
                    path_coords_in_start = np.where(start_region_path > 0)
                    start_y, start_x = path_coords_in_start[0][0], path_coords_in_start[1][0]
                    # #print(f"      使用起点区域内的路径点: ({start_y}, {start_x})")

            # 创建终点区域的set用于快速查找，但只包含路径点
            end_region_path = end_mask * path_binary
            end_path_coords = np.where(end_region_path > 0)
            end_points = set(zip(end_path_coords[0], end_path_coords[1]))
            # #print(f"      终点区域包含 {len(end_points)} 个路径像素点（原始终点区域{np.sum(end_mask)}个像素）")

            # 检查终点区域是否有路径点
            end_path_pixels = np.sum(end_region_path)
            # #print(f"      终点区域内路径像素数: {end_path_pixels}")

            if end_path_pixels == 0:
                # #print(f"      ✗ 终点区域内没有路径像素")
                return 0.0

            # 新策略：多起点并行BFS搜索（解决路径断裂问题）
            # 1. 从膨胀后起点区域内的所有路径点开始搜索
            start_region_path = start_mask * path_binary
            start_path_coords = np.where(start_region_path > 0)
            all_start_points = list(zip(start_path_coords[0], start_path_coords[1]))

            # #print(f"      膨胀后起点区域内路径点总数: {len(all_start_points)}")
            # #print(f"      膨胀后起点区域覆盖范围: y=[{np.min(start_path_coords[0])}, {np.max(start_path_coords[0])}], x=[{np.min(start_path_coords[1])}, {np.max(start_path_coords[1])}]")

            # 新策略：如果单起点BFS覆盖率太低，使用多起点策略
            # 分批启动多个起点进行BFS，提高覆盖率
            max_start_points = min(8, len(all_start_points))  # 最多使用10个起点
            # #print(f"      将使用多起点策略，最多{max_start_points}个起点")

            # #print(f"      实现多起点BFS策略")

            # 多起点BFS策略：从起点区域的多个路径点同时开始搜索
            visited = set()
            queue = deque()
            directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]  # 8方向
            h, w = path_binary.shape

            # 选择分散的多个起点
            selected_start_points = []

            # 策略1：选择高连通性起点
            point_quality = []
            for sy, sx in all_start_points:
                neighbor_count = 0
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < h and 0 <= nx < w and path_binary[ny, nx] > 0:
                        neighbor_count += 1
                point_quality.append(((sy, sx), neighbor_count))

            # 按连通性排序，选择前8个
            point_quality.sort(key=lambda x: x[1], reverse=True)
            selected_start_points = [point for point, _ in point_quality[:8]]

            # #print(f"      选择{len(selected_start_points)}个高连通性起点进行多起点BFS")

            # 将所有起点加入队列和visited集合
            for start_point in selected_start_points:
                queue.append(start_point)
                visited.add(start_point)

            # #print(f"      多起点BFS初始队列: {len(queue)}个起点，覆盖范围更广")

            step_count = 0
            max_steps = min(h * w, 200000)  # 增加步数限制

            while queue and step_count < max_steps:
                step_count += 1
                y, x = queue.popleft()

                # 检查是否到达终点区域
                if (y, x) in end_points:
                    #print(f"      ✓ 多起点BFS找到路径! 步数: {step_count}")
                    #print(f"      到达终点: ({y}, {x})")
                    # 成功时也保存一次可视化结果
                    # try:
                    #     visited_mask = np.zeros_like(path_binary)
                    #     for vy, vx in visited:
                    #         visited_mask[vy, vx] = 1

                    #     debug_img = np.zeros((h, w, 3), dtype=np.uint8)
                    #     debug_img[path_binary > 0] = [255, 255, 255]
                    #     debug_img[visited_mask > 0] = [0, 255, 0]

                    #     os.makedirs("test_images", exist_ok=True)
                    #     Image.fromarray(debug_img).save("test_images/multi_start_bfs_result.png")
                    #     #print(f"      已保存多起点BFS结果（成功版）: test_images/multi_start_bfs_result.png")
                    # except Exception as e:
                    #     #print(f"      保存BFS成功结果失败: {e}")
                    return 1.0

                # 搜索8个方向的邻居
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and
                        (ny, nx) not in visited and
                        path_binary[ny, nx] > 0):
                        visited.add((ny, nx))
                        queue.append((ny, nx))

                # 输出进度
                if step_count % 500 == 0:
                    coverage = len(visited) / max(1, np.sum(path_binary > 0))
                    # #print(f"      多起点BFS第{step_count}步: 队列{len(queue)}, 已访问{len(visited)}, 覆盖率{coverage:.1%}")

            # #print(f"      多起点BFS完成: 步数{step_count}, 覆盖率{len(visited)/max(1, np.sum(path_binary > 0)):.1%}")

        except Exception as e:
            #print(f"  BFS连通性检查出错: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


    def _check_connectivity_with_random_point(self, inverted_result: np.ndarray,
                                             expanded_start_mask: np.ndarray,
                                             expanded_end_mask: np.ndarray,
                                             reference_solution: torch.Tensor) -> float:
        """
        新的连通性检查策略：
        1. 从reference_solution中提取真实的起点和终点位置
        2. 在inverted_result中随机选择一个黑色点
        3. 从该点BFS两次，找到连通组件的两个端点
        4. 判断这两个端点是否分别位于起点和终点区域内

        Args:
            inverted_result: 反转后的路径图像（1=路径，0=背景）
            expanded_start_mask: 膨胀后的起点区域
            expanded_end_mask: 膨胀后的终点区域
            reference_solution: 参考解图像，用于提取真实起终点位置

        Returns:
            连通性分数 (0-1)
        """
        try:
            from collections import deque
            import random

            h, w = inverted_result.shape
            directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

            # 0. 从reference_solution中提取真实的起点和终点位置
            # #print(f"    从reference_solution提取真实起终点位置")
            try:
            #     # 调试：检查reference_solution的状态
            #     #print(f"    reference_solution类型: {type(reference_solution)}")
                if isinstance(reference_solution, torch.Tensor):
            #         #print(f"    reference_solution tensor形状: {reference_solution.shape}")
            #         #print(f"    reference_solution tensor设备: {reference_solution.device}")
            #         #print(f"    reference_solution tensor数据类型: {reference_solution.dtype}")
                    # 移动到CPU并转换为float32以便检查
                    ref_sol_cpu = reference_solution.detach().float().cpu()
                    # #print(f"    reference_solution值范围: [{ref_sol_cpu.min():.3f}, {ref_sol_cpu.max():.3f}]")
                    # #print(f"    reference_solution是否全零: {(ref_sol_cpu == 0).all().item()}")
                    
                    # 如果值范围不在[0,1]，需要归一化
                    if ref_sol_cpu.max() > 1.0:
                        #print(f"    警告: reference_solution值范围超出[0,1]，需要归一化")
                        ref_sol_cpu = ref_sol_cpu / 255.0
                    
                    # 确保形状是(C, H, W)
                    if ref_sol_cpu.dim() == 4:
                        ref_sol_cpu = ref_sol_cpu.squeeze(0)
                    elif ref_sol_cpu.dim() == 2:
                        ref_sol_cpu = ref_sol_cpu.unsqueeze(0)
                    
                    # 确保是3通道
                    if ref_sol_cpu.shape[0] == 1:
                        ref_sol_cpu = ref_sol_cpu.repeat(3, 1, 1)
                    elif ref_sol_cpu.shape[0] != 3:
                        print(f"    警告: reference_solution通道数异常: {ref_sol_cpu.shape}")
                    
                    reference_solution_for_extract = ref_sol_cpu
                else:
                    # 如果不是tensor，使用_convert_to_tensor转换
                    reference_solution_for_extract = self._convert_to_tensor(reference_solution, target_size=(256, 256))
                    # #print(f"    使用_convert_to_tensor转换后的形状: {reference_solution_for_extract.shape}")
                

                _, ref_start_mask, ref_end_mask = extract_red_markers(reference_solution_for_extract)
                from scipy.ndimage import binary_dilation
                structure = np.ones((3, 3), dtype=bool)
                expanded_ref_start_mask = binary_dilation(ref_start_mask.astype(bool), structure=structure, iterations=5).astype(np.float32)
                expanded_ref_end_mask = binary_dilation(ref_end_mask.astype(bool), structure=structure, iterations=5).astype(np.float32)

                ref_start_pixels = np.sum(expanded_ref_start_mask)
                ref_end_pixels = np.sum(expanded_ref_end_mask)
                # #print(f"    参考起点像素数: {ref_start_pixels}")
                # #print(f"    参考终点像素数: {ref_end_pixels}")

                if ref_start_pixels == 0 or ref_end_pixels == 0:
                    # #print(f"    ✗ 未在reference_solution中找到起点或终点")
                    return 0.0
            except Exception as e:
                # #print(f"    ✗ 提取reference_solution起终点失败: {e}")
                return 0.0

            # 1. 找到所有黑色路径点
            path_points = np.where(inverted_result > 0)
            if len(path_points[0]) == 0:
                # #print(f"    ✗ 没有找到路径点")
                return 0.0

            all_path_coords = list(zip(path_points[0], path_points[1]))
#            #print(f"    总路径点数: {len(all_path_coords)}")

            # 2. 随机选择一个路径点作为起始点
            random_start = random.choice(all_path_coords)
            # #print(f"    随机选择起始点: {random_start}")

            # 3. 通过两次BFS找到连通组件的两个端点
            # 第一次BFS：从随机点开始，找到最远的点作为第一个端点
            def bfs_find_farthest(start_point, path_mask):
                visited = set()
                queue = deque([(start_point, 0)])  # (point, distance)
                visited.add(start_point)
                farthest_point = start_point
                max_distance = 0

                while queue:
                    (y, x), dist = queue.popleft()
                    if dist > max_distance:
                        max_distance = dist
                        farthest_point = (y, x)

                    for dy, dx in directions:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and
                            (ny, nx) not in visited and
                            path_mask[ny, nx] > 0):
                            visited.add((ny, nx))
                            queue.append(((ny, nx), dist + 1))

                return farthest_point, max_distance

            # 第一次BFS：从随机点找到第一个端点
            endpoint1, dist1 = bfs_find_farthest(random_start, inverted_result)
            # #print(f"    第一次BFS: 从{random_start}找到端点1: {endpoint1}, 距离: {dist1}")

            # 第二次BFS：从第一个端点找到第二个端点（连通组件的另一端）
            endpoint2, dist2 = bfs_find_farthest(endpoint1, inverted_result)
            # #print(f"    第二次BFS: 从{endpoint1}找到端点2: {endpoint2}, 距离: {dist2}")

            if endpoint1 == endpoint2:
                # #print(f"    ✗ 两个端点相同，连通组件可能太小")
                return 0.0

            # 4. 检查两个端点是否分别位于参考起点和终点区域
            ep1_in_ref_start =expanded_ref_start_mask[endpoint1[0], endpoint1[1]] > 0
            ep1_in_ref_end = expanded_ref_end_mask[endpoint1[0], endpoint1[1]] > 0
            ep2_in_ref_start = expanded_ref_start_mask[endpoint2[0], endpoint2[1]] > 0
            ep2_in_ref_end = expanded_ref_end_mask[endpoint2[0], endpoint2[1]] > 0

            # #print(f"    端点1在参考起点区域: {ep1_in_ref_start}, 在参考终点区域: {ep1_in_ref_end}")
            # #print(f"    端点2在参考起点区域: {ep2_in_ref_start}, 在参考终点区域: {ep2_in_ref_end}")

            # 5. 判断是否有一个端点在起点，另一个在终点（顺序可颠倒）
            if (ep1_in_ref_start and ep2_in_ref_end) or (ep1_in_ref_end and ep2_in_ref_start):
                # #print(f"    ✓ 端点匹配成功，路径连通参考起终点，返回1")
                return 1.0

            # 6. 如果端点不匹配，使用指数衰减距离策略
            # #print(f"    端点不匹配，使用距离衰减策略")

            # 计算端点到参考起终点区域的最短距离
            ref_start_coords = np.where(expanded_ref_start_mask > 0)
            ref_end_coords = np.where(expanded_ref_end_mask > 0)

            if len(ref_start_coords[0]) == 0 or len(ref_end_coords[0]) == 0:
                return 0.0

            from scipy.spatial.distance import cdist

            # 计算两个端点到参考起终点区域的最短距离
            endpoints = np.array([endpoint1, endpoint2])
            ref_start_points = np.array(list(zip(ref_start_coords[0], ref_start_coords[1])))
            ref_end_points = np.array(list(zip(ref_end_coords[0], ref_end_coords[1])))

            # 端点到参考起点区域的距离
            dist_ep1_to_start = np.min(cdist([endpoint1], ref_start_points))
            dist_ep2_to_end = np.min(cdist([endpoint2], ref_end_points))
            # 端点到参考终点区域的距离
            dist_ep1_to_end = np.min(cdist([endpoint1], ref_end_points))
            dist_ep2_to_start = np.min(cdist([endpoint2], ref_start_points))

            # 距离分配：离起点最近端点算起点距离，离终点最近端点算终点距离
            min_start_dist = min(dist_ep1_to_start, dist_ep2_to_start)
            min_end_dist = min(dist_ep1_to_end, dist_ep2_to_end)

            # 距离得分各自指数衰减，最后加权平均
            start_score = max(0.0, np.exp(-min_start_dist/10))
            end_score = max(0.0, np.exp(-min_end_dist/10))
            distance_score = 0.5 * start_score + 0.5 * end_score

            # #print(f"    端点到起点最短距离: {min_start_dist:.2f}, 到终点最短距离: {min_end_dist:.2f}")
            # #print(f"    第二个reward: avg={distance_score:.3f}")
            return distance_score

        except Exception as e:
            # #print(f"    随机点连通性检查出错: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _decode_cell_map(self, cell_map_input: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
        """
        解码cell map图像，将编码的RGB值转换为cell ID。
        
        与 visualize_cell_map.py 中的解码方式一致。
        JavaScript保存时使用BGR顺序写入buffer，但PNG文件格式是RGB。
        OpenCV读取：BGR格式 -> id = R | (G << 8) | (B << 16)，其中 R=[:,:,2], G=[:,:,1], B=[:,:,0]
        PIL读取：RGB格式，但需要按OpenCV的方式解码（因为原始数据是BGR顺序）

        Args:
            cell_map_input: Cell map图像 (PIL Image, numpy array HWC, 或 torch Tensor CHW)

        Returns:
            Cell ID数组 (H, W)，每个像素对应一个cell ID
        """
        try:
            # 转换为numpy数组 HWC格式
            if isinstance(cell_map_input, Image.Image):
                cell_map_np = np.array(cell_map_input)  # HWC, RGB格式, uint8 [0, 255]
            elif isinstance(cell_map_input, torch.Tensor):
                if cell_map_input.device != torch.device('cpu'):
                    cell_map_np = cell_map_input.detach().float().cpu().numpy()
                else:
                    cell_map_np = cell_map_input.float().numpy()
                # 转换为HWC格式
                if cell_map_np.shape[0] == 3:
                    cell_map_np = cell_map_np.transpose(1, 2, 0)  # CHW -> HWC
                # 转换到[0, 255]范围
                cell_map_np = np.clip(cell_map_np * 255, 0, 255).astype(np.uint8)
            else:
                cell_map_np = np.array(cell_map_input)
                if len(cell_map_np.shape) == 3 and cell_map_np.shape[0] == 3:
                    cell_map_np = cell_map_np.transpose(1, 2, 0)  # CHW -> HWC
                if cell_map_np.dtype != np.uint8 or cell_map_np.max() <= 1.0:
                    cell_map_np = np.clip(cell_map_np * 255, 0, 255).astype(np.uint8)

            # 解码方式说明：
            # JavaScript编码：r = (id >> 16) & 0xFF, g = (id >> 8) & 0xFF, b = id & 0xFF
            # JavaScript保存时使用BGR顺序写入buffer，但Sharp保存PNG时按RGB格式保存到文件
            # PNG文件中：R通道=r, G通道=g, B通道=b
            # OpenCV读取（cv2.imread）：转换为BGR格式 -> [:,:,0]=b, [:,:,1]=g, [:,:,2]=r
            #   解码：id = r | (g << 8) | (b << 16) = [:,:,2] | ([:,:,1] << 8) | ([:,:,0] << 16)
            # PIL读取：保持RGB格式 -> [:,:,0]=r, [:,:,1]=g, [:,:,2]=b
            #   解码：id = r | (g << 8) | (b << 16) = [:,:,0] | ([:,:,1] << 8) | ([:,:,2] << 16)
            r_channel = cell_map_np[:, :, 0].astype(np.uint32)  # R通道 = r值
            g_channel = cell_map_np[:, :, 1].astype(np.uint32)  # G通道 = g值
            b_channel = cell_map_np[:, :, 2].astype(np.uint32)  # B通道 = b值
            
            cell_ids = r_channel | (g_channel << 8) | (b_channel << 16)

            return cell_ids

        except Exception as e:
            #print(f"    解码cell map失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((256, 256), dtype=np.int32)

    def compute_gt_cell_coverage_reward(self, generated_image: torch.Tensor,
                                       cell_map: Union[str, Image.Image, torch.Tensor],
                                       metadata_json: str,
                                       solution_mask: Union[str, Image.Image, torch.Tensor] = None,
                                       reference_solution: Union[str, Image.Image, torch.Tensor] = None,
                                       sample_index: int = 0) -> float:
        """
        第三个reward：使用BFS遍历生成的路径，计算经过的GT cell数量占比。

        Args:
            generated_image: 模型输出的生成图像 (C, H, W)
            cell_map: 格子分割图（BGR格式），每个像素的RGB值编码了cell ID
            metadata_json: 元数据JSON字符串，包含path_cell_ids（GT路径经过的cell列表）
            solution_mask: 解空间mask PNG（255=解空间，0=其他），必须提供
            reference_solution: 参考解图像，用于提取起终点位置（可选）

        Returns:
            GT cell覆盖率 (0-1, higher is better)
        """
        if solution_mask is None:
            print("  错误: solution_mask必须提供")
            return 0.0

        try:
            import json

            # 1. 解析metadata获取GT path cell IDs
            if isinstance(metadata_json, str):
                metadata = json.loads(metadata_json)
            else:
                metadata = metadata_json

            if 'path_cell_ids' not in metadata or len(metadata['path_cell_ids']) == 0:
                print("  警告: metadata中没有path_cell_ids")
                return 0.0

            gt_cell_ids = set(metadata['path_cell_ids'])
            gt_cell_count = len(gt_cell_ids)
            print(f"  GT路径经过的cell数量: {gt_cell_count}")
            print(f"  GT cell IDs: {sorted(gt_cell_ids)}")  # 只显示前10个

            # 2. 解码cell_map
            # cell_ids_array = cell_map_tensor = self._convert_to_tensor(cell_map)
            if IS_ISCIRCLE: 
                cell_ids_array = self._decode_cell_map(cell_map)
            else:
                cell_ids_array = cell_map_tensor = self._convert_to_tensor(cell_map)
            
            unique_cells = np.unique(cell_ids_array)
            
            solution_mask = torch.tensor(np.array(solution_mask).transpose(2, 0, 1)) #HWC-CHW
            
            Image.fromarray(solution_mask.numpy().transpose(1,2, 0).astype(np.uint8)).save("sl_.jpg")  #CHW-HWC
            
            result_img_tensor = (generated_image * ((solution_mask > 0).float())) # CHW
            # Image.fromarray(generated_image.numpy().transpose(2,1, 0).astype(np.uint8)).save("gener.jpg") 
            result_img = result_img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)

            Image.fromarray(result_img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)).save("bg_.jpg")

            
            # 5. 使用extract_blue_path提取蓝色路径
            #print(result_img_tensor.shape)
            generated_tensor_resized = result_img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8) # HWC
            blue_path_mask = extract_blue_path(generated_tensor_resized)

            Image.fromarray(np.stack([blue_path_mask * 255] * 3, axis=2).astype(np.uint8)).save("pt_.jpg")

            kernel = np.ones((3, 3), dtype=np.uint8)
            image = cv2.erode((blue_path_mask * 255).astype(np.uint8), kernel, iterations=1)
            # if ((image!=0).sum())<500:
            #     return -1.0
            # Image.fromarray(np.stack([image] * 3, axis=2).astype(np.uint8)).save("pt_erode.jpg")

            image = cv2.dilate(image, kernel, iterations=1)
            Image.fromarray(np.stack([image] * 3, axis=2).astype(np.uint8)).save("pt_restore.jpg")

            path_binary = mask_binary = blue_path_mask = (image > 0).astype(np.uint8)


#            # 方法1：提取蓝色路径（模型可能生成蓝色路径）
#            blue_path_mask = extract_blue_path(generated_tensor_resized)

            # 转换solution_mask为二进制图像
            mask_tensor = solution_mask #self._convert_to_tensor(solution_mask, target_size=(256, 256))
            mask_np = mask_tensor.float().cpu().numpy()

            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]

#            mask_binary = (mask_np > 0.5).astype(np.float32)
#
#            # 只保留在解空间内的蓝色路径
#            path_binary = blue_path_mask * mask_binary
#
#            # #print(f"  路径像素总数: {np.sum(path_binary)}")

            if np.sum(path_binary) == 0:
                print("  警告: 没有检测到路径像素")
                return 0.0

            # 获取路径尺寸和所有路径点（在try外面定义，以便fallback使用）
            h, w = path_binary.shape
            path_points = np.where(path_binary > 0)

            if len(path_points[0]) == 0:
                print("  警告: 没有路径点")
                return 0.0

            all_path_coords = set(zip(path_points[0], path_points[1]))
            visited = all_path_coords

            # #print(f"  BFS遍历的路径像素数: {len(visited)}")

            # 5. 提取BFS遍历路径经过的cell IDs
            if len(visited) == 0:
                print("  警告: BFS未找到路径像素")
                return 0.0

            # 从visited中提取所有坐标
            visited_coords = list(visited)
            visited_y = [coord[0] for coord in visited_coords]
            visited_x = [coord[1] for coord in visited_coords]

            # 从cell_map中提取这些像素对应的cell ID
            # print(f"cell_ids_array: {cell_ids_array}")
            path_cell_ids_array = cell_ids_array[path_points[0], path_points[1]]
            #path_cell_ids_array = cell_ids_array[visited_y, visited_x]
            path_cell_ids = set(path_cell_ids_array.flatten())

            # 移除0（背景ID）
            path_cell_ids.discard(0)

            print(f"  生成路径经过的cell数量: {len(path_cell_ids)}")
            print(f"  生成路径cell IDs前10个: {sorted(path_cell_ids)}")

            # 6. 计算与GT cell IDs的交集
            covered_gt_cells = gt_cell_ids.intersection(path_cell_ids)
            coverage_ratio = len(covered_gt_cells) / gt_cell_count if gt_cell_count > 0 else 0.0


            return max(0.0, min(1.0, coverage_ratio))

        except Exception as e:
            print(f"Error computing GT cell coverage reward: {e}")
            raise e
            import traceback
            traceback.print_exc()
            return 0.0


    def _pil_image_mask_multiply(self, sol_img: Union[str, Image.Image, torch.Tensor],
                                 mask_img: Union[str, Image.Image, torch.Tensor],
                                 original_img: Union[str, Image.Image, torch.Tensor],
                                 target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        参考download.py中的pil_image_mask_multiply实现
        使用sol_img, mask_img, original_img进行mask相乘操作
        
        Args:
            sol_img: 解图像（对应download.py中的img_path）
            mask_img: 掩码图像（对应download.py中的mask_path）
            original_img: 原始图像（对应download.py中的mask_boundary_path）
            target_size: 目标尺寸 (height, width)
            
        Returns:
            处理后的图像数组 (H, W, 3)，值范围[0, 255]
            结果中：解空间区域是全黑色，非解空间区域是白色/带路径的白色
        """
        # 1. 转换所有图像为PIL格式并调整尺寸
        if isinstance(sol_img, str):
            sol_pil = Image.open(sol_img).convert("RGB")
        elif isinstance(sol_img, Image.Image):
            sol_pil = sol_img.convert("RGB")
        elif isinstance(sol_img, torch.Tensor):
            # 转换为PIL
            sol_np = sol_img.detach().cpu().numpy()
            if sol_np.shape[0] == 3:  # CHW
                sol_np = sol_np.transpose(1, 2, 0)  # HWC
            sol_np = np.clip(sol_np * 255, 0, 255).astype(np.uint8)
            sol_pil = Image.fromarray(sol_np)
        else:
            raise ValueError(f"Unsupported sol_img type: {type(sol_img)}")
        
        sol_pil = sol_pil.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
        
        if isinstance(mask_img, str):
            mask_pil = Image.open(mask_img).convert("L")
        elif isinstance(mask_img, Image.Image):
            mask_pil = mask_img.convert("L")
        elif isinstance(mask_img, torch.Tensor):
            # 转换为PIL
            mask_np = mask_img.detach().cpu().numpy()
            if mask_np.shape[0] == 3:  # CHW
                mask_np = np.mean(mask_np, axis=0)  # 转为灰度
            elif mask_np.shape[0] == 1:  # CHW
                mask_np = mask_np[0]
            mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
        else:
            raise ValueError(f"Unsupported mask_img type: {type(mask_img)}")
        
        mask_pil = mask_pil.resize((target_size[1], target_size[0]))
        
        if isinstance(original_img, str):
            original_pil = Image.open(original_img).convert("L")
        elif isinstance(original_img, Image.Image):
            original_pil = original_img.convert("L")
        elif isinstance(original_img, torch.Tensor):
            # 转换为PIL
            orig_np = original_img.detach().cpu().numpy()
            if orig_np.shape[0] == 3:  # CHW
                orig_np = np.mean(orig_np, axis=0)  # 转为灰度
            elif orig_np.shape[0] == 1:  # CHW
                orig_np = orig_np[0]
            orig_np = np.clip(orig_np * 255, 0, 255).astype(np.uint8)
            original_pil = Image.fromarray(orig_np)
        else:
            raise ValueError(f"Unsupported original_img type: {type(original_img)}")
        
        original_pil = original_pil.resize((target_size[1], target_size[0]))
        
        # 2. 转换为NumPy数组
        sol_arr = np.array(sol_pil, dtype=np.float32)
        mask_arr = np.array(mask_pil, dtype=np.float32)
        original_arr = np.array(original_pil, dtype=np.float32)
        
        # 3. 扩展通道数 + 逐元素相乘
        mask_3c = np.repeat(mask_arr[:, :, np.newaxis], 3, axis=2)
        mask_3c = mask_3c / 255.0  # 归一化
        mask_3c = 1 - mask_3c  # 取反
        
        mask_boundary_3c = np.repeat(original_arr[:, :, np.newaxis], 3, axis=2)
        mask_boundary_3c = mask_boundary_3c / 255.0  # 归一化
        
        # 4. 执行mask相乘
        mask_whole = mask_3c * mask_boundary_3c
        result_arr = sol_arr * mask_whole
        
        # 6. 转换回uint8
        result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)
        
        return result_arr

    def compute_background_violation_reward(self, generated_image: torch.Tensor,
                                           cell_map: Union[str, Image.Image, torch.Tensor],
                                           metadata_json: str,
                                           solution_mask: Union[str, Image.Image, torch.Tensor] = None,
                                           sol_img: Union[str, Image.Image, torch.Tensor] = None,
                                           original_img: Union[str, Image.Image, torch.Tensor] = None,
                                           sample_index: int = 0) -> float:
        """
        第四个reward：使用generated_image, solution_mask, original_img进行mask相乘，计算路径在非解空间区域格子的占比。

        步骤：
        1. 使用pil_image_mask_multiply逻辑处理generated_image, solution_mask, original_img
        2. 在结果图像中，解空间区域是全黑色，非解空间区域是白色/带路径的白色
        3. 提取非解空间区域的路径（白色部分）
        4. 匹配到对应的cell map id
        5. 计算路径在非解空间区域格子的占比

        Args:
            generated_image: 模型输出的生成图像 (C, H, W)，用于mask相乘
            cell_map: 格子分割图（BGR格式），每个像素的RGB值编码了cell ID CHW
            metadata_json: 元数据JSON字符串，包含path_cell_ids（GT路径经过的cell列表）
            solution_mask: 解空间mask PNG（255=解空间，0=其他），对应mask_img
            sol_img: 解图像（未使用，保留接口兼容性）
            original_img: 原始图像（对应download.py中的mask_boundary_path）
            sample_index: 样本索引，用于保存调试图像

        Returns:
            背景违规比例 (0-1, lower is better，所以会用-1权重)
        """
        if solution_mask is None or original_img is None:
            print("  错误: solution_mask和original_img都必须提供")
            return 1.0

        try:
            import json
            from collections import deque

            # 1. 解析metadata获取GT path cell IDs
            if isinstance(metadata_json, str):
                metadata = json.loads(metadata_json)
            else:
                metadata = metadata_json

            if 'path_cell_ids' not in metadata or len(metadata['path_cell_ids']) == 0:
                print("  警告: metadata中没有path_cell_ids")
                return 1.0

            gt_cell_ids = set(metadata['path_cell_ids'])
#            print(f"  第四个reward - GT路径cell数量: {len(gt_cell_ids)}")


            # 2. 解码cell_map
            # cell_ids_array = cell_map_tensor = self._convert_to_tensor(cell_map)
            # cell_map_tensor = torch.tensor(np.array(cell_map).transpose(2,1,0)) #CHW
            if IS_CIRCLE:
                cell_ids_array = self._decode_cell_map(cell_map)
            else:
                cell_ids_array = cell_map_tensor = self._convert_to_tensor(cell_map)
            # print("solution_mask shape before: ", solution_mask.size)
            solution_mask = torch.tensor(np.array(solution_mask).transpose(2, 0, 1))
            


            result_img_tensor = (generated_image * (1 - (solution_mask > 0).float()))

            result_img = result_img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)

            #Image.fromarray(result_img_tensor.numpy().transpose(1,2,0).astype(np.uint8)).save("bg_.jpg")

            
            # 5. 使用extract_blue_path提取蓝色路径
    #        print(result_img_tensor.shape)
            blue_path_mask = extract_blue_path(result_img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8))

            # Image.fromarray(np.stack([blue_path_mask * 255] * 3, axis=2).astype(np.uint8)).save("pt_.jpg")

            kernel = np.ones((3, 3), dtype=np.uint8)
            image = cv2.erode((blue_path_mask * 255).astype(np.uint8), kernel, iterations=1)

            #Image.fromarray(np.stack([image] * 3, axis=2).astype(np.uint8)).save("pt_erode.jpg")


            image = cv2.dilate(image, kernel, iterations=1)
            #Image.fromarray(np.stack([image] * 3, axis=2).astype(np.uint8)).save("pt_restore.jpg")

            blue_path_mask = (image > 0).astype(np.uint8)
            
            #np.set_printoptions(threshold=np.inf, linewidth=np.nan) 

            # # 6. 提取白色区域（全白区域）
            # result_gray = np.mean(result_img, axis=2)  # 转为灰度
            # white_mask = (result_gray > 200).astype(np.float32)  # 白色阈值
            
            # 7. 合并蓝色路径和白色区域
            # path_mask = ((blue_path_mask > 0) | (white_mask > 0)).astype(np.float32)
            violation_path = path_mask = blue_path_mask #.astype(np.float32)
            # 8. 获取解空间mask（用于确认非解空间区域）
            mask_tensor = solution_mask #self._convert_to_tensor(solution_mask, target_size=(256, 256))
            mask_np = mask_tensor.float().cpu().numpy()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]

#            solution_binary = mask_np > 0
#            background_binary = 1 - solution_binary  # 非解空间区域
#            
#            # 9. 提取非解空间区域的路径（蓝色路径或白色区域在背景区域）
#            violation_path = path_mask * background_binary

            #print(path_mask.sum())

            violation_pixels = np.sum(violation_path)
            
     #       print(f"  非解空间区域的路径像素数: {violation_pixels}")
            
            if violation_pixels == 0:
                print("  没有违规路径，返回0.0")
                return 0.0
            
            # 7. 使用BFS遍历违规路径，统计经过的格子
            h, w = violation_path.shape

            violation_coords = np.where(violation_path > 0)
            if len(violation_coords[0]) == 0:
                return 0.0
            
            visited_y = violation_coords[0]
            visited_x = violation_coords[1]
            # 修复：检查cell_ids_array的尺寸，并过滤超出边界的坐标
            cell_h, cell_w = cell_ids_array.shape[:2]
            
            # 过滤掉超出cell_ids_array边界的坐标
            valid_mask = (visited_y >= 0) & (visited_y < cell_h) & (visited_x >= 0) & (visited_x < cell_w)
            visited_y = visited_y[valid_mask]
            visited_x = visited_x[valid_mask]
            
            if len(visited_y) == 0:
                return 0.0
            #from IPython import embed; embed()
            
            violation_cell_ids_array = cell_ids_array[visited_y, visited_x]
            violation_cell_ids = set(violation_cell_ids_array.flatten())
            #print(violation_cell_ids)
            #print(set(cell_ids_array.flatten().tolist()))
            violation_cell_ids.discard(0)  # 移除墙壁（cell_id=0不是格子）

            
            # 只保留真正的背景违规格子（排除GT路径格子）
            violation_cell_ids_in_gt = violation_cell_ids.intersection(gt_cell_ids)
            # if len(violation_cell_ids_in_gt) > 0:
                #print(f"  警告: 检测到{len(violation_cell_ids_in_gt)}个违规像素在GT路径格子中（可能是mask边界不精确）")
                #print(f"       违规像素覆盖的GT格子IDs: {sorted(violation_cell_ids_in_gt)}")
            
            # 只保留真正的背景违规格子
            true_violation_cell_ids = violation_cell_ids - gt_cell_ids
            violation_cell_count = len(true_violation_cell_ids)

#            print(violation_cell_ids)
#            print(gt_cell_ids)
#            print(violation_cell_count) 

            #from IPython import embed; embed()
            
            #print(f"  BFS遍历的总格子数: {len(violation_cell_ids)} (包含{len(violation_cell_ids_in_gt)}个GT格子)")
            #print(f"  BFS遍历的背景违规格子数: {violation_cell_count}")
            
            # 9. 计算总背景格子数
            # 背景格子 = 所有有效格子（cell_id >= 1）- GT路径格子
            unique_cells = set(np.unique(cell_ids_array).flatten())
            unique_cells.discard(0)  # 移除墙壁（cell_id=0）
            
            background_cell_ids = unique_cells - gt_cell_ids
            total_background_cells = len(background_cell_ids)
            
            #print(f"  总背景格子数: {total_background_cells}")
            
            if total_background_cells == 0:
                print("  警告: 没有背景格子")
                return 0.0
            
            # 10. 计算违规比例
            violation_ratio = violation_cell_count / total_background_cells
            
      #      print(f"  第四个reward（背景违规比例）: {violation_ratio:.4f}")
            
            
            return max(0.0, min(1.0, violation_ratio))

        except Exception as e:
            print(f"Error computing background violation reward: {e}")
            #raise e
            import traceback
            traceback.print_exc()
            return 1.0  # 出错返回最差分数


    def __call__(self, images: torch.Tensor, prompts: List[str],
                 metadata: List[Dict], only_strict: bool = False) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Compute combined reward for maze generation.

        Args:
            images: Generated images tensor (B, C, H, W)
            prompts: List of prompts
            metadata: List of metadata dictionaries
            only_strict: Whether to use only strict evaluation

        Returns:
            Tuple of (rewards_dict, reward_metadata)
        """
        batch_size = len(prompts)
        #print("--------")
        # 修改：初始化两个数组来存储MSE
        mse_inside_scores = np.zeros(batch_size) 
        mse_outside_scores = np.zeros(batch_size)
        mse_solution_scores = np.zeros(batch_size)  # 直接和sol_img做MSE的指标

        white_region_overlap_rewards = np.zeros(batch_size)  # Rule 2: White region preservation
        path_quality_rewards = np.zeros(batch_size)
        gt_cell_coverage_rewards = np.zeros(batch_size)  # 第三个reward: GT cell coverage
        background_violation_rewards = np.zeros(batch_size)  # 第四个reward: Background violation

        # Process each sample
        for i in range(batch_size):
            try:

                # 1. 首先从metadata中获取image_size（原始图像尺寸）
                import json
                image_size = None
                metadata_json_str = metadata[i].get('metadata', '{}')
                if isinstance(metadata_json_str, str):
                    try:
                        metadata_parsed = json.loads(metadata_json_str)
                        image_size = metadata_parsed.get('image_size', None)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(metadata_json_str, dict):
                    image_size = metadata_json_str.get('image_size', None)
                
                # 如果找到了image_size，使用它作为target_size
                target_size = None  # (H, W)
                if image_size is not None:
                    img_width = image_size.get('width', None)
                    img_height = image_size.get('height', None)
                    if img_width is not None and img_height is not None:
                        target_size = (img_height, img_width)  # (H, W)
                        print(f"[Sample {i}] Using image_size from metadata: {target_size} (H, W)")
                
                # 如果没有image_size，使用默认值（保持向后兼容）
                if target_size is None:
                    target_size = (1024, 1024)
                    print(f"[Sample {i}] Warning: No image_size in metadata, using default: {target_size}")
                
                
                # Get reference images (can be PIL Images, tensors, or file paths)
                ori_image, maze_ref, solution_ref, solution_mask = self.get_reference_images(prompts[i], metadata[i])
                
                # Extract single image
                single_image = images[i]  # Shape: (C, H, W)
                
                # Convert references to tensors for consistent processing
                # 检查每个reference是否存在，如果为None则抛出错误
                if ori_image is None:
                    raise ValueError(f"ori_image for sample {i} is None! Check metadata['original_img'] or metadata['original_img']")
                if maze_ref is None:
                    raise ValueError(f"maze_ref for sample {i} is None! Check metadata['m_original_img']")
                if solution_ref is None:
                    raise ValueError(f"solution_ref for sample {i} is None! Check metadata['sol_img']")
                if solution_mask is None:
                    raise ValueError(f"solution_mask for sample {i} is None! Check metadata['mask_img'] or metadata['solution_mask']")

                # print("solution_ref shape: ", solution_ref.size)
                # print("ori_image shape: ", ori_image.size)
                # print("maze_ref shape: ", maze_ref.size) #WH
                # print("solution_mask shape: ", solution_mask.size)
                # print("single_image shape: ", single_image.shape)
                # input()

                solution_ref = solution_ref.resize((target_size[1], target_size[0]), Image.BILINEAR)
                ori_image = ori_image.resize((target_size[1], target_size[0]), Image.BILINEAR) # HWC
                maze_ref = maze_ref.resize((target_size[1], target_size[0]), Image.BILINEAR)
                solution_mask = solution_mask.resize((target_size[1], target_size[0]), Image.BILINEAR) #HWC
                # print("solution_ref shape: ", solution_ref.size) 
                # print("ori_image shape: ", ori_image.size)
                # print("maze_ref shape: ", maze_ref.size)
                # print("solution_mask shape: ", solution_mask.size)
                # input()

                ori_image_tensor = torch.tensor(np.array(ori_image))#self._convert_to_tensor(ori_image)
                maze_ref_tensor = torch.tensor(np.array(maze_ref))#self._convert_to_tensor(maze_ref)
                solution_ref_tensor = torch.tensor(np.array(solution_ref))#self._convert_to_tensor(solution_ref)HWC

                # print(ori_image_tensor.shape, maze_ref_tensor.shape, solution_ref_tensor.shape) #HWC
                # Ensure same device
                if single_image.device != ori_image_tensor.device:
                    ori_image_tensor = ori_image_tensor.to(single_image.device)
                if single_image.device != maze_ref_tensor.device:
                    maze_ref_tensor = maze_ref_tensor.to(single_image.device)
                if single_image.device != solution_ref_tensor.device:
                    solution_ref_tensor = solution_ref_tensor.to(single_image.device)
                #single_image = self._convert_to_tensor(single_image)
        #        print(type(single_image))
            #    print(single_image.dtype)
                sing_img = (single_image*255.0).to(torch.uint8)
                
                
                # print("1", sing_img.shape) #CHW
                sing_img = cv2.resize(sing_img.numpy().transpose(1, 2, 0), (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
                sing_img = torch.tensor(sing_img.transpose(2, 0,1))  # HWC -> CHW
                # print("1.", sing_img.shape)
                Image.fromarray(sing_img.numpy().transpose(1, 2, 0).astype(np.uint8)).save("sing_img.jpg")
                # --- 修改开始：调用第一个 reward 接收两个返回值 ---
                mse_in, mse_out = self.compute_solution_space_reward(
                    sing_img, ori_image_tensor, solution_ref_tensor, solution_mask
                )
                print(f"第一个reward：{mse_in}, {mse_out}")
                # input()
                
                # 第二个reward：使用mask来提取IC和解图像来提取起终点，可选的原始图像检查黑色区域
                path_reward = 0#self.compute_path_quality_reward(single_image, solution_ref_tensor, ori_image_tensor, solution_mask)
        
                # 第三个reward：计算GT cell覆盖率
                cell_coverage_reward = 0.0
                if 'cell_map' in metadata[i] and metadata[i]['cell_map'] is not None:
                    cell_map = metadata[i]['cell_map']
                #     # print("cell_map shape: ", cell_map.size)
                #     cell_map = cell_map.resize((target_size[1], target_size[0]), Image.BILINEAR)
                #     print("cell_map shape2: ", cell_map.size)
                #     input()
                    metadata_json = metadata[i].get('metadata', '{}')
                    cell_coverage_reward = self.compute_gt_cell_coverage_reward(
                        sing_img, cell_map, metadata_json, solution_mask, solution_ref_tensor, sample_index=i
                    )
                    print(f"格子内reward：{cell_coverage_reward}")
                    if cell_coverage_reward < 0:
                       print("生成乱码图像，格子内reward返回-1")
                else:
                    print(f"  样本 {i}: 缺少cell_map，跳过第三个reward")

                # 第四个reward：计算背景违规比例
                background_violation_reward = 0.0
                if 'cell_map' in metadata[i] and metadata[i]['cell_map'] is not None:
                    cell_map = metadata[i]['cell_map']
                #     cell_map = cell_map.resize((target_size[0], target_size[1]), Image.BILINEAR)
                    # print(cell_map.size)
                    # input()
                    metadata_json = metadata[i].get('metadata', '{}')
                    background_violation_reward = self.compute_background_violation_reward(
                        sing_img, cell_map, metadata_json, solution_mask, 
                        sol_img=solution_ref, original_img=ori_image, sample_index=i
                    )
                    print(f"格子外reward：{background_violation_reward}")
                else:
                    print(f"  样本 {i}: 缺少cell_map，跳过第四个reward")

                # --- 修改：存储 MSE ---
                mse_inside_scores[i] = mse_in
                mse_outside_scores[i] = mse_out

                # 计算直接和sol_img的MSE
                # solution_ref_tensor 是从 np.array(solution_ref) 得到的，应该是 HWC 格式
                if isinstance(solution_ref_tensor, np.ndarray):
                    solution_ref_tensor = torch.tensor(solution_ref_tensor)
                
                # 确保格式一致：都转换为 (C, H, W) 格式
                if solution_ref_tensor.ndim == 3 and solution_ref_tensor.shape[2] == 3:  # HWC格式
                    solution_ref_chw = solution_ref_tensor.permute(2, 0, 1)  # 转为CHW
                elif solution_ref_tensor.ndim == 3 and solution_ref_tensor.shape[0] == 3:  # 已经是CHW格式
                    solution_ref_chw = solution_ref_tensor
                else:
                    solution_ref_chw = solution_ref_tensor
                
                # 确保在CPU上处理（因为sing_img在CPU上）
                if solution_ref_chw.device != torch.device('cpu'):
                    solution_ref_chw = solution_ref_chw.cpu()
                
                # 归一化到[0, 1]范围
                solution_ref_chw = solution_ref_chw.float()
                if solution_ref_chw.max() > 1.0:
                    solution_ref_chw = solution_ref_chw / 255.0
                
                # sing_img是uint8格式[0, 255]，需要归一化
                sing_img_normalized = sing_img.float() / 255.0
                
                # 确保尺寸一致
                if sing_img_normalized.shape != solution_ref_chw.shape:
                    solution_ref_chw = F.interpolate(
                        solution_ref_chw.unsqueeze(0),
                        size=sing_img_normalized.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                # 计算MSE
                mse_solution = F.mse_loss(sing_img_normalized, solution_ref_chw).item()
                mse_solution_scores[i] = mse_solution

                path_quality_rewards[i] = path_reward
                gt_cell_coverage_rewards[i] = cell_coverage_reward
                background_violation_rewards[i] = background_violation_reward

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Set default rewards for failed samples
                white_region_overlap_rewards[i] = 0.0
                path_quality_rewards[i] = 0.0
                gt_cell_coverage_rewards[i] = 0.0
                background_violation_rewards[i] = 0.0
                mse_solution_scores[i] = 0.0

        # Compute combined reward
        combined_rewards = (self.weights['solution_space'] * white_region_overlap_rewards +
                          self.weights['path_quality'] * path_quality_rewards +
                          self.weights['gt_cell_coverage'] * gt_cell_coverage_rewards +
                          self.weights['background_violation'] * background_violation_rewards)
        gt_and_bg_reward = (mse_inside_scores * self.weights['solution_space'] + mse_outside_scores * self.weights['solution_space'] + self.weights['gt_cell_coverage'] * gt_cell_coverage_rewards +
                          self.weights['background_violation'] * background_violation_rewards)
        # gt_and_bg_reward = 1-np.abs(gt_and_bg_reward)
        combined_rewards = np.where(gt_and_bg_reward < 0, 0.0, gt_and_bg_reward)
        # Prepare rewards dictionary
        rewards = {
            'avg': combined_rewards,
            'mse_inside': mse_inside_scores,   # 返回原始 MSE Inside
            'mse_outside': mse_outside_scores, # 返回原始 MSE Outside
            'mse_solution': mse_solution_scores,  # 直接和sol_img做MSE的指标
            'path_validity': path_quality_rewards,  # 第二个reward: path validity
            'gt_cell_coverage': gt_cell_coverage_rewards,  # 第三个reward: GT cell coverage
            'background_violation': background_violation_rewards  # 第四个reward: Background violation
        }

        # Metadata for logging
        reward_metadata = {
            'weights': self.weights,
            'mean_mse_inside': np.mean(mse_inside_scores),    # Log mean MSE Inside
            'mean_mse_outside': np.mean(mse_outside_scores),  # Log mean MSE Outside
            'mean_mse_solution': np.mean(mse_solution_scores),  # Log mean MSE with solution image
            'mean_path_validity': np.mean(path_quality_rewards),  # 第二个reward
            'mean_gt_cell_coverage': np.mean(gt_cell_coverage_rewards),  # 第三个reward
            'mean_background_violation': np.mean(background_violation_rewards),  # 第四个reward
            'mean_combined': np.mean(combined_rewards)
        }

        return rewards, reward_metadata


# Factory function to create the reward function
def create_maze_reward_function(config=None):
    """
    Factory function to create maze reward function.

    For parquet-based datasets, images are stored directly in metadata as PIL Images.

    Args:
        config: Configuration object (optional)

    Returns:
        Maze reward function compatible with flow_grpo reward interface
    """
    # Default weights
    weights = {
        'solution_space': -1.0,
        'path_quality': 0.0,
        'gt_cell_coverage': 1.0,         # 第三个reward: GT cell coverage (权重1)
        'background_violation': -1.0     # 第四个reward: Background violation (权重-1，惩罚违规)
    }

    if config and hasattr(config, 'reward_weights'):
        weights.update(config.reward_weights)

    # Create reward function (images are read from metadata in parquet format)
    maze_reward = MazeRewardFunction(weights=weights)

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        # Call the maze reward function
        rewards, reward_metadata = maze_reward(images, prompts, metadata, only_strict)
        return rewards, reward_metadata

    return _fn


def maze_metric(device):
    """
    Maze reward function compatible with flow_grpo reward interface.

    Args:
        device: Device to run the reward function on

    Returns:
        Reward function that returns (scores, metadata) format
    """
    # Create the maze reward function with default settings
    maze_reward_fn = create_maze_reward_function()

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        # Ensure images are on CPU for processing
        if isinstance(images, torch.Tensor) and images.device != torch.device('cpu'):
            images = images.cpu()

        # Call the maze reward function
        rewards_dict, reward_metadata = maze_reward_fn(images, prompts, metadata, ref_images, only_strict)

        # Return the combined/average scores in the expected format
        # The multi_score function expects (scores, metadata) where scores is a list/array
        scores = rewards_dict['avg']  # This should be the combined reward scores
        return scores, rewards_dict

    return _fn


