#!/usr/bin/env python3
"""
多尺度Token提取和可视化工具

这个脚本实现了您想要的功能：
1. 输入一张图片，让它过一遍Infinity的流程
2. 提取前四个尺度的原始token和纠正后token
3. 保存这些token到文件
4. 可视化token并生成解码后的图片

使用方法：
    python extract_multiscale_tokens.py
    
只需要修改IMAGE_PATH变量为您的图片路径即可。
"""

import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

# 添加项目路径到sys.path
sys.path.append('.')

from infinity.models.bsq_vae.vae import vae_model
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


# ==================== 配置参数 ====================
# 您只需要修改这里的路径配置
IMAGE_PATH = "data/infinity_toy_data/images/5134521536907147208.jpg"  # 修改为您的图片路径
VAE_MODEL_PATH = "weights/infinity_vae_d32reg.pth"  # VAE模型路径
OUTPUT_DIR = "output/extracted_tokens"  # 输出目录

# VAE配置参数
VAE_CONFIG = {
    'vae_type': 32,  # codebook_dim
    'apply_spatial_patchify': 0,  # 0表示不使用spatial patchify
    'schedule_mode': "dynamic",
    'patch_size': 16,
    'encoder_ch_mult': [1, 2, 4, 4, 4],
    'decoder_ch_mult': [1, 2, 4, 4, 4],
}

# 位级自我纠正配置
BSC_CONFIG = {
    'noise_apply_layers': 4,  # 前4层应用噪声纠正
    'noise_apply_requant': True,
    'noise_apply_strength': 0.1,  # 噪声强度
    'debug_bsc': False,
}
# ===================================================


class TokenExtractor:
    """多尺度Token提取器"""
    
    def __init__(self, vae_model_path, vae_config, bsc_config):
        """
        初始化提取器
        
        Args:
            vae_model_path: VAE模型路径
            vae_config: VAE配置参数
            bsc_config: 位级自我纠正配置
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae_config = vae_config
        self.bsc_config = bsc_config
        
        print(f"正在初始化TokenExtractor...")
        print(f"设备: {self.device}")
        
        # 加载VAE模型
        self.vae = self._load_vae_model(vae_model_path, vae_config)
        
        # 创建位级自我纠正模块
        self.bitwise_self_correction = self._create_bsc_module(bsc_config)
        
        print("TokenExtractor初始化完成！")
    
    def _load_vae_model(self, model_path, config):
        """加载VAE模型"""
        print(f"正在加载VAE模型: {model_path}")
        
        codebook_dim = config['vae_type']
        codebook_size = 2 ** codebook_dim
        
        if config['apply_spatial_patchify']:
            patch_size = 8
            encoder_ch_mult = [1, 2, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4]
        else:
            patch_size = config['patch_size']
            encoder_ch_mult = config['encoder_ch_mult']
            decoder_ch_mult = config['decoder_ch_mult']
        
        vae = vae_model(
            vqgan_ckpt=model_path,
            schedule_mode=config['schedule_mode'],
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            patch_size=patch_size,
            encoder_ch_mult=encoder_ch_mult,
            decoder_ch_mult=decoder_ch_mult,
            test_mode=True
        ).to(self.device)
        
        # 设置为评估模式
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
            
        print(f"VAE模型加载完成，codebook_dim={codebook_dim}")
        return vae
    
    def _create_bsc_module(self, bsc_config):
        """创建位级自我纠正模块"""
        # 创建一个简单的args对象来传递配置
        class Args:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        # 添加必要的配置参数
        args = Args(
            noise_apply_layers=bsc_config['noise_apply_layers'],
            noise_apply_requant=bsc_config['noise_apply_requant'],
            noise_apply_strength=bsc_config['noise_apply_strength'],
            apply_spatial_patchify=self.vae_config['apply_spatial_patchify'],
            debug_bsc=bsc_config['debug_bsc']
        )
        
        return BitwiseSelfCorrection(self.vae, args)
    
    def _preprocess_image(self, image_path, target_h=512, target_w=512):
        """
        图像预处理 - 强制转换为512x512
        
        这个函数将任意尺寸的输入图像转换为512x512，并转换为模型需要的tensor格式。
        无论输入图像的原始尺寸如何，都会被调整为512x512以匹配模型要求。
        
        处理策略：
        1. 首先按比例缩放，使得较长边等于512
        2. 然后进行中心裁剪，确保最终尺寸为512x512
        3. 这样可以保持图像的核心内容，避免严重的拉伸变形
        """
        print(f"正在预处理图像: {image_path}")
        
        # 加载图像
        pil_image = Image.open(image_path).convert('RGB')
        width, height = pil_image.size
        print(f"原始图像尺寸: {width} x {height}")
        
        # 强制目标尺寸为512x512（适配您的模型）
        target_h, target_w = 512, 512
        
        # 计算缩放比例 - 使较长边等于512，保持宽高比
        scale_factor = max(target_w / width, target_h / height)
        
        # 按比例缩放
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        print(f"缩放后尺寸: {new_width} x {new_height}")
        pil_image = pil_image.resize((new_width, new_height), resample=Image.LANCZOS)
        
        # 转换为numpy数组进行裁剪
        arr = np.array(pil_image)
        
        # 中心裁剪到512x512
        crop_y = max(0, (arr.shape[0] - target_h) // 2)
        crop_x = max(0, (arr.shape[1] - target_w) // 2)
        
        # 确保裁剪区域不超出图像边界
        crop_y_end = min(arr.shape[0], crop_y + target_h)
        crop_x_end = min(arr.shape[1], crop_x + target_w)
        
        cropped_arr = arr[crop_y:crop_y_end, crop_x:crop_x_end]
        
        # 如果裁剪后的图像小于512x512，进行填充
        if cropped_arr.shape[0] < target_h or cropped_arr.shape[1] < target_w:
            padded_arr = np.ones((target_h, target_w, 3), dtype=np.uint8) * 128  # 灰色填充
            
            # 计算填充位置（居中放置）
            pad_y = (target_h - cropped_arr.shape[0]) // 2
            pad_x = (target_w - cropped_arr.shape[1]) // 2
            
            padded_arr[pad_y:pad_y + cropped_arr.shape[0], 
                      pad_x:pad_x + cropped_arr.shape[1]] = cropped_arr
            
            cropped_arr = padded_arr
            print(f"图像已填充至: {target_h} x {target_w}")
        
        # 转换为tensor
        im = to_tensor(cropped_arr)
        
        # 转换到[-1, 1]范围（从[0, 1]转换）
        im = im * 2.0 - 1.0
        
        print(f"最终预处理后图像尺寸: {target_h} x {target_w}")
        print(f"tensor形状: {im.shape}, 数值范围: [{im.min():.3f}, {im.max():.3f}]")
        
        return im.unsqueeze(0).to(self.device)
    
    def extract_multiscale_tokens(self, image_path, h_div_w=1.0, pn='1M'):
        """
        提取多尺度token的主函数
        
        Args:
            image_path: 输入图像路径
            h_div_w: 宽高比
            pn: 像素数量级别 ('0.06M', '0.25M', '1M')
            
        Returns:
            dict: 包含原始token和纠正后token的字典
        """
        print(f"\n开始提取多尺度token...")
        print(f"输入图像: {image_path}")
        print(f"宽高比: {h_div_w}, 像素级别: {pn}")
        
        # 确定最接近的宽高比模板
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        print(f"使用的宽高比模板: {h_div_w_template}")
        
        # 获取尺度调度表
        scale_info = dynamic_resolution_h_w[h_div_w_template][pn]
        scale_schedule = scale_info['scales']
        target_h, target_w = scale_info['pixel']
        
        print(f"目标尺寸: {target_h} x {target_w}")
        print(f"尺度调度表: {scale_schedule}")
        
        # VAE需要的尺度调度表（如果使用spatial patchify需要调整）
        if self.vae_config['apply_spatial_patchify']:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        # 预处理图像
        inp_tensor = self._preprocess_image(image_path, target_h, target_w)
        
        # 第一步：VAE编码获取原始特征
        print("\n第一步：VAE编码获取原始特征...")
        with torch.no_grad():
            raw_features, _, _ = self.vae.encode_for_raw_features(
                inp_tensor, 
                scale_schedule=vae_scale_schedule
            )
        print(f"原始特征形状: {raw_features.shape}")
        
        # 第二步：位级自我纠正获取多尺度token
        print("\n第二步：位级自我纠正获取多尺度token...")
        with torch.no_grad():
            x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(
                vae_scale_schedule, 
                inp_tensor, 
                raw_features, 
                self.device
            )
        
        # 获取纠正后的token（通过重新运行flip_requant获取）
        # 这里我们需要访问BitwiseSelfCorrection内部的pred_all_bit_indices
        # 由于原始代码中没有直接返回，我们需要修改一下逻辑
        pred_ms_idx_Bl = self._extract_corrected_tokens(
            vae_scale_schedule, inp_tensor, raw_features
        )
        
        print(f"提取到 {len(gt_ms_idx_Bl)} 个尺度的原始token")
        print(f"提取到 {len(pred_ms_idx_Bl)} 个尺度的纠正后token")
        
        # 只保留前四个尺度
        num_scales_to_keep = min(4, len(gt_ms_idx_Bl))
        
        result = {
            'original_tokens': gt_ms_idx_Bl[:num_scales_to_keep],
            'corrected_tokens': pred_ms_idx_Bl[:num_scales_to_keep],
            'scale_schedule': vae_scale_schedule[:num_scales_to_keep],
            'input_tensor': inp_tensor,
            'raw_features': raw_features,
            'metadata': {
                'image_path': image_path,
                'h_div_w_template': h_div_w_template,
                'target_size': (target_h, target_w),
                'num_scales': num_scales_to_keep,
                'vae_config': self.vae_config,
                'bsc_config': self.bsc_config
            }
        }
        
        print(f"\n成功提取前{num_scales_to_keep}个尺度的token！")
        return result
    
    def _extract_corrected_tokens(self, vae_scale_schedule, inp_B3HW, raw_features):
        """
        提取纠正后的token
        
        这个函数重新实现了flip_requant的部分逻辑，专门用于获取纠正后的token
        """
        with torch.amp.autocast('cuda', enabled=False):
            B = raw_features.shape[0]
            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2)
            else:
                codes_out = raw_features
            
            cum_var_input = 0
            pred_all_bit_indices = []
            
            for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                residual = codes_out - cum_var_input
                if si != len(vae_scale_schedule) - 1:
                    residual = F.interpolate(
                        residual, 
                        size=vae_scale_schedule[si], 
                        mode=self.vae.quantizer.z_interplote_down
                    ).contiguous()
                
                # 量化
                quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual)
                
                # 应用纠正（如果在纠正层数范围内）
                if si < self.bsc_config['noise_apply_layers']:
                    noise_apply_strength = np.random.randint(
                        0, 100 * self.bsc_config['noise_apply_strength'] + 1
                    ) * 0.01
                    mask = torch.rand(*bit_indices.shape).to(self.device) < noise_apply_strength
                    pred_bit_indices = bit_indices.clone()
                    pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
                    pred_all_bit_indices.append(pred_bit_indices)
                    
                    if self.bsc_config['noise_apply_requant']:
                        quantized = self.vae.quantizer.lfq.indices_to_codes(
                            pred_bit_indices, label_type='bit_label'
                        )
                else:
                    pred_all_bit_indices.append(bit_indices)
                
                # 累积输入
                cum_var_input = cum_var_input + F.interpolate(
                    quantized, 
                    size=vae_scale_schedule[-1], 
                    mode=self.vae.quantizer.z_interplote_up
                ).contiguous()
        
        # 转换为与原始token相同的格式
        if self.vae_config['apply_spatial_patchify']:
            pred_ms_idx_Bl = []
            for item in pred_all_bit_indices:
                # item shape: (B,1,H,W,d)
                item = item.squeeze(1).permute(0, 3, 1, 2)  # (B,d,H,W)
                # (B,d,H,W) -> (B,4d,H/2,W/2)
                item = torch.nn.functional.pixel_unshuffle(item, 2)
                # (B,4d,H/2,W/2) -> (B,H/2,W/2,4d) -> (B,H/2*W/2,4d)
                item = item.permute(0, 2, 3, 1).reshape(B, -1, 4 * self.vae.codebook_dim)
                pred_ms_idx_Bl.append(item)
        else:
            pred_ms_idx_Bl = [
                item.reshape(B, -1, self.vae.codebook_dim) 
                for item in pred_all_bit_indices
            ]
        
        return pred_ms_idx_Bl
    
    def save_tokens(self, extraction_result, output_dir):
        """
        保存提取的token到文件
        
        Args:
            extraction_result: extract_multiscale_tokens的返回结果
            output_dir: 输出目录
        """
        print(f"\n正在保存token到: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始token
        original_tokens_path = osp.join(output_dir, 'original_tokens.pth')
        torch.save(extraction_result['original_tokens'], original_tokens_path)
        print(f"原始token已保存到: {original_tokens_path}")
        
        # 保存纠正后token
        corrected_tokens_path = osp.join(output_dir, 'corrected_tokens.pth')
        torch.save(extraction_result['corrected_tokens'], corrected_tokens_path)
        print(f"纠正后token已保存到: {corrected_tokens_path}")
        
        # 保存元数据
        metadata_path = osp.join(output_dir, 'metadata.pth')
        torch.save(extraction_result['metadata'], metadata_path)
        print(f"元数据已保存到: {metadata_path}")
        
        # 保存尺度信息为可读格式
        scale_info_path = osp.join(output_dir, 'scale_info.txt')
        with open(scale_info_path, 'w') as f:
            f.write(f"图像路径: {extraction_result['metadata']['image_path']}\n")
            f.write(f"目标尺寸: {extraction_result['metadata']['target_size']}\n")
            f.write(f"尺度数量: {extraction_result['metadata']['num_scales']}\n")
            f.write(f"尺度调度表:\n")
            for i, scale in enumerate(extraction_result['scale_schedule']):
                f.write(f"  尺度{i+1}: {scale}\n")
        print(f"尺度信息已保存到: {scale_info_path}")
    
    def visualize_tokens(self, extraction_result, output_dir):
        """
        可视化token和解码后的图片
        
        Args:
            extraction_result: extract_multiscale_tokens的返回结果
            output_dir: 输出目录
        """
        print(f"\n正在进行可视化...")
        
        vis_dir = osp.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        original_tokens = extraction_result['original_tokens']
        corrected_tokens = extraction_result['corrected_tokens']
        scale_schedule = extraction_result['scale_schedule']
        input_tensor = extraction_result['input_tensor']
        
        # 保存原始输入图像
        self._save_tensor_as_image(
            input_tensor, 
            osp.join(vis_dir, 'original_input.jpg'),
            "原始输入图像"
        )
        
        # 对每个尺度进行可视化
        for scale_idx in range(len(original_tokens)):
            print(f"正在处理尺度 {scale_idx + 1}/{len(original_tokens)}...")
            
            scale_info = scale_schedule[scale_idx]
            
            # 创建该尺度的输出目录
            scale_dir = osp.join(vis_dir, f'scale_{scale_idx+1}')
            os.makedirs(scale_dir, exist_ok=True)
            
            # 可视化原始token
            self._visualize_single_scale_tokens(
                [original_tokens[scale_idx]], 
                scale_schedule[:scale_idx+1],
                scale_dir, 
                f'original_scale_{scale_idx+1}',
                f"原始Token - 尺度{scale_idx+1} {scale_info}"
            )
            
            # 可视化纠正后token
            self._visualize_single_scale_tokens(
                [corrected_tokens[scale_idx]], 
                scale_schedule[:scale_idx+1],
                scale_dir, 
                f'corrected_scale_{scale_idx+1}',
                f"纠正后Token - 尺度{scale_idx+1} {scale_info}"
            )
        
        # 创建对比可视化
        self._create_comparison_visualization(
            original_tokens, corrected_tokens, scale_schedule, vis_dir
        )
        
        print(f"可视化完成，结果保存在: {vis_dir}")
    
    def _visualize_single_scale_tokens(self, tokens_list, scale_schedule, output_dir, prefix, title):
        """可视化单个尺度的token"""
        try:
            with torch.no_grad():
                # 使用VAE解码token
                summed_codes, decoded_images = self.vae.decode_from_indices(
                    tokens_list, scale_schedule, label_type='bit_label'
                )
                
                # 保存解码后的图像
                if len(decoded_images) > 0:
                    self._save_tensor_as_image(
                        decoded_images[0:1], 
                        osp.join(output_dir, f'{prefix}_decoded.jpg'),
                        title
                    )
                
                # 可视化token的统计信息
                self._plot_token_statistics(
                    tokens_list[0], 
                    osp.join(output_dir, f'{prefix}_stats.png'),
                    title
                )
                
        except Exception as e:
            print(f"可视化{prefix}时出错: {e}")
    
    def _save_tensor_as_image(self, tensor, filepath, title=""):
        """将tensor保存为图像文件"""
        if tensor.dim() == 4:
            # 假设tensor格式为 (B, C, H, W)，值域为[-1, 1]
            img = tensor[0]  # 取第一个batch
            img = (img + 1) / 2  # 转换到[0, 1]
            img = img.permute(1, 2, 0)  # 转换到(H, W, C)
            img = torch.clamp(img, 0, 1)
            
            # 转换为numpy并保存
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            
            # 如果是RGB格式，转换为BGR（OpenCV格式）
            if img_np.shape[2] == 3:
                img_np = img_np[:, :, ::-1]  # RGB to BGR
            
            cv2.imwrite(filepath, img_np)
            print(f"  - 图像已保存: {filepath}")
    
    def _plot_token_statistics(self, tokens, filepath, title):
        """绘制token的统计信息"""
        try:
            # tokens shape: (B, seq_len, feature_dim)
            tokens_np = tokens[0].cpu().numpy()  # 取第一个batch
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(title, fontsize=14)
            
            # 绘制token值的分布直方图
            axes[0, 0].hist(tokens_np.flatten(), bins=50, alpha=0.7)
            axes[0, 0].set_title('Token值分布')
            axes[0, 0].set_xlabel('Token值')
            axes[0, 0].set_ylabel('频次')
            
            # 绘制每个特征维度的均值
            feature_means = np.mean(tokens_np, axis=0)
            axes[0, 1].plot(feature_means)
            axes[0, 1].set_title('各特征维度均值')
            axes[0, 1].set_xlabel('特征维度')
            axes[0, 1].set_ylabel('均值')
            
            # 绘制序列位置的均值
            seq_means = np.mean(tokens_np, axis=1)
            axes[1, 0].plot(seq_means)
            axes[1, 0].set_title('各序列位置均值')
            axes[1, 0].set_xlabel('序列位置')
            axes[1, 0].set_ylabel('均值')
            
            # 绘制token的2D可视化（使用前两个特征维度）
            if tokens_np.shape[1] >= 2:
                axes[1, 1].scatter(tokens_np[:, 0], tokens_np[:, 1], alpha=0.6, s=1)
                axes[1, 1].set_title('Token 2D分布 (前两维)')
                axes[1, 1].set_xlabel('特征维度1')
                axes[1, 1].set_ylabel('特征维度2')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  - 统计图已保存: {filepath}")
            
        except Exception as e:
            print(f"  - 绘制统计图时出错: {e}")
    
    def _create_comparison_visualization(self, original_tokens, corrected_tokens, scale_schedule, output_dir):
        """创建原始和纠正后token的对比可视化"""
        print("正在创建对比可视化...")
        
        try:
            # 解码所有尺度的原始token
            print("  - 解码原始tokens...")
            with torch.no_grad():
                _, original_images = self.vae.decode_from_indices(
                    original_tokens, scale_schedule, label_type='bit_label'
                )
            
            # 解码所有尺度的纠正后token
            print("  - 解码纠正后tokens...")
            with torch.no_grad():
                _, corrected_images = self.vae.decode_from_indices(
                    corrected_tokens, scale_schedule, label_type='bit_label'
                )
            
            # 保存对比图像
            if len(original_images) > 0 and len(corrected_images) > 0:
                self._save_tensor_as_image(
                    original_images[0:1], 
                    osp.join(output_dir, 'all_scales_original.jpg'),
                    "所有尺度原始Token解码结果"
                )
                
                self._save_tensor_as_image(
                    corrected_images[0:1], 
                    osp.join(output_dir, 'all_scales_corrected.jpg'),
                    "所有尺度纠正后Token解码结果"
                )
                
                print("  - 对比可视化完成")
            
        except Exception as e:
            print(f"  - 创建对比可视化时出错: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("多尺度Token提取和可视化工具")
    print("=" * 60)
    
    # 检查输入图像是否存在
    if not osp.exists(IMAGE_PATH):
        print(f"错误：输入图像不存在: {IMAGE_PATH}")
        print("请修改IMAGE_PATH变量为正确的图像路径")
        return
    
    # 检查VAE模型是否存在
    if not osp.exists(VAE_MODEL_PATH):
        print(f"错误：VAE模型不存在: {VAE_MODEL_PATH}")
        print("请修改VAE_MODEL_PATH变量为正确的模型路径")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化Token提取器
    extractor = TokenExtractor(
        vae_model_path=VAE_MODEL_PATH,
        vae_config=VAE_CONFIG,
        bsc_config=BSC_CONFIG
    )
    
    # 提取多尺度token
    start_time = time.time()
    extraction_result = extractor.extract_multiscale_tokens(
        image_path=IMAGE_PATH,
        h_div_w=1.0,  # 1:1宽高比
        pn='1M'       # 1M像素级别
    )
    extraction_time = time.time() - start_time
    
    print(f"\nToken提取完成，耗时: {extraction_time:.2f}秒")
    
    # 保存token
    extractor.save_tokens(extraction_result, OUTPUT_DIR)
    
    # 进行可视化
    extractor.visualize_tokens(extraction_result, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"结果保存在: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 打印一些统计信息
    print(f"\n统计信息:")
    print(f"- 输入图像: {IMAGE_PATH}")
    print(f"- 提取的尺度数量: {len(extraction_result['original_tokens'])}")
    print(f"- 目标尺寸: {extraction_result['metadata']['target_size']}")
    
    for i, (orig_token, corr_token) in enumerate(zip(
        extraction_result['original_tokens'], 
        extraction_result['corrected_tokens']
    )):
        print(f"- 尺度{i+1} Token形状: {orig_token.shape}")


if __name__ == "__main__":
    main()
