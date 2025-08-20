#!/usr/bin/env python3
"""
多尺度Token提取和可视化工具 - 健壮版本

这个版本修复了原版本中的维度处理问题，能够正确处理不同配置下的token形状。

主要改进：
1. 修复了BitWiseSelfCorrection中的维度错误
2. 更健壮的维度处理逻辑
3. 更清晰的调试信息
4. 简化的token提取流程

使用方法：
    python extract_multiscale_tokens_robust.py
"""

import os
import os.path as osp
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

# 添加项目路径
sys.path.append('.')

from infinity.models.bsq_vae.vae import vae_model
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


# ==================== 配置参数 ====================
# 修改这里的路径
IMAGE_PATH = "000_turn_left_LeftView.png"  # 修改为您的图片路径
VAE_MODEL_PATH = "weights/infinity_vae_d56_f8_14_patchify.pth"  # VAE模型路径
OUTPUT_DIR = "output/extracted_tokens_robust"  # 输出目录

# VAE配置（匹配您的inf.py配置）
VAE_CONFIG = {
    'vae_type': 14,  # codebook_dim=14
    'apply_spatial_patchify': 1,  # 1表示使用spatial patchify
    'schedule_mode': "dynamic",
    'patch_size': 8,
    'encoder_ch_mult': [1, 2, 4, 4],
    'decoder_ch_mult': [1, 2, 4, 4],
}

# 自我纠错配置
NOISE_CONFIG = {
    'apply_noise_layers': 4,  # 对前4层应用噪声
    'noise_strength': 0.1,    # 噪声强度
}
# ===================================================


class RobustTokenExtractor:
    """健壮的多尺度Token提取器"""
    
    def __init__(self, vae_model_path, vae_config):
        """
        初始化提取器
        
        Args:
            vae_model_path: VAE模型路径
            vae_config: VAE配置参数
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae_config = vae_config
        
        print(f"正在初始化健壮的TokenExtractor...")
        print(f"设备: {self.device}")
        
        # 加载VAE模型
        self.vae = self._load_vae_model(vae_model_path, vae_config)
        
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
    
    def _preprocess_image(self, image_path, target_h=512, target_w=512):
        """
        图像预处理 - 将任意尺寸图像转换为512x512
        
        Args:
            image_path: 输入图像路径
            target_h: 目标高度
            target_w: 目标宽度
            
        Returns:
            preprocessed_tensor: 预处理后的张量 [1, 3, 512, 512]，值域[-1, 1]
        """
        print(f"正在预处理图像: {image_path}")
        
        # 加载图像
        pil_image = Image.open(image_path).convert('RGB')
        width, height = pil_image.size
        print(f"原始图像尺寸: {width} x {height}")
        
        # 计算缩放比例，保持宽高比，使较长边适配目标尺寸
        scale_factor = max(target_w / width, target_h / height)
        
        # 按比例缩放
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        print(f"缩放后尺寸: {new_width} x {new_height}")
        pil_image = pil_image.resize((new_width, new_height), resample=Image.LANCZOS)
        
        # 转换为numpy数组进行中心裁剪
        arr = np.array(pil_image)
        
        # 中心裁剪到目标尺寸
        crop_y = max(0, (arr.shape[0] - target_h) // 2)
        crop_x = max(0, (arr.shape[1] - target_w) // 2)
        
        crop_y_end = min(arr.shape[0], crop_y + target_h)
        crop_x_end = min(arr.shape[1], crop_x + target_w)
        
        cropped_arr = arr[crop_y:crop_y_end, crop_x:crop_x_end]
        
        # 如果裁剪后小于目标尺寸，进行填充
        if cropped_arr.shape[0] < target_h or cropped_arr.shape[1] < target_w:
            padded_arr = np.ones((target_h, target_w, 3), dtype=np.uint8) * 128  # 灰色填充
            
            pad_y = (target_h - cropped_arr.shape[0]) // 2
            pad_x = (target_w - cropped_arr.shape[1]) // 2
            
            padded_arr[pad_y:pad_y + cropped_arr.shape[0], 
                      pad_x:pad_x + cropped_arr.shape[1]] = cropped_arr
            
            cropped_arr = padded_arr
            print(f"图像已填充至: {target_h} x {target_w}")
        
        # 转换为tensor并归一化到[-1, 1]
        im = to_tensor(cropped_arr)  # 转为[0, 1]
        im = im * 2.0 - 1.0          # 转为[-1, 1]
        
        print(f"最终预处理后图像: {target_h} x {target_w}")
        print(f"tensor形状: {im.shape}, 数值范围: [{im.min():.3f}, {im.max():.3f}]")
        
        return im.unsqueeze(0).to(self.device)  # 添加batch维度
    
    def _robust_multiscale_quantization(self, raw_features, scale_schedule, noise_config):
        """
        健壮的多尺度量化处理
        
        这个方法重新实现了BitwiseSelfCorrection的核心逻辑，
        但增加了对不同维度情况的健壮处理。
        
        Args:
            raw_features: VAE编码后的原始特征
            scale_schedule: 尺度调度表
            noise_config: 噪声配置
            
        Returns:
            original_tokens: 原始量化token列表
            corrected_tokens: 纠错后token列表
        """
        print("正在进行健壮的多尺度量化...")
        
        with torch.amp.autocast('cuda', enabled=False):
            B = raw_features.shape[0]
            
            # 确保raw_features有正确的维度
            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2)  # 添加时间维度
            else:
                codes_out = raw_features
            
            print(f"量化输入特征形状: {codes_out.shape}")
            
            cum_var_input = 0
            original_tokens = []
            corrected_tokens = []
            
            for si, (pt, ph, pw) in enumerate(scale_schedule):
                print(f"处理尺度 {si+1}/{len(scale_schedule)}: ({pt}, {ph}, {pw})")
                
                # 计算当前尺度的残差
                residual = codes_out - cum_var_input
                
                # 如果不是最后一个尺度，调整分辨率
                if si != len(scale_schedule) - 1:
                    residual = F.interpolate(
                        residual, 
                        size=scale_schedule[si], 
                        mode=self.vae.quantizer.z_interplote_down
                    ).contiguous()
                
                # 进行量化
                quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual)
                
                print(f"  - 量化后bit_indices形状: {bit_indices.shape}")
                
                # 处理原始token
                original_token = self._process_bit_indices(bit_indices, si, B)
                original_tokens.append(original_token)
                
                # 生成纠错后的token
                corrected_bit_indices = self._apply_noise_correction(
                    bit_indices, si, noise_config
                )
                corrected_token = self._process_bit_indices(corrected_bit_indices, si, B)
                corrected_tokens.append(corrected_token)
                
                # 更新累积输入
                if noise_config['apply_noise_layers'] > si:
                    # 如果应用了纠错，使用纠错后的quantized
                    quantized = self.vae.quantizer.lfq.indices_to_codes(
                        corrected_bit_indices, label_type='bit_label'
                    )
                
                cum_var_input = cum_var_input + F.interpolate(
                    quantized, 
                    size=scale_schedule[-1], 
                    mode=self.vae.quantizer.z_interplote_up
                ).contiguous()
        
        print(f"多尺度量化完成，生成了 {len(original_tokens)} 个尺度的token")
        return original_tokens, corrected_tokens
    
    def _process_bit_indices(self, bit_indices, scale_index, batch_size):
        """
        健壮地处理bit_indices，无论其维度如何
        
        这个方法能够处理不同形状的bit_indices：
        - 4D: (B, H, W, D)
        - 5D: (B, T, H, W, D) 或 (B, H, W, C, D)
        - 6D: (B, T, H, W, C, D)
        
        Args:
            bit_indices: 量化后的bit索引
            scale_index: 当前尺度索引
            batch_size: 批次大小
            
        Returns:
            processed_token: 处理后的token，形状为 (B, seq_len, feature_dim)
        """
        original_shape = bit_indices.shape
        print(f"    处理bit_indices: {original_shape}")
        
        # 确保bit_indices在CPU上进行处理（避免某些操作的设备问题）
        device = bit_indices.device
        bit_indices = bit_indices.cpu()
        
        # 根据维度数量进行不同的处理
        if bit_indices.dim() == 4:
            # 形状: (B, H, W, D)
            B, H, W, D = bit_indices.shape
            processed = bit_indices
            
        elif bit_indices.dim() == 5:
            # 可能是 (B, T, H, W, D) 或 (B, H, W, C, D)
            if bit_indices.shape[1] == 1:
                # 假设是 (B, 1, H, W, D)，移除时间维度
                processed = bit_indices.squeeze(1)  # (B, H, W, D)
                B, H, W, D = processed.shape
            else:
                # 假设是 (B, H, W, C, D)，展平最后两个维度
                B, H, W, C, D = bit_indices.shape
                processed = bit_indices.reshape(B, H, W, C * D)
                D = C * D
                
        elif bit_indices.dim() == 6:
            # 形状: (B, T, H, W, C, D)
            B, T, H, W, C, D = bit_indices.shape
            if T == 1:
                # 移除时间维度并展平最后两个维度
                processed = bit_indices.squeeze(1).reshape(B, H, W, C * D)
                D = C * D
            else:
                raise ValueError(f"不支持的时间维度大小: T={T}")
                
        else:
            raise ValueError(f"不支持的bit_indices维度: {bit_indices.dim()}")
        
        # 现在processed的形状应该是 (B, H, W, D)
        assert processed.dim() == 4, f"处理后的维度应该是4，但得到{processed.dim()}"
        
        # 如果使用spatial patchify，需要进行相应的变换
        if self.vae_config['apply_spatial_patchify']:
            # 重排维度: (B, H, W, D) -> (B, D, H, W)
            processed = processed.permute(0, 3, 1, 2)
            
            # 应用pixel_unshuffle: (B, D, H, W) -> (B, 4*D, H/2, W/2)
            processed = torch.nn.functional.pixel_unshuffle(processed, 2)
            
            # 重排并reshape: (B, 4*D, H/2, W/2) -> (B, H/2*W/2, 4*D)
            processed = processed.permute(0, 2, 3, 1)
            final_token = processed.reshape(batch_size, -1, processed.shape[-1])
        else:
            # 不使用spatial patchify，直接reshape
            final_token = processed.reshape(batch_size, -1, D)
        
        # 转回原始设备
        final_token = final_token.to(device)
        
        print(f"    最终token形状: {final_token.shape}")
        return final_token
    
    def _apply_noise_correction(self, bit_indices, scale_index, noise_config):
        """
        应用噪声纠错
        
        Args:
            bit_indices: 原始bit索引
            scale_index: 当前尺度索引
            noise_config: 噪声配置
            
        Returns:
            corrected_bit_indices: 纠错后的bit索引
        """
        if scale_index >= noise_config['apply_noise_layers']:
            # 超出纠错层数范围，直接返回原始值
            return bit_indices.clone()
        
        # 应用噪声纠错
        corrected = bit_indices.clone()
        noise_strength = np.random.randint(
            0, 100 * noise_config['noise_strength'] + 1
        ) * 0.01
        
        mask = torch.rand(*corrected.shape).to(corrected.device) < noise_strength
        corrected[mask] = 1 - corrected[mask]  # 翻转bit
        
        return corrected
    
    def extract_multiscale_tokens(self, image_path, h_div_w=1.0, pn='0.25M', num_scales_to_keep=4):
        """
        提取多尺度token的主函数
        
        Args:
            image_path: 输入图像路径
            h_div_w: 宽高比
            pn: 像素数量级别
            num_scales_to_keep: 保留的尺度数量
            
        Returns:
            dict: 包含提取结果的字典
        """
        print(f"\n开始提取多尺度token...")
        print(f"输入图像: {image_path}")
        print(f"宽高比: {h_div_w}, 像素级别: {pn}")
        
        # 确定宽高比模板
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        print(f"使用的宽高比模板: {h_div_w_template}")
        
        # 获取尺度调度表
        scale_info = dynamic_resolution_h_w[h_div_w_template][pn]
        scale_schedule = scale_info['scales']
        target_h, target_w = scale_info['pixel']
        
        print(f"目标尺寸: {target_h} x {target_w}")
        print(f"尺度调度表: {scale_schedule}")
        
        # VAE使用的尺度调度表
        if self.vae_config['apply_spatial_patchify']:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        print(f"VAE尺度调度表: {vae_scale_schedule}")
        
        # 预处理图像
        inp_tensor = self._preprocess_image(image_path, target_h, target_w)
        
        # VAE编码获取原始特征
        print("\n第一步：VAE编码获取原始特征...")
        with torch.no_grad():
            raw_features, _, _ = self.vae.encode_for_raw_features(
                inp_tensor, 
                scale_schedule=vae_scale_schedule
            )
        print(f"原始特征形状: {raw_features.shape}")
        
        # 多尺度量化处理
        print("\n第二步：多尺度量化处理...")
        original_tokens, corrected_tokens = self._robust_multiscale_quantization(
            raw_features, vae_scale_schedule, NOISE_CONFIG
        )
        
        # 只保留前num_scales_to_keep个尺度
        num_scales_to_keep = min(num_scales_to_keep, len(original_tokens))
        
        result = {
            'original_tokens': original_tokens[:num_scales_to_keep],
            'corrected_tokens': corrected_tokens[:num_scales_to_keep],
            'scale_schedule': vae_scale_schedule[:num_scales_to_keep],
            'input_tensor': inp_tensor,
            'raw_features': raw_features,
            'metadata': {
                'image_path': image_path,
                'h_div_w_template': h_div_w_template,
                'target_size': (target_h, target_w),
                'num_scales': num_scales_to_keep,
                'vae_config': self.vae_config,
                'noise_config': NOISE_CONFIG
            }
        }
        
        print(f"\n成功提取前{num_scales_to_keep}个尺度的token！")
        return result
    
    def save_tokens(self, extraction_result, output_dir):
        """保存提取的token到文件"""
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
        
        # 保存详细信息为文本文件
        info_path = osp.join(output_dir, 'extraction_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"多尺度Token提取信息\n")
            f.write(f"{'='*40}\n")
            f.write(f"图像路径: {extraction_result['metadata']['image_path']}\n")
            f.write(f"目标尺寸: {extraction_result['metadata']['target_size']}\n")
            f.write(f"尺度数量: {extraction_result['metadata']['num_scales']}\n")
            f.write(f"VAE配置: {extraction_result['metadata']['vae_config']}\n")
            f.write(f"噪声配置: {extraction_result['metadata']['noise_config']}\n")
            f.write(f"\n尺度调度表:\n")
            for i, scale in enumerate(extraction_result['scale_schedule']):
                f.write(f"  尺度{i+1}: {scale}\n")
            f.write(f"\nToken形状信息:\n")
            for i, (orig_token, corr_token) in enumerate(zip(
                extraction_result['original_tokens'], 
                extraction_result['corrected_tokens']
            )):
                f.write(f"  尺度{i+1}:\n")
                f.write(f"    原始token: {orig_token.shape}\n")
                f.write(f"    纠正后token: {corr_token.shape}\n")
        print(f"提取信息已保存到: {info_path}")
    
    def visualize_tokens(self, extraction_result, output_dir):
        """可视化token和解码后的图片"""
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
            print(f"正在可视化尺度 {scale_idx + 1}/{len(original_tokens)}...")
            
            # 创建该尺度的目录
            scale_dir = osp.join(vis_dir, f'scale_{scale_idx+1}')
            os.makedirs(scale_dir, exist_ok=True)
            
            # 可视化当前尺度的解码结果
            self._visualize_single_scale(
                original_tokens[:scale_idx+1], 
                corrected_tokens[:scale_idx+1],
                scale_schedule[:scale_idx+1],
                scale_dir, 
                scale_idx + 1
            )
        
        # 创建完整的对比可视化
        self._create_full_comparison(
            original_tokens, corrected_tokens, scale_schedule, vis_dir
        )
        
        print(f"可视化完成，结果保存在: {vis_dir}")
    
    def _visualize_single_scale(self, orig_tokens, corr_tokens, scale_schedule, output_dir, scale_num):
        """可视化单个尺度的结果"""
        try:
            with torch.no_grad():
                # 解码原始token
                _, orig_images = self.vae.decode_from_indices(
                    orig_tokens, scale_schedule, label_type='bit_label'
                )
                
                # 解码纠正后token
                _, corr_images = self.vae.decode_from_indices(
                    corr_tokens, scale_schedule, label_type='bit_label'
                )
                
                # 保存解码图像
                if len(orig_images) > 0:
                    self._save_tensor_as_image(
                        orig_images[0:1], 
                        osp.join(output_dir, f'original_decoded_up_to_scale_{scale_num}.jpg'),
                        f"原始Token解码结果（前{scale_num}尺度）"
                    )
                
                if len(corr_images) > 0:
                    self._save_tensor_as_image(
                        corr_images[0:1], 
                        osp.join(output_dir, f'corrected_decoded_up_to_scale_{scale_num}.jpg'),
                        f"纠正后Token解码结果（前{scale_num}尺度）"
                    )
                    
        except Exception as e:
            print(f"  - 可视化尺度{scale_num}时出错: {e}")
    
    def _create_full_comparison(self, original_tokens, corrected_tokens, scale_schedule, output_dir):
        """创建完整的对比可视化"""
        print("正在创建完整对比可视化...")
        
        try:
            with torch.no_grad():
                # 解码所有原始token
                _, orig_full = self.vae.decode_from_indices(
                    original_tokens, scale_schedule, label_type='bit_label'
                )
                
                # 解码所有纠正后token
                _, corr_full = self.vae.decode_from_indices(
                    corrected_tokens, scale_schedule, label_type='bit_label'
                )
                
                # 保存完整解码结果
                if len(orig_full) > 0:
                    self._save_tensor_as_image(
                        orig_full[0:1], 
                        osp.join(output_dir, 'full_original_reconstruction.jpg'),
                        "完整原始Token重建结果"
                    )
                
                if len(corr_full) > 0:
                    self._save_tensor_as_image(
                        corr_full[0:1], 
                        osp.join(output_dir, 'full_corrected_reconstruction.jpg'),
                        "完整纠正后Token重建结果"
                    )
                    
        except Exception as e:
            print(f"  - 创建完整对比时出错: {e}")
    
    def _save_tensor_as_image(self, tensor, filepath, title=""):
        """将tensor保存为图像文件"""
        if tensor.dim() == 4:
            # tensor格式: (B, C, H, W)，值域[-1, 1]
            img = tensor[0]  # 取第一个batch
            img = (img + 1) / 2  # 转换到[0, 1]
            img = img.permute(1, 2, 0)  # 转换到(H, W, C)
            img = torch.clamp(img, 0, 1)
            
            # 转换为numpy并保存
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            
            # RGB转BGR（OpenCV格式）
            if img_np.shape[2] == 3:
                img_np = img_np[:, :, ::-1]
            
            cv2.imwrite(filepath, img_np)
            print(f"  - 图像已保存: {filepath}")


def main():
    """主函数"""
    print("=" * 60)
    print("多尺度Token提取和可视化工具 - 健壮版本")
    print("=" * 60)
    
    # 检查文件
    if not osp.exists(IMAGE_PATH):
        print(f"错误：输入图像不存在: {IMAGE_PATH}")
        print("请修改IMAGE_PATH变量")
        return
    
    if not osp.exists(VAE_MODEL_PATH):
        print(f"错误：VAE模型不存在: {VAE_MODEL_PATH}")
        print("请修改VAE_MODEL_PATH变量")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化提取器
    extractor = RobustTokenExtractor(
        vae_model_path=VAE_MODEL_PATH,
        vae_config=VAE_CONFIG
    )
    
    # 提取多尺度token
    start_time = time.time()
    extraction_result = extractor.extract_multiscale_tokens(
        image_path=IMAGE_PATH,
        h_div_w=1.0,
        pn='0.25M',  # 对应512x512分辨率
        num_scales_to_keep=4  # 保留前4个尺度
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
    
    # 打印统计信息
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
