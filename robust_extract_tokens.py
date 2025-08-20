#!/usr/bin/env python3
"""
多尺度Token提取和可视化工具 - 最终健壮版本

这个版本彻底解决了维度处理问题，核心改进：
1. 正确处理图像的时间维度（强制为1）
2. 智能的维度检测和处理逻辑
3. 完善的错误处理和调试信息
4. 符合训练时逻辑的token提取流程

使用方法：
    python extract_multiscale_tokens_final.py
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
from torchvision.transforms.functional import to_tensor

# 添加项目路径
sys.path.append('.')

from infinity.models.bsq_vae.vae import vae_model
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


# ==================== 配置参数 ====================
# 只需要修改这两个路径
IMAGE_PATH = "000_turn_left_LeftView.png"  # 修改为您的图片路径
VAE_MODEL_PATH = "weights/infinity_vae_d56_f8_14_patchify.pth"  # VAE模型路径
OUTPUT_DIR = "output/extracted_tokens_final"  # 输出目录

# VAE配置（匹配您的配置）
VAE_CONFIG = {
    'vae_type': 14,  # codebook_dim=14
    'apply_spatial_patchify': 1,  # 使用spatial patchify
    'schedule_mode': "dynamic",
    'patch_size': 8,
    'encoder_ch_mult': [1, 2, 4, 4],
    'decoder_ch_mult': [1, 2, 4, 4],
}

# 纠错配置
CORRECTION_CONFIG = {
    'apply_layers': 4,      # 对前4层应用纠错
    'noise_strength': 0.1,  # 噪声强度
}
# ===================================================


class FinalTokenExtractor:
    """最终版本的多尺度Token提取器"""
    
    def __init__(self, vae_model_path, vae_config):
        """初始化提取器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae_config = vae_config
        
        print(f"正在初始化最终版TokenExtractor...")
        print(f"设备: {self.device}")
        
        # 加载VAE模型
        self.vae = self._load_vae_model(vae_model_path, vae_config)
        
        print("TokenExtractor初始化完成！")
    
    def _load_vae_model(self, model_path, config):
        """加载VAE模型"""
        print(f"正在加载VAE模型: {model_path}")
        
        codebook_dim = config['vae_type']
        codebook_size = 2 ** codebook_dim
        
        # 根据配置设置参数
        if config['apply_spatial_patchify']:
            patch_size = 8
            encoder_ch_mult = [1, 2, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4]
        else:
            patch_size = config['patch_size']
            encoder_ch_mult = config['encoder_ch_mult']
            decoder_ch_mult = config['decoder_ch_mult']
        
        # 创建VAE模型
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
    
    def _preprocess_image(self, image_path, target_size=512):
        """
        图像预处理：转换为512x512
        
        使用与训练时相同的预处理策略：
        1. 保持宽高比进行缩放
        2. 中心裁剪到目标尺寸
        3. 归一化到[-1, 1]范围
        """
        print(f"正在预处理图像: {image_path}")
        
        # 加载并转换为RGB
        pil_image = Image.open(image_path).convert('RGB')
        width, height = pil_image.size
        print(f"原始图像尺寸: {width} x {height}")
        
        # 计算缩放比例（使较长边适配目标尺寸）
        scale_factor = max(target_size / width, target_size / height)
        
        # 按比例缩放
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        print(f"缩放后尺寸: {new_width} x {new_height}")
        
        pil_image = pil_image.resize((new_width, new_height), resample=Image.LANCZOS)
        
        # 转换为numpy进行裁剪
        arr = np.array(pil_image)
        
        # 中心裁剪
        crop_y = max(0, (arr.shape[0] - target_size) // 2)
        crop_x = max(0, (arr.shape[1] - target_size) // 2)
        
        crop_y_end = min(arr.shape[0], crop_y + target_size)
        crop_x_end = min(arr.shape[1], crop_x + target_size)
        
        cropped_arr = arr[crop_y:crop_y_end, crop_x:crop_x_end]
        
        # 如果需要，进行填充
        if cropped_arr.shape[0] < target_size or cropped_arr.shape[1] < target_size:
            padded_arr = np.ones((target_size, target_size, 3), dtype=np.uint8) * 128
            
            pad_y = (target_size - cropped_arr.shape[0]) // 2
            pad_x = (target_size - cropped_arr.shape[1]) // 2
            
            padded_arr[pad_y:pad_y + cropped_arr.shape[0], 
                      pad_x:pad_x + cropped_arr.shape[1]] = cropped_arr
            
            cropped_arr = padded_arr
            print(f"图像已填充到: {target_size} x {target_size}")
        
        # 转换为tensor并归一化
        im = to_tensor(cropped_arr)  # 转为[0, 1]
        im = im * 2.0 - 1.0          # 转为[-1, 1]
        
        print(f"预处理完成: {target_size} x {target_size}")
        print(f"tensor形状: {im.shape}, 数值范围: [{im.min():.3f}, {im.max():.3f}]")
        
        return im.unsqueeze(0).to(self.device)  # 添加batch维度
    
    def _create_image_scale_schedule(self, base_schedule):
        """
        为图像处理创建正确的尺度调度表
        
        关键修改：将所有时间维度设为1（因为我们处理的是图像，不是视频）
        
        Args:
            base_schedule: 原始尺度调度表 [(t, h, w), ...]
            
        Returns:
            image_schedule: 图像处理的尺度调度表 [(1, h, w), ...]
        """
        # 将所有时间维度设为1，只保留空间维度信息
        image_schedule = [(1, h, w) for (t, h, w) in base_schedule]
        
        print(f"原始调度表: {base_schedule[:5]}...")  # 只显示前5个
        print(f"图像调度表: {image_schedule[:5]}...")  # 只显示前5个
        
        return image_schedule
    
    def _extract_tokens_from_vae(self, inp_tensor, scale_schedule, correction_config):
        """
        使用VAE提取多尺度token
        
        这个方法直接使用VAE的encode方法，避免了手动实现量化逻辑
        """
        print("正在使用VAE提取多尺度token...")
        
        with torch.no_grad():
            # 使用VAE的encode方法进行编码和量化
            # 这个方法会返回完整的量化结果
            h, z, all_indices, all_bit_indices, residual_norm_per_scale, var_input = self.vae.encode(
                inp_tensor, 
                scale_schedule=scale_schedule, 
                return_residual_norm_per_scale=True
            )
            
            print(f"VAE编码完成:")
            print(f"  - 原始特征形状: {h.shape}")
            print(f"  - 量化特征形状: {z.shape}")
            print(f"  - 提取到 {len(all_bit_indices)} 个尺度的bit indices")
            
            # 处理bit indices，转换为统一格式
            original_tokens = self._process_all_bit_indices(all_bit_indices, "原始")
            
            # 生成纠错后的token
            corrected_tokens = self._apply_correction_to_tokens(
                original_tokens, correction_config
            )
            
            return original_tokens, corrected_tokens, h, z
    
    def _process_all_bit_indices(self, all_bit_indices, token_type):
        """
        处理所有尺度的bit indices
        
        这个方法统一处理不同尺度的bit indices，
        确保输出格式的一致性
        """
        print(f"正在处理{token_type}bit indices...")
        
        processed_tokens = []
        
        for scale_idx, bit_indices in enumerate(all_bit_indices):
            print(f"  处理尺度 {scale_idx + 1}: 原始形状 {bit_indices.shape}")
            
            # 统一处理为 (B, seq_len, feature_dim) 格式
            processed_token = self._standardize_bit_indices(bit_indices, scale_idx)
            processed_tokens.append(processed_token)
            
            print(f"    处理后形状: {processed_token.shape}")
        
        return processed_tokens
    
    def _standardize_bit_indices(self, bit_indices, scale_idx):
        """
        标准化bit indices为统一格式
        
        无论输入的维度如何，都转换为 (B, seq_len, feature_dim) 格式
        """
        original_shape = bit_indices.shape
        device = bit_indices.device
        
        # 移到CPU进行维度操作（避免某些CUDA操作的限制）
        bit_indices = bit_indices.cpu()
        
        # 根据维度数量进行处理
        if bit_indices.dim() == 4:
            # 形状: (B, H, W, D)
            B, H, W, D = bit_indices.shape
            result = bit_indices.reshape(B, H * W, D)
            
        elif bit_indices.dim() == 5:
            B, T, H, W, D = bit_indices.shape
            
            # 对于图像处理，我们期望T=1
            if T == 1:
                # 移除时间维度: (B, 1, H, W, D) -> (B, H, W, D)
                result = bit_indices.squeeze(1).reshape(B, H * W, D)
            else:
                # 如果T!=1，我们将其视为额外的空间维度
                # 这种情况下，我们将T*H*W展平为序列长度
                result = bit_indices.reshape(B, T * H * W, D)
                
        elif bit_indices.dim() == 6:
            B, T, H, W, C, D = bit_indices.shape
            # 将所有空间和时间维度展平，合并最后两个特征维度
            result = bit_indices.reshape(B, T * H * W, C * D)
            
        else:
            raise ValueError(f"不支持的bit_indices维度数: {bit_indices.dim()}")
        
        # 如果需要应用spatial patchify，进行相应处理
        if self.vae_config['apply_spatial_patchify'] and bit_indices.dim() >= 4:
            result = self._apply_spatial_patchify_to_token(result, original_shape)
        
        # 转回原设备
        result = result.to(device)
        
        return result
    
    def _apply_spatial_patchify_to_token(self, token, original_shape):
        """
        对token应用spatial patchify变换
        
        这个方法谨慎地处理spatial patchify，
        确保维度兼容性
        """
        try:
            B, seq_len, D = token.shape
            
            # 尝试从原始形状推断空间维度
            if len(original_shape) >= 4:
                if len(original_shape) == 5:
                    _, T, H, W, _ = original_shape
                    if T == 1:
                        spatial_H, spatial_W = H, W
                    else:
                        # 如果T!=1，我们不应用spatial patchify
                        return token
                else:
                    _, H, W, _ = original_shape
                    spatial_H, spatial_W = H, W
                
                # 检查spatial patchify的前提条件
                if spatial_H % 2 == 0 and spatial_W % 2 == 0:
                    # 重塑为空间格式
                    token_spatial = token.reshape(B, spatial_H, spatial_W, D)
                    
                    # 变换维度并应用pixel_unshuffle
                    token_spatial = token_spatial.permute(0, 3, 1, 2)  # (B, D, H, W)
                    token_spatial = F.pixel_unshuffle(token_spatial, 2)  # (B, 4D, H/2, W/2)
                    
                    # 转回序列格式
                    _, new_D, new_H, new_W = token_spatial.shape
                    token = token_spatial.permute(0, 2, 3, 1).reshape(B, new_H * new_W, new_D)
                else:
                    print(f"    警告: 空间维度 ({spatial_H}, {spatial_W}) 不适合pixel_unshuffle，跳过spatial patchify")
            
        except Exception as e:
            print(f"    应用spatial patchify时出错: {e}，使用原始token")
        
        return token
    
    def _apply_correction_to_tokens(self, original_tokens, correction_config):
        """
        对token应用纠错处理
        
        这个方法在token级别应用纠错，而不是在量化级别
        """
        print("正在应用纠错处理...")
        
        corrected_tokens = []
        
        for scale_idx, token in enumerate(original_tokens):
            if scale_idx < correction_config['apply_layers']:
                # 应用纠错
                corrected_token = self._apply_token_noise(token, correction_config['noise_strength'])
                print(f"  尺度 {scale_idx + 1}: 已应用纠错")
            else:
                # 不应用纠错
                corrected_token = token.clone()
                print(f"  尺度 {scale_idx + 1}: 未应用纠错")
            
            corrected_tokens.append(corrected_token)
        
        return corrected_tokens
    
    def _apply_token_noise(self, token, noise_strength):
        """
        对token应用噪声纠错
        
        由于token是bit级别的（0或1），我们随机翻转一些bit
        """
        corrected = token.clone()
        
        # 生成随机强度
        actual_strength = np.random.uniform(0, noise_strength)
        
        # 创建噪声掩码
        noise_mask = torch.rand_like(corrected.float()) < actual_strength
        
        # 翻转被掩码的bit
        corrected[noise_mask] = 1 - corrected[noise_mask]
        
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
        
        # 获取尺度调度表
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_info = dynamic_resolution_h_w[h_div_w_template][pn]
        base_scale_schedule = scale_info['scales']
        target_h, target_w = scale_info['pixel']
        
        print(f"使用的宽高比模板: {h_div_w_template}")
        print(f"目标尺寸: {target_h} x {target_w}")
        
        # 创建适合图像处理的尺度调度表
        image_scale_schedule = self._create_image_scale_schedule(base_scale_schedule)
        
        # 如果使用spatial patchify，调整VAE的尺度调度表
        if self.vae_config['apply_spatial_patchify']:
            vae_scale_schedule = [(t, 2*h, 2*w) for (t, h, w) in image_scale_schedule]
            print(f"VAE尺度调度表（spatial patchify）: {vae_scale_schedule[:3]}...")
        else:
            vae_scale_schedule = image_scale_schedule
            print(f"VAE尺度调度表: {vae_scale_schedule[:3]}...")
        
        # 预处理图像
        inp_tensor = self._preprocess_image(image_path, target_size=512)
        
        # 提取token
        original_tokens, corrected_tokens, raw_features, quantized_features = self._extract_tokens_from_vae(
            inp_tensor, vae_scale_schedule, CORRECTION_CONFIG
        )
        
        # 限制保留的尺度数量
        num_scales_to_keep = min(num_scales_to_keep, len(original_tokens))
        
        result = {
            'original_tokens': original_tokens[:num_scales_to_keep],
            'corrected_tokens': corrected_tokens[:num_scales_to_keep],
            'scale_schedule': vae_scale_schedule[:num_scales_to_keep],
            'input_tensor': inp_tensor,
            'raw_features': raw_features,
            'quantized_features': quantized_features,
            'metadata': {
                'image_path': image_path,
                'h_div_w_template': h_div_w_template,
                'target_size': (target_h, target_w),
                'num_scales': num_scales_to_keep,
                'vae_config': self.vae_config,
                'correction_config': CORRECTION_CONFIG
            }
        }
        
        print(f"\n成功提取前{num_scales_to_keep}个尺度的token！")
        return result
    
    def save_tokens(self, extraction_result, output_dir):
        """保存提取的token到文件"""
        print(f"\n正在保存token到: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始token
        original_path = osp.join(output_dir, 'original_tokens.pth')
        torch.save(extraction_result['original_tokens'], original_path)
        print(f"原始token已保存: {original_path}")
        
        # 保存纠正后token
        corrected_path = osp.join(output_dir, 'corrected_tokens.pth')
        torch.save(extraction_result['corrected_tokens'], corrected_path)
        print(f"纠正后token已保存: {corrected_path}")
        
        # 保存原始特征
        raw_features_path = osp.join(output_dir, 'raw_features.pth')
        torch.save(extraction_result['raw_features'], raw_features_path)
        print(f"原始特征已保存: {raw_features_path}")
        
        # 保存元数据
        metadata_path = osp.join(output_dir, 'metadata.pth')
        torch.save(extraction_result['metadata'], metadata_path)
        print(f"元数据已保存: {metadata_path}")
        
        # 创建可读的信息文件
        info_path = osp.join(output_dir, 'extraction_info.txt')
        with open(info_path, 'w') as f:
            f.write("多尺度Token提取结果\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"输入图像: {extraction_result['metadata']['image_path']}\n")
            f.write(f"目标尺寸: {extraction_result['metadata']['target_size']}\n")
            f.write(f"提取尺度数: {extraction_result['metadata']['num_scales']}\n")
            f.write(f"VAE配置: {extraction_result['metadata']['vae_config']}\n")
            f.write(f"纠错配置: {extraction_result['metadata']['correction_config']}\n\n")
            
            f.write("各尺度Token信息:\n")
            f.write("-" * 20 + "\n")
            for i, (orig, corr) in enumerate(zip(
                extraction_result['original_tokens'], 
                extraction_result['corrected_tokens']
            )):
                f.write(f"尺度 {i+1}:\n")
                f.write(f"  原始token形状: {orig.shape}\n")
                f.write(f"  纠正后token形状: {corr.shape}\n")
                f.write(f"  尺度参数: {extraction_result['scale_schedule'][i]}\n\n")
        
        print(f"提取信息已保存: {info_path}")
    
    def visualize_tokens(self, extraction_result, output_dir):
        """可视化token和解码结果"""
        print(f"\n正在进行可视化...")
        
        vis_dir = osp.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 保存原始输入图像
        self._save_tensor_as_image(
            extraction_result['input_tensor'], 
            osp.join(vis_dir, 'original_input.jpg')
        )
        
        # 可视化每个尺度的累积效果
        self._visualize_progressive_reconstruction(
            extraction_result['original_tokens'],
            extraction_result['corrected_tokens'],
            extraction_result['scale_schedule'],
            vis_dir
        )
        
        print(f"可视化完成，结果保存在: {vis_dir}")
    
    def _visualize_progressive_reconstruction(self, original_tokens, corrected_tokens, scale_schedule, output_dir):
        """可视化渐进式重建效果"""
        print("正在生成渐进式重建可视化...")
        
        for i in range(len(original_tokens)):
            scale_num = i + 1
            print(f"  可视化前{scale_num}个尺度的重建效果...")
            
            try:
                # 使用前i+1个尺度进行重建
                current_original = original_tokens[:scale_num]
                current_corrected = corrected_tokens[:scale_num]
                current_schedule = scale_schedule[:scale_num]
                
                with torch.no_grad():
                    # 重建原始token
                    _, orig_img = self.vae.decode_from_indices(
                        current_original, current_schedule, label_type='bit_label'
                    )
                    
                    # 重建纠正后token
                    _, corr_img = self.vae.decode_from_indices(
                        current_corrected, current_schedule, label_type='bit_label'
                    )
                    
                    # 保存重建图像
                    if len(orig_img) > 0:
                        self._save_tensor_as_image(
                            orig_img[0:1], 
                            osp.join(output_dir, f'reconstruction_original_scale_{scale_num}.jpg')
                        )
                    
                    if len(corr_img) > 0:
                        self._save_tensor_as_image(
                            corr_img[0:1], 
                            osp.join(output_dir, f'reconstruction_corrected_scale_{scale_num}.jpg')
                        )
                        
            except Exception as e:
                print(f"    可视化尺度{scale_num}时出错: {e}")
    
    def _save_tensor_as_image(self, tensor, filepath):
        """将tensor保存为图像文件"""
        if tensor.dim() == 4:
            # tensor格式: (B, C, H, W)，值域[-1, 1]
            img = tensor[0]  # 取第一个batch
            img = torch.clamp((img + 1) / 2, 0, 1)  # 转换到[0, 1]并裁剪
            img = img.permute(1, 2, 0)  # 转换到(H, W, C)
            
            # 转换为numpy并保存
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            
            # RGB转BGR（OpenCV格式）
            if img_np.shape[2] == 3:
                img_np = img_np[:, :, ::-1]
            
            cv2.imwrite(filepath, img_np)
            print(f"    图像已保存: {osp.basename(filepath)}")


def main():
    """主函数"""
    print("=" * 60)
    print("多尺度Token提取和可视化工具 - 最终版本")
    print("=" * 60)
    
    # 检查输入文件
    if not osp.exists(IMAGE_PATH):
        print(f"错误：输入图像不存在: {IMAGE_PATH}")
        print("请修改IMAGE_PATH变量为正确的图像路径")
        return
    
    if not osp.exists(VAE_MODEL_PATH):
        print(f"错误：VAE模型不存在: {VAE_MODEL_PATH}")
        print("请修改VAE_MODEL_PATH变量为正确的模型路径")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 初始化提取器
        extractor = FinalTokenExtractor(
            vae_model_path=VAE_MODEL_PATH,
            vae_config=VAE_CONFIG
        )
        
        # 提取多尺度token
        print(f"\n开始token提取流程...")
        start_time = time.time()
        
        extraction_result = extractor.extract_multiscale_tokens(
            image_path=IMAGE_PATH,
            h_div_w=1.0,      # 1:1宽高比
            pn='0.25M',       # 对应512x512分辨率
            num_scales_to_keep=4  # 保留前4个尺度
        )
        
        extraction_time = time.time() - start_time
        print(f"\nToken提取完成，耗时: {extraction_time:.2f}秒")
        
        # 保存结果
        extractor.save_tokens(extraction_result, OUTPUT_DIR)
        
        # 生成可视化
        extractor.visualize_tokens(extraction_result, OUTPUT_DIR)
        
        # 输出总结信息
        print("\n" + "=" * 60)
        print("处理完成！")
        print(f"结果保存在: {OUTPUT_DIR}")
        print("=" * 60)
        
        print(f"\n提取结果总结:")
        print(f"- 输入图像: {IMAGE_PATH}")
        print(f"- 目标尺寸: {extraction_result['metadata']['target_size']}")
        print(f"- 提取尺度数: {len(extraction_result['original_tokens'])}")
        print(f"- 原始特征形状: {extraction_result['raw_features'].shape}")
        
        for i, (orig_token, corr_token) in enumerate(zip(
            extraction_result['original_tokens'], 
            extraction_result['corrected_tokens']
        )):
            print(f"- 尺度{i+1} Token形状: {orig_token.shape}")
        
        print(f"\n输出文件:")
        print(f"├── original_tokens.pth      # 原始token（前4尺度）")
        print(f"├── corrected_tokens.pth     # 纠正后token（前4尺度）")
        print(f"├── raw_features.pth         # 原始连续特征")
        print(f"├── metadata.pth             # 元数据信息")
        print(f"├── extraction_info.txt      # 可读的提取信息")
        print(f"└── visualizations/          # 可视化结果目录")
        print(f"    ├── original_input.jpg")
        print(f"    ├── reconstruction_original_scale_*.jpg")
        print(f"    └── reconstruction_corrected_scale_*.jpg")
        
    except Exception as e:
        print(f"\n执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n如果问题持续存在，请检查：")
        print("1. 图像文件是否存在且可读")
        print("2. VAE模型文件是否存在且格式正确")
        print("3. CUDA内存是否充足")


if __name__ == "__main__":
    main()
