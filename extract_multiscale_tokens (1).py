"""
Multi-scale Visual Token Extractor for Infinity Model

这个脚本的核心功能是从输入图片中提取前4个scale的两种视觉表征：
1. 原始token（gt_tokens）：未经扰动的真实视觉表征
2. 纠正后token（corrected_tokens）：经过bitwise self-correction处理的表征

使用方法：
1. 修改IMAGE_PATH变量为您要分析的图片路径
2. 运行: python extract_multiscale_tokens.py
3. 查看输出目录中保存的token文件和可视化结果
"""

import os
import random
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np
from tools.run_infinity import *
from PIL import Image
import torch.nn.functional as F
from datetime import datetime

# ===================== 配置部分 =====================
# 只需要修改这个路径！
IMAGE_PATH = "path/to/your/image.jpg"  # 修改为您的图片路径

# 模型配置（与inf.py保持一致）
model_path = 'weights/infinity_8b_512x512_weights'
vae_path = 'weights/infinity_vae_d56_f8_14_patchify.pth'
text_encoder_ckpt = 'weights/flan-t5-xl-official'

# 输出目录
OUTPUT_DIR = "extracted_tokens"

# ===================== 模型配置 =====================
args = argparse.Namespace(
    pn='0.25M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=14,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_8b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=1,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch_shard',
    seed=0,
    bf16=1,
)

class MultiscaleTokenExtractor:
    """
    多尺度token提取器
    
    这个类封装了从图片到多尺度视觉表征的完整流程，
    包括预处理、VAE编码、Bitwise Self-Correction等步骤。
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae = None
        self.scale_schedule = None
        self.vae_scale_schedule = None
        
        # BSC相关参数
        self.noise_apply_layers = 4  # 前4层都可能添加噪声
        self.noise_apply_strength = 0.15  # 噪声强度
        
    def load_model(self):
        """加载VAE模型"""
        print("正在加载VAE模型...")
        self.vae = load_visual_tokenizer(self.args)
        print(f"VAE加载完成。Codebook维度: {self.vae.codebook_dim}")
        
        # 获取scale_schedule
        h_div_w_template = self.args.h_div_w_template
        self.scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.args.pn]['scales']
        self.scale_schedule = [(1, h, w) for (_, h, w) in self.scale_schedule]
        
        # 如果使用spatial_patchify，转换为vae_scale_schedule
        if self.args.apply_spatial_patchify:
            self.vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in self.scale_schedule]
        else:
            self.vae_scale_schedule = self.scale_schedule
            
        print(f"Scale schedule: {self.scale_schedule[:6]}")
        print(f"VAE scale schedule: {self.vae_scale_schedule[:6]}")
        
    def preprocess_image(self, image_path):
        """
        图片预处理：将任意尺寸的图片转换为512x512
        
        处理流程：
        1. 加载图片
        2. 智能resize（保持宽高比）
        3. Center crop到512x512
        4. 归一化到[-1, 1]范围
        """
        print(f"正在预处理图片: {image_path}")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 加载图片
        pil_image = Image.open(image_path).convert('RGB')
        original_size = pil_image.size
        print(f"原始图片尺寸: {original_size[0]}x{original_size[1]}")
        
        # 使用transform函数进行预处理
        tgt_h, tgt_w = 512, 512
        transformed = transform(pil_image, tgt_h, tgt_w)
        
        # 添加batch维度并移到GPU
        inp_B3HW = transformed.unsqueeze(0).to(self.device)
        print(f"预处理后形状: {inp_B3HW.shape}")
        print(f"数值范围: [{inp_B3HW.min():.3f}, {inp_B3HW.max():.3f}]")
        
        return inp_B3HW, pil_image
        
    def extract_raw_features(self, inp_B3HW):
        """
        使用VAE提取原始特征
        
        这一步将图片编码为连续的特征表示，
        作为后续多尺度量化的输入。
        """
        print("\n正在提取VAE原始特征...")
        
        with torch.no_grad():
            raw_features, _, _ = self.vae.encode_for_raw_features(
                inp_B3HW, scale_schedule=self.vae_scale_schedule
            )
        
        print(f"Raw features形状: {raw_features.shape}")
        print(f"Raw features数值范围: [{raw_features.min():.3f}, {raw_features.max():.3f}]")
        
        return raw_features
        
    def extract_multiscale_tokens(self, raw_features):
        """
        提取多尺度token表征
        
        这是整个流程的核心部分，实现了Bitwise Self-Correction机制：
        1. 逐尺度处理residual特征
        2. 对每个尺度进行BSQ量化
        3. 在前几个尺度添加噪声模拟推理错误
        4. 累积各尺度的贡献
        
        返回两种token：
        - gt_tokens: 真实的、未扰动的token
        - corrected_tokens: 经过self-correction的token
        """
        print("\n正在提取多尺度token...")
        
        B = raw_features.shape[0]
        
        # 处理维度：确保是5D tensor (B, C, T, H, W)
        if raw_features.dim() == 4:
            codes_out = raw_features.unsqueeze(2)  # 添加时间维度
        else:
            codes_out = raw_features
            
        print(f"初始codes_out形状: {codes_out.shape}")
        
        # 初始化累积变量和结果列表
        cum_var_input = 0
        gt_tokens = []  # 真实token
        corrected_tokens = []  # 纠正后token
        
        with torch.no_grad():
            # 逐尺度处理（只处理前4个尺度）
            for si, (pt, ph, pw) in enumerate(self.vae_scale_schedule[:4]):
                print(f"\n--- 处理Scale {si}: ({pt}, {ph}, {pw}) ---")
                
                # 计算当前尺度的residual
                residual = codes_out - cum_var_input
                
                # 如果不是最后一个尺度，需要插值到当前尺度的分辨率
                if si < len(self.vae_scale_schedule) - 1:
                    residual = F.interpolate(
                        residual, 
                        size=(pt, ph, pw), 
                        mode=self.vae.quantizer.z_interplote_down
                    ).contiguous()
                
                print(f"Residual形状: {residual.shape}")
                
                # BSQ量化得到真实token
                quantized, _, bit_indices, _ = self.vae.quantizer.lfq(residual)
                gt_tokens.append(bit_indices.clone())
                
                print(f"Scale {si} token形状: {bit_indices.shape}")
                print(f"Token数量: {bit_indices.numel() // bit_indices.shape[-1]} tokens，每个token {bit_indices.shape[-1]} bits")
                
                # 生成纠正后的token（模拟bitwise self-correction）
                corrected_bit_indices = bit_indices.clone()
                
                # 在前几个尺度添加噪声
                if si < self.noise_apply_layers:
                    # 随机选择要翻转的bits
                    noise_strength = self.noise_apply_strength * (1.0 - si * 0.05)  # 尺度越高噪声越少
                    mask = torch.rand_like(bit_indices, dtype=torch.float) < noise_strength
                    corrected_bit_indices[mask] = 1 - corrected_bit_indices[mask]
                    
                    flipped_bits = mask.sum().item()
                    total_bits = bit_indices.numel()
                    print(f"添加噪声：翻转了 {flipped_bits}/{total_bits} bits ({flipped_bits/total_bits*100:.2f}%)")
                else:
                    print("未添加噪声（尺度过高）")
                
                corrected_tokens.append(corrected_bit_indices)
                
                # 累积当前尺度的贡献
                cum_var_input = cum_var_input + F.interpolate(
                    quantized, 
                    size=self.vae_scale_schedule[-1][::-1],  # (W, H, T) -> (T, H, W)
                    mode=self.vae.quantizer.z_interplote_up
                ).contiguous()
        
        print(f"\n提取完成！共提取了 {len(gt_tokens)} 个尺度的token")
        return gt_tokens, corrected_tokens
    
    def reconstruct_images(self, tokens_list, token_type=""):
        """
        从token重构图片
        
        使用VAE的decoder将每个尺度的token重构为图片，
        这有助于可视化不同尺度捕获的视觉信息。
        """
        print(f"\n正在重构{token_type}图片...")
        
        reconstructed_images = []
        
        with torch.no_grad():
            for si, tokens in enumerate(tokens_list):
                try:
                    # 将bit indices转换为codes
                    codes = self.vae.quantizer.lfq.indices_to_codes(
                        tokens, label_type='bit_label'
                    )
                    
                    # 插值到最终尺寸并解码
                    final_size = self.vae_scale_schedule[-1]
                    codes_upsampled = F.interpolate(
                        codes, 
                        size=final_size, 
                        mode=self.vae.quantizer.z_interplote_up
                    )
                    
                    # VAE解码
                    if codes_upsampled.dim() == 5:
                        codes_upsampled = codes_upsampled.squeeze(2)  # 移除时间维度
                    
                    reconstructed = self.vae.decode(codes_upsampled)
                    
                    # 转换为可显示的图片格式
                    img = (reconstructed + 1) / 2  # [-1,1] -> [0,1]
                    img = img.clamp(0, 1)
                    img = img.squeeze(0).permute(1, 2, 0)  # CHW -> HWC
                    img = (img * 255).cpu().numpy().astype(np.uint8)
                    
                    reconstructed_images.append(img)
                    print(f"Scale {si} 重构图片形状: {img.shape}")
                    
                except Exception as e:
                    print(f"Scale {si} 重构失败: {e}")
                    # 创建空白图片作为占位符
                    reconstructed_images.append(np.zeros((512, 512, 3), dtype=np.uint8))
        
        return reconstructed_images
    
    def save_tokens(self, gt_tokens, corrected_tokens, output_dir, image_name):
        """保存提取的token到文件"""
        print(f"\n正在保存token到 {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存token数据
        token_data = {
            'gt_tokens': gt_tokens,
            'corrected_tokens': corrected_tokens,
            'scale_schedule': self.scale_schedule[:4],
            'vae_scale_schedule': self.vae_scale_schedule[:4],
            'image_name': image_name,
            'extraction_time': datetime.now().isoformat(),
            'model_config': {
                'vae_type': self.args.vae_type,
                'apply_spatial_patchify': self.args.apply_spatial_patchify,
                'codebook_dim': self.vae.codebook_dim,
            }
        }
        
        token_file = os.path.join(output_dir, f"{image_name}_tokens.pt")
        torch.save(token_data, token_file)
        print(f"Token数据已保存到: {token_file}")
        
        # 保存token统计信息
        stats_file = os.path.join(output_dir, f"{image_name}_stats.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Multi-scale Token Extraction Statistics\n")
            f.write(f"========================================\n\n")
            f.write(f"Image: {image_name}\n")
            f.write(f"Extraction Time: {datetime.now()}\n")
            f.write(f"Model: Infinity-8B (512x512)\n")
            f.write(f"VAE Type: {self.args.vae_type}\n")
            f.write(f"Codebook Dimension: {self.vae.codebook_dim}\n")
            f.write(f"Spatial Patchify: {self.args.apply_spatial_patchify}\n\n")
            
            f.write("Scale Information:\n")
            for i, (gt_token, corrected_token) in enumerate(zip(gt_tokens, corrected_tokens)):
                scale_info = self.vae_scale_schedule[i]
                token_count = gt_token.shape[1] * gt_token.shape[2] * gt_token.shape[3]
                bit_count = token_count * gt_token.shape[-1]
                
                f.write(f"Scale {i}: {scale_info}\n")
                f.write(f"  Token形状: {gt_token.shape}\n")
                f.write(f"  Token数量: {token_count}\n")
                f.write(f"  总bit数: {bit_count}\n")
                f.write(f"  每token bit数: {gt_token.shape[-1]}\n\n")
        
        print(f"统计信息已保存到: {stats_file}")
        
    def add_token_info_to_image(self, img, scale_idx, token_count, token_type):
        """在图片上添加token信息标注"""
        img_with_text = img.copy()
        
        # 准备文本信息
        text_lines = [
            f"Scale {scale_idx}",
            f"{token_count} tokens",
            f"({token_type})"
        ]
        
        # 设置文本参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)  # 白色文字
        thickness = 2
        bg_color = (0, 0, 0)  # 黑色背景
        
        # 计算文本大小用于背景
        text_sizes = []
        for text in text_lines:
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_sizes.append((text_width, text_height + baseline))
        
        max_width = max([size[0] for size in text_sizes])
        total_height = sum([size[1] for size in text_sizes]) + 10  # 额外间距
        
        # 绘制半透明背景
        overlay = img_with_text.copy()
        cv2.rectangle(overlay, (10, 10), (10 + max_width + 20, 10 + total_height), bg_color, -1)
        cv2.addWeighted(overlay, 0.7, img_with_text, 0.3, 0, img_with_text)
        
        # 绘制文本
        y_offset = 35
        for i, text in enumerate(text_lines):
            cv2.putText(img_with_text, text, (20, y_offset), font, font_scale, color, thickness)
            y_offset += text_sizes[i][1] + 5
        
        return img_with_text

    def save_visualizations(self, gt_reconstructed, corrected_reconstructed, 
                          original_image, output_dir, image_name, gt_tokens, corrected_tokens):
        """保存可视化结果"""
        print(f"\n正在保存可视化结果...")
        
        # 保存原始图片
        original_path = os.path.join(output_dir, f"{image_name}_original.jpg")
        original_image.save(original_path)
        
        # 保存各个尺度的重构图片（带token信息标注）
        for i, (gt_img, corrected_img) in enumerate(zip(gt_reconstructed, corrected_reconstructed)):
            # 计算当前尺度的token数量
            token_shape = gt_tokens[i].shape  # (B, T, H, W, D)
            token_count = token_shape[1] * token_shape[2] * token_shape[3]  # T * H * W
            
            # 为图片添加token信息标注
            gt_img_annotated = self.add_token_info_to_image(gt_img, i, token_count, "Original")
            corrected_img_annotated = self.add_token_info_to_image(corrected_img, i, token_count, "Corrected")
            
            # 保存标注后的图片
            gt_path = os.path.join(output_dir, f"{image_name}_scale{i}_gt_reconstructed.jpg")
            cv2.imwrite(gt_path, gt_img_annotated[:, :, ::-1])  # RGB -> BGR
            
            corrected_path = os.path.join(output_dir, f"{image_name}_scale{i}_corrected_reconstructed.jpg")
            cv2.imwrite(corrected_path, corrected_img_annotated[:, :, ::-1])  # RGB -> BGR
            
            print(f"Scale {i} 可视化已保存 ({token_count} tokens)")
        
        # 创建对比图
        self.create_comparison_visualization(
            gt_reconstructed, corrected_reconstructed, 
            output_dir, image_name, gt_tokens
        )
    
    def create_comparison_visualization(self, gt_reconstructed, corrected_reconstructed, 
                                     output_dir, image_name, gt_tokens):
        """创建对比可视化图"""
        print("正在创建对比可视化...")
        
        # 创建2x4的对比图 (2行，4列)
        fig_height = 512 * 2
        fig_width = 512 * 4
        comparison = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
        
        for i in range(4):
            # 计算token数量
            token_shape = gt_tokens[i].shape
            token_count = token_shape[1] * token_shape[2] * token_shape[3]
            
            # 为重构图片添加标注
            gt_img_annotated = self.add_token_info_to_image(gt_reconstructed[i], i, token_count, "Original")
            corrected_img_annotated = self.add_token_info_to_image(corrected_reconstructed[i], i, token_count, "Corrected")
            
            # 第一行：真实token重构
            y1, y2 = 0, 512
            x1, x2 = i * 512, (i + 1) * 512
            comparison[y1:y2, x1:x2] = gt_img_annotated
            
            # 第二行：纠正后token重构
            y1, y2 = 512, 1024
            comparison[y1:y2, x1:x2] = corrected_img_annotated
        
        # 保存对比图
        comparison_path = os.path.join(output_dir, f"{image_name}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison[:, :, ::-1])  # RGB -> BGR
        print(f"对比图已保存到: {comparison_path}")
        
    def save_tokens_as_txt(self, gt_tokens, corrected_tokens, output_dir, image_name):
        """将token以人类可读的格式保存到txt文件"""
        print(f"\n正在保存token数据到txt文件...")
        
        # 保存真实token
        gt_txt_path = os.path.join(output_dir, f"{image_name}_gt_tokens.txt")
        with open(gt_txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Infinity模型 - 真实Token数据 (Original Tokens)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"图片: {image_name}\n")
            f.write(f"提取时间: {datetime.now()}\n")
            f.write(f"模型: Infinity-8B (512x512)\n")
            f.write(f"VAE类型: {self.args.vae_type}\n")
            f.write(f"编码本维度: {self.vae.codebook_dim}\n\n")
            
            for i, tokens in enumerate(gt_tokens):
                scale_info = self.vae_scale_schedule[i]
                token_shape = tokens.shape  # (B, T, H, W, D)
                token_count = token_shape[1] * token_shape[2] * token_shape[3]
                
                f.write("-" * 60 + "\n")
                f.write(f"Scale {i}: 分辨率 {scale_info}\n")
                f.write("-" * 60 + "\n")
                f.write(f"Token形状: {token_shape}\n")
                f.write(f"Token数量: {token_count}\n")
                f.write(f"每个token的bit数: {token_shape[-1]}\n")
                f.write(f"总bit数: {token_count * token_shape[-1]}\n\n")
                
                # 将token数据展平并保存
                tokens_flat = tokens.squeeze(0).reshape(-1, token_shape[-1])  # (token_count, bit_dim)
                
                f.write("Token数据 (每行一个token，每个数字代表一个bit):\n")
                for token_idx, token in enumerate(tokens_flat):
                    bit_string = ''.join([str(int(bit.item())) for bit in token])
                    f.write(f"Token_{token_idx:04d}: {bit_string}\n")
                    
                    # 为了文件不过大，每个尺度最多保存前100个token的详细信息
                    if token_idx >= 99:
                        remaining = len(tokens_flat) - 100
                        if remaining > 0:
                            f.write(f"... (还有{remaining}个token，格式相同)\n")
                        break
                
                f.write("\n")
        
        # 保存纠正后token
        corrected_txt_path = os.path.join(output_dir, f"{image_name}_corrected_tokens.txt")
        with open(corrected_txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Infinity模型 - 纠正后Token数据 (Self-Corrected Tokens)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"图片: {image_name}\n")
            f.write(f"提取时间: {datetime.now()}\n")
            f.write(f"模型: Infinity-8B (512x512)\n")
            f.write(f"噪声应用层数: {self.noise_apply_layers}\n")
            f.write(f"噪声强度: {self.noise_apply_strength}\n\n")
            
            for i, (gt_tokens_scale, corrected_tokens_scale) in enumerate(zip(gt_tokens, corrected_tokens)):
                scale_info = self.vae_scale_schedule[i]
                token_shape = corrected_tokens_scale.shape
                token_count = token_shape[1] * token_shape[2] * token_shape[3]
                
                f.write("-" * 60 + "\n")
                f.write(f"Scale {i}: 分辨率 {scale_info}\n")
                f.write("-" * 60 + "\n")
                f.write(f"Token形状: {token_shape}\n")
                f.write(f"Token数量: {token_count}\n")
                f.write(f"每个token的bit数: {token_shape[-1]}\n")
                
                # 计算与原始token的差异
                gt_flat = gt_tokens_scale.squeeze(0).reshape(-1, token_shape[-1])
                corrected_flat = corrected_tokens_scale.squeeze(0).reshape(-1, token_shape[-1])
                diff_mask = (gt_flat != corrected_flat)
                total_bits = gt_flat.numel()
                flipped_bits = diff_mask.sum().item()
                
                f.write(f"总bit数: {total_bits}\n")
                f.write(f"翻转的bit数: {flipped_bits}\n")
                f.write(f"翻转比例: {flipped_bits/total_bits*100:.2f}%\n\n")
                
                f.write("纠正后Token数据 (每行一个token，*标记表示与原始token不同的bit):\n")
                for token_idx, (gt_token, corrected_token) in enumerate(zip(gt_flat, corrected_flat)):
                    # 创建bit字符串，标记差异
                    bit_string = ""
                    for bit_idx, (gt_bit, corr_bit) in enumerate(zip(gt_token, corrected_token)):
                        if gt_bit != corr_bit:
                            bit_string += f"{int(corr_bit.item())}*"  # 标记翻转的bit
                        else:
                            bit_string += str(int(corr_bit.item()))
                    
                    f.write(f"Token_{token_idx:04d}: {bit_string}\n")
                    
                    # 限制输出数量
                    if token_idx >= 99:
                        remaining = len(corrected_flat) - 100
                        if remaining > 0:
                            f.write(f"... (还有{remaining}个token，格式相同)\n")
                        break
                
                f.write("\n")
        
        print(f"真实token数据已保存到: {gt_txt_path}")
        print(f"纠正后token数据已保存到: {corrected_txt_path}")
        
        # 创建汇总统计文件
        summary_path = os.path.join(output_dir, f"{image_name}_token_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Token数量汇总 - Multi-scale Token Summary\n")
            f.write("=" * 50 + "\n\n")
            
            total_tokens = 0
            total_bits = 0
            
            for i, tokens in enumerate(gt_tokens):
                scale_info = self.vae_scale_schedule[i]
                token_shape = tokens.shape
                token_count = token_shape[1] * token_shape[2] * token_shape[3]
                bit_count = token_count * token_shape[-1]
                
                total_tokens += token_count
                total_bits += bit_count
                
                f.write(f"Scale {i}: {scale_info}\n")
                f.write(f"  Token数量: {token_count:,}\n")
                f.write(f"  每个token bit数: {token_shape[-1]}\n")
                f.write(f"  该尺度总bit数: {bit_count:,}\n\n")
            
            f.write("-" * 30 + "\n")
            f.write(f"总计:\n")
            f.write(f"  所有尺度token总数: {total_tokens:,}\n")
            f.write(f"  所有尺度bit总数: {total_bits:,}\n")
            f.write(f"  平均每个token bit数: {total_bits/total_tokens:.1f}\n")
        
        print(f"Token汇总已保存到: {summary_path}")
        
    def extract_from_image(self, image_path):
        """
        从图片提取多尺度token的完整流程
        
        这是整个类的主要接口，封装了从图片加载到结果保存的完整流程。
        """
        # 获取图片名称（用于文件命名）
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(OUTPUT_DIR, image_name)
        
        print(f"开始处理图片: {image_path}")
        print(f"输出目录: {output_dir}")
        
        # 1. 预处理图片
        inp_B3HW, original_image = self.preprocess_image(image_path)
        
        # 2. 提取原始特征
        raw_features = self.extract_raw_features(inp_B3HW)
        
        # 3. 提取多尺度token
        gt_tokens, corrected_tokens = self.extract_multiscale_tokens(raw_features)
        
        # 4. 重构图片用于可视化
        gt_reconstructed = self.reconstruct_images(gt_tokens, "真实token")
        corrected_reconstructed = self.reconstruct_images(corrected_tokens, "纠正后token")
        
        # 5. 保存结果
        self.save_tokens(gt_tokens, corrected_tokens, output_dir, image_name)
        self.save_tokens_as_txt(gt_tokens, corrected_tokens, output_dir, image_name)
        self.save_visualizations(
            gt_reconstructed, corrected_reconstructed, 
            original_image, output_dir, image_name, gt_tokens, corrected_tokens
        )
        
        print(f"\n✅ 处理完成！结果已保存到: {output_dir}")
        return gt_tokens, corrected_tokens

def main():
    """主函数"""
    print("=" * 60)
    print("Infinity多尺度视觉表征提取器")
    print("=" * 60)
    
    # 检查图片路径
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 错误：图片文件不存在: {IMAGE_PATH}")
        print("请修改脚本开头的IMAGE_PATH变量为正确的图片路径")
        return
    
    # 创建提取器并加载模型
    extractor = MultiscaleTokenExtractor(args)
    extractor.load_model()
    
    # 提取token
    try:
        gt_tokens, corrected_tokens = extractor.extract_from_image(IMAGE_PATH)
        
        print("\n" + "=" * 60)
        print("提取摘要:")
        print("=" * 60)
        print(f"✅ 成功提取了 {len(gt_tokens)} 个尺度的视觉表征")
        print(f"📁 结果保存在: {OUTPUT_DIR}")
        print(f"🖼️ 可视化图片可用于分析不同尺度捕获的视觉信息")
        print(f"💾 Token数据已保存，可用于后续分析")
        
    except Exception as e:
        print(f"❌ 提取过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
