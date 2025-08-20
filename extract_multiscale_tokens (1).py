import os
import random
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np
from tools.run_infinity import *
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================
# 配置区域 - 您只需要修改这里的路径
# =====================================
INPUT_IMAGE_PATH = "path/to/your/image.jpg"  # 修改为您的图片路径
OUTPUT_DIR = "extracted_tokens"  # 输出目录

# 模型配置（与您的inf.py保持一致）
model_path = 'weights/infinity_8b_512x512_weights'
vae_path = 'weights/infinity_vae_d56_f8_14_patchify.pth'
text_encoder_ckpt = 'weights/flan-t5-xl-official'

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
    noise_apply_layers=2,  # 前2层添加噪声
    noise_apply_strength=0.1,  # 噪声强度
    debug_bsc=False
)

class TokenExtractor:
    """Token提取器：实现多尺度视觉表征的提取和可视化"""
    
    def __init__(self, args, output_dir):
        self.args = args
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载VAE模型
        print("正在加载VAE模型...")
        self.vae = load_visual_tokenizer(args)
        print(f"VAE加载完成 - codebook_dim: {self.vae.codebook_dim}")
        
        # 获取scale schedule配置
        h_div_w_template = args.h_div_w_template
        self.scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        self.scale_schedule = [(1, h, w) for (_, h, w) in self.scale_schedule]
        
        # 如果启用spatial patchify，需要调整VAE的scale schedule
        if args.apply_spatial_patchify:
            self.vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in self.scale_schedule]
        else:
            self.vae_scale_schedule = self.scale_schedule
            
        print(f"Scale schedule (前4层): {self.scale_schedule[:4]}")
        print(f"VAE scale schedule (前4层): {self.vae_scale_schedule[:4]}")
    
    def preprocess_image(self, image_path):
        """
        图片预处理：将任意尺寸的图片转换为512x512的标准输入
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            tuple: (原始PIL图片, 预处理后的tensor)
        """
        print(f"\n=== 图片预处理 ===")
        print(f"输入图片: {image_path}")
        
        # 加载图片
        pil_image = Image.open(image_path).convert('RGB')
        original_size = pil_image.size  # (width, height)
        print(f"原始尺寸: {original_size[0]}x{original_size[1]}")
        
        # 使用run_infinity.py中的transform函数进行预处理
        # 这会智能地resize和crop到512x512，保持宽高比
        tgt_h, tgt_w = 512, 512
        transformed = transform(pil_image, tgt_h, tgt_w)
        print(f"预处理后形状: {transformed.shape}")
        print(f"数值范围: [{transformed.min():.3f}, {transformed.max():.3f}]")
        
        # 添加batch维度并移到GPU
        inp_B3HW = transformed.unsqueeze(0).cuda()
        print(f"最终输入形状: {inp_B3HW.shape}")
        
        return pil_image, inp_B3HW
    
    def extract_multiscale_tokens(self, inp_B3HW):
        """
        提取多尺度tokens：实现Bitwise Self-Correction流程
        
        Args:
            inp_B3HW: 预处理后的输入图片tensor
            
        Returns:
            tuple: (原始tokens列表, 纠正后tokens列表)
        """
        print(f"\n=== 多尺度Token提取 ===")
        
        # 步骤1: 通过VAE编码器获取原始特征
        print("步骤1: VAE编码...")
        with torch.amp.autocast('cuda', enabled=False):
            raw_features, _, _ = self.vae.encode_for_raw_features(
                inp_B3HW, scale_schedule=self.vae_scale_schedule
            )
        print(f"原始特征形状: {raw_features.shape}")
        
        # 步骤2: 实施Bitwise Self-Correction流程
        print("步骤2: Bitwise Self-Correction...")
        gt_tokens, corrected_tokens = self._bitwise_self_correction(
            raw_features, inp_B3HW.device
        )
        
        return gt_tokens, corrected_tokens
    
    def _bitwise_self_correction(self, raw_features, device):
        """
        实现Bitwise Self-Correction的核心算法
        这个方法模拟了训练时的噪声注入和纠错过程
        """
        print("开始逐尺度处理...")
        
        # 初始化variables
        B = raw_features.shape[0]
        if raw_features.dim() == 4:
            codes_out = raw_features.unsqueeze(2)  # 添加时间维度
        else:
            codes_out = raw_features
        
        cum_var_input = 0  # 累积的变分输入
        gt_all_tokens = []  # 存储原始tokens
        corrected_all_tokens = []  # 存储纠正后的tokens
        
        # 逐尺度处理前4个scale
        for si, (pt, ph, pw) in enumerate(self.vae_scale_schedule[:4]):
            print(f"\n--- 处理Scale {si}: ({pt}, {ph}, {pw}) ---")
            
            # 计算当前尺度的residual（剩余信息）
            residual = codes_out - cum_var_input
            print(f"Residual形状: {residual.shape}")
            
            # 如果不是最后一个尺度，需要插值到当前尺度
            if si < len(self.vae_scale_schedule) - 1:
                residual = F.interpolate(
                    residual, 
                    size=(pt, ph, pw), 
                    mode=self.vae.quantizer.z_interplote_down
                ).contiguous()
                print(f"插值后Residual形状: {residual.shape}")
            
            # 通过BSQ进行量化，获取bit indices
            quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual)
            print(f"Quantized形状: {quantized.shape}")
            print(f"Bit indices形状: {bit_indices.shape}")
            
            # 保存原始的bit indices（真实tokens）
            gt_all_tokens.append(bit_indices.clone())
            
            # 实施噪声注入进行Self-Correction（模拟训练时的纠错过程）
            if si < self.args.noise_apply_layers:
                # 为前几层添加噪声来模拟可能的预测错误
                corrected_indices = bit_indices.clone()
                noise_strength = np.random.uniform(0, self.args.noise_apply_strength)
                
                # 随机翻转一些bits
                mask = torch.rand(*bit_indices.shape, device=device) < noise_strength
                corrected_indices[mask] = 1 - corrected_indices[mask]
                
                corrected_all_tokens.append(corrected_indices)
                flipped_bits = mask.sum().item()
                print(f"添加噪声: 翻转了 {flipped_bits} 个bits (强度: {noise_strength:.3f})")
            else:
                # 后面的尺度不添加噪声
                corrected_all_tokens.append(bit_indices.clone())
                print("未添加噪声")
            
            # 累积当前尺度的贡献到最大尺度
            # 这是VAR模型的核心：每个尺度都在之前所有尺度的基础上添加更精细的细节
            cum_var_input = cum_var_input + F.interpolate(
                quantized, 
                size=self.vae_scale_schedule[-1], 
                mode=self.vae.quantizer.z_interplote_up
            ).contiguous()
            print(f"累积特征形状: {cum_var_input.shape}")
        
        print(f"\n提取完成! 获得了 {len(gt_all_tokens)} 个尺度的原始tokens和纠正后tokens")
        return gt_all_tokens, corrected_all_tokens
    
    def save_tokens(self, gt_tokens, corrected_tokens, prefix="tokens"):
        """
        保存tokens到文件
        
        Args:
            gt_tokens: 原始tokens列表
            corrected_tokens: 纠正后tokens列表  
            prefix: 文件前缀
        """
        print(f"\n=== 保存Tokens ===")
        
        tokens_dir = self.output_dir / "tokens"
        tokens_dir.mkdir(exist_ok=True)
        
        for i in range(len(gt_tokens)):
            # 保存原始tokens
            gt_path = tokens_dir / f"{prefix}_scale{i}_original.pt"
            torch.save(gt_tokens[i].cpu(), gt_path)
            
            # 保存纠正后tokens
            corrected_path = tokens_dir / f"{prefix}_scale{i}_corrected.pt"
            torch.save(corrected_tokens[i].cpu(), corrected_path)
            
            print(f"Scale {i}: 已保存到 {gt_path.name} 和 {corrected_path.name}")
            print(f"  形状: {gt_tokens[i].shape}")
    
    def visualize_tokens(self, gt_tokens, corrected_tokens, original_image):
        """
        可视化tokens：包括重构图片和统计信息
        
        Args:
            gt_tokens: 原始tokens列表
            corrected_tokens: 纠正后tokens列表
            original_image: 原始输入图片
        """
        print(f"\n=== 可视化Tokens ===")
        
        # 创建可视化目录
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. 保存原始图片
        original_image.save(vis_dir / "00_original_input.jpg")
        
        # 2. 为每个尺度生成重构图片
        self._generate_reconstruction_images(gt_tokens, corrected_tokens, vis_dir)
        
        # 3. 生成tokens差异统计
        self._generate_difference_statistics(gt_tokens, corrected_tokens, vis_dir)
        
        # 4. 生成综合对比图
        self._generate_comparison_grid(gt_tokens, corrected_tokens, vis_dir, original_image)
        
        print("可视化完成! 请查看visualizations文件夹")
    
    def _generate_reconstruction_images(self, gt_tokens, corrected_tokens, vis_dir):
        """生成每个尺度的重构图片"""
        print("生成重构图片...")
        
        for i in range(len(gt_tokens)):
            scale_info = self.vae_scale_schedule[i]
            print(f"Scale {i} ({scale_info}): 生成重构图片")
            
            try:
                # 重构原始tokens
                gt_img = self._reconstruct_from_tokens(gt_tokens[i], i)
                if gt_img is not None:
                    gt_img.save(vis_dir / f"scale{i}_original_reconstruction.jpg")
                
                # 重构纠正后tokens
                corrected_img = self._reconstruct_from_tokens(corrected_tokens[i], i)
                if corrected_img is not None:
                    corrected_img.save(vis_dir / f"scale{i}_corrected_reconstruction.jpg")
                    
            except Exception as e:
                print(f"Scale {i} 重构失败: {e}")
    
    def _reconstruct_from_tokens(self, bit_indices, scale_idx):
        """
        从bit indices重构图片
        这个方法将tokens转换回图片格式以便可视化
        """
        try:
            # 使用VAE的解码器从bit indices重构图片
            with torch.no_grad():
                # 将bit indices转换为codes
                codes = self.vae.quantizer.lfq.indices_to_codes(
                    bit_indices, label_type='bit_label'
                )
                
                # 解码为图片
                recon_img = self.vae.decode(codes.squeeze(-3))
                
                # 转换为PIL图片格式
                recon_img = (recon_img + 1) / 2  # 从[-1,1]转换到[0,1]
                recon_img = recon_img.clamp(0, 1)
                recon_img = recon_img.squeeze(0).permute(1, 2, 0)  # CHW -> HWC
                recon_img = (recon_img.cpu().numpy() * 255).astype(np.uint8)
                
                return Image.fromarray(recon_img)
                
        except Exception as e:
            print(f"重构错误: {e}")
            return None
    
    def _generate_difference_statistics(self, gt_tokens, corrected_tokens, vis_dir):
        """生成tokens差异的统计信息"""
        print("生成差异统计...")
        
        stats = []
        for i in range(len(gt_tokens)):
            gt = gt_tokens[i]
            corrected = corrected_tokens[i]
            
            # 计算差异
            diff = (gt != corrected).float()
            diff_ratio = diff.mean().item()
            total_bits = gt.numel()
            changed_bits = diff.sum().item()
            
            scale_stats = {
                'scale': i,
                'shape': list(gt.shape),
                'total_bits': total_bits,
                'changed_bits': int(changed_bits),
                'change_ratio': diff_ratio,
                'scale_info': self.vae_scale_schedule[i]
            }
            stats.append(scale_stats)
            
            print(f"Scale {i}: {changed_bits}/{total_bits} bits改变 ({diff_ratio:.3%})")
        
        # 保存统计信息
        import json
        with open(vis_dir / "difference_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _generate_comparison_grid(self, gt_tokens, corrected_tokens, vis_dir, original_image):
        """生成综合对比网格图"""
        print("生成对比网格...")
        
        try:
            num_scales = len(gt_tokens)
            fig, axes = plt.subplots(3, num_scales, figsize=(4*num_scales, 12))
            
            if num_scales == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(num_scales):
                # 第一行：原始重构
                gt_img = self._reconstruct_from_tokens(gt_tokens[i], i)
                if gt_img is not None:
                    axes[0, i].imshow(gt_img)
                axes[0, i].set_title(f'Scale {i}\nOriginal')
                axes[0, i].axis('off')
                
                # 第二行：纠正后重构
                corrected_img = self._reconstruct_from_tokens(corrected_tokens[i], i)
                if corrected_img is not None:
                    axes[1, i].imshow(corrected_img)
                axes[1, i].set_title(f'Scale {i}\nCorrected')
                axes[1, i].axis('off')
                
                # 第三行：差异可视化
                gt = gt_tokens[i].float()
                corrected = corrected_tokens[i].float()
                diff = (gt != corrected).float().mean(dim=-1).squeeze()  # 在最后一维求平均
                
                if diff.dim() >= 2:
                    diff_img = diff.cpu().numpy()
                    im = axes[2, i].imshow(diff_img, cmap='Reds', vmin=0, vmax=1)
                    plt.colorbar(im, ax=axes[2, i], fraction=0.046)
                axes[2, i].set_title(f'Scale {i}\nDifference')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(vis_dir / "comparison_grid.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"生成对比网格失败: {e}")
    
    def run_extraction(self, image_path):
        """
        运行完整的token提取流程
        
        Args:
            image_path: 输入图片路径
        """
        print("=" * 60)
        print("开始Infinity多尺度Token提取")
        print("=" * 60)
        
        # 1. 预处理图片
        original_image, inp_B3HW = self.preprocess_image(image_path)
        
        # 2. 提取多尺度tokens
        gt_tokens, corrected_tokens = self.extract_multiscale_tokens(inp_B3HW)
        
        # 3. 保存tokens
        image_name = Path(image_path).stem
        self.save_tokens(gt_tokens, corrected_tokens, prefix=image_name)
        
        # 4. 生成可视化
        self.visualize_tokens(gt_tokens, corrected_tokens, original_image)
        
        print("\n" + "=" * 60)
        print("提取完成!")
        print(f"结果保存在: {self.output_dir.absolute()}")
        print("=" * 60)
        
        return gt_tokens, corrected_tokens

def main():
    """主函数：执行token提取流程"""
    
    # 检查输入图片是否存在
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"错误: 图片文件不存在: {INPUT_IMAGE_PATH}")
        print("请修改INPUT_IMAGE_PATH变量为正确的图片路径")
        return
    
    # 创建token提取器
    extractor = TokenExtractor(args, OUTPUT_DIR)
    
    # 运行提取流程
    try:
        gt_tokens, corrected_tokens = extractor.run_extraction(INPUT_IMAGE_PATH)
        
        # 打印结果摘要
        print(f"\n提取摘要:")
        print(f"- 处理图片: {INPUT_IMAGE_PATH}")
        print(f"- 提取尺度数: {len(gt_tokens)}")
        for i, token in enumerate(gt_tokens):
            print(f"  Scale {i}: {token.shape}")
        
    except Exception as e:
        print(f"提取过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
