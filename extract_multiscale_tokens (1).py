import random
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np
from tools.run_infinity import *
from PIL import Image
import torch.nn.functional as F

# 使用与inf.py相同的配置
model_path='weights/infinity_8b_512x512_weights'
vae_path='weights/infinity_vae_d56_f8_14_patchify.pth'
text_encoder_ckpt = 'weights/flan-t5-xl-official'

args=argparse.Namespace(
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
    save_file='tmp.jpg'
)

def test_image_preprocessing():
    """测试图片预处理的形状变化"""
    print("=== 测试图片预处理 ===")
    
    # 创建一个模拟的非正方形图片 (640x480)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    print(f"原始图片尺寸: {pil_image.size}")  # (width, height)
    
    # 使用run_infinity.py中的transform函数
    tgt_h, tgt_w = 512, 512
    transformed = transform(pil_image, tgt_h, tgt_w)
    print(f"预处理后tensor形状: {transformed.shape}")
    print(f"数值范围: [{transformed.min():.3f}, {transformed.max():.3f}]")
    
    # 添加batch维度
    inp_B3HW = transformed.unsqueeze(0)
    print(f"添加batch维度后: {inp_B3HW.shape}")
    
    return inp_B3HW

def test_vae_encoding(vae, inp_B3HW):
    """测试VAE编码阶段的形状变化"""
    print("\n=== 测试VAE编码 ===")
    
    # 获取scale_schedule
    h_div_w_template = 1.0
    scale_schedule = dynamic_resolution_h_w[h_div_w_template]['0.25M']['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    print(f"Scale schedule: {scale_schedule[:5]}...")  # 只打印前5个
    
    # 如果有spatial_patchify，需要转换scale_schedule
    if args.apply_spatial_patchify:
        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        print(f"VAE scale schedule (patchify): {vae_scale_schedule[:5]}...")
    else:
        vae_scale_schedule = scale_schedule
    
    device = inp_B3HW.device
    
    # 1. 测试原始特征提取
    print("\n--- VAE encode_for_raw_features ---")
    raw_features, hs, hs_mid = vae.encode_for_raw_features(inp_B3HW, scale_schedule=vae_scale_schedule)
    print(f"Raw features形状: {raw_features.shape}")
    print(f"Raw features数值范围: [{raw_features.min():.3f}, {raw_features.max():.3f}]")
    
    # 2. 测试完整编码（包括量化）
    print("\n--- VAE encode (with quantization) ---")
    h, z, all_indices, all_bit_indices, residual_norm, var_input = vae.encode(inp_B3HW, scale_schedule=vae_scale_schedule)
    print(f"编码后h形状: {h.shape}")
    print(f"量化后z形状: {z.shape}")
    print(f"all_bit_indices数量: {len(all_bit_indices)}")
    
    # 打印前几个scale的bit_indices形状
    for i, bit_indices in enumerate(all_bit_indices[:4]):
        print(f"Scale {i} bit_indices形状: {bit_indices.shape}")
    
    return raw_features, vae_scale_schedule, all_bit_indices

def test_bitwise_self_correction(vae, inp_B3HW, raw_features, vae_scale_schedule):
    """测试Bitwise Self-Correction的形状变化"""
    print("\n=== 测试Bitwise Self-Correction ===")
    
    # 创建一个简化的BSC对象来测试
    class SimpleBSC:
        def __init__(self, vae, args):
            self.vae = vae
            self.apply_spatial_patchify = args.apply_spatial_patchify
            self.noise_apply_layers = 2  # 前2层添加噪声
            self.noise_apply_strength = 0.1
            
        def test_flip_requant(self, vae_scale_schedule, inp_B3HW, raw_features, device):
            print("开始逐scale处理...")
            
            B = raw_features.shape[0]
            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2)
            else:
                codes_out = raw_features
            print(f"初始codes_out形状: {codes_out.shape}")
            
            cum_var_input = 0
            gt_all_bit_indices = []
            pred_all_bit_indices = []
            
            for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                print(f"\n--- Scale {si}: ({pt}, {ph}, {pw}) ---")
                
                # 计算residual
                residual = codes_out - cum_var_input
                print(f"Residual形状 (before interpolate): {residual.shape}")
                
                if si != len(vae_scale_schedule)-1:
                    residual = F.interpolate(residual, size=vae_scale_schedule[si], mode=self.vae.quantizer.z_interplote_down).contiguous()
                    print(f"Residual形状 (after interpolate): {residual.shape}")
                
                # BSQ量化
                quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual)
                print(f"Quantized形状: {quantized.shape}")
                print(f"Bit indices形状: {bit_indices.shape}")
                
                # 保存真实indices
                gt_all_bit_indices.append(bit_indices)
                
                # 模拟噪声添加
                if si < self.noise_apply_layers:
                    pred_bit_indices = bit_indices.clone()
                    noise_strength = 0.05  # 简化的噪声强度
                    mask = torch.rand(*bit_indices.shape).to(device) < noise_strength
                    pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
                    pred_all_bit_indices.append(pred_bit_indices)
                    print(f"添加噪声，翻转了 {mask.sum().item()} 个bits")
                else:
                    pred_all_bit_indices.append(bit_indices)
                    print("未添加噪声")
                
                # 累积到最大尺度
                cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1], mode=self.vae.quantizer.z_interplote_up).contiguous()
                print(f"累积后cum_var_input形状: {cum_var_input.shape}")
                
                # 只测试前4个scale
                if si >= 3:
                    break
            
            return gt_all_bit_indices, pred_all_bit_indices
    
    # 创建测试对象
    bsc = SimpleBSC(vae, args)
    device = inp_B3HW.device
    
    # 运行测试
    gt_indices, pred_indices = bsc.test_flip_requant(vae_scale_schedule, inp_B3HW, raw_features, device)
    
    print(f"\n最终结果:")
    print(f"真实indices数量: {len(gt_indices)}")
    print(f"预测indices数量: {len(pred_indices)}")
    
    return gt_indices, pred_indices

def main():
    """主测试函数"""
    print("开始形状测试...")
    
    # 1. 测试图片预处理
    inp_B3HW = test_image_preprocessing()
    inp_B3HW = inp_B3HW.cuda()
    
    # 2. 加载VAE
    print("\n=== 加载VAE ===")
    vae = load_visual_tokenizer(args)
    print(f"VAE codebook_dim: {vae.codebook_dim}")
    print(f"VAE apply_spatial_patchify: {args.apply_spatial_patchify}")
    
    # 3. 测试VAE编码
    raw_features, vae_scale_schedule, all_bit_indices = test_vae_encoding(vae, inp_B3HW)
    
    # 4. 测试Bitwise Self-Correction
    gt_indices, pred_indices = test_bitwise_self_correction(vae, inp_B3HW, raw_features, vae_scale_schedule)
    
    print("\n=== 测试完成 ===")
    print("请检查以上形状输出是否符合预期")

if __name__ == '__main__':
    main()
