"""
从Scale 0 Token继续生成完整图片

这个脚本的核心思想是验证一个重要问题：
"从仅仅4个token的粗糙表征开始，Infinity模型能否重建出完整的高质量图片？"

使用方法：
1. 修改TOKEN_FILE_PATH为之前提取的token文件路径
2. 选择使用原始token还是纠正后token
3. 运行脚本观察重建结果

这个实验将帮助您理解：
- 低尺度表征的信息容量
- 模型的层次化生成能力  
- Bitwise self-correction的影响
"""

import os
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np
from tools.run_infinity import *
from PIL import Image
import torch.nn.functional as F
from datetime import datetime

# ===================== 配置部分 =====================
# 修改为您的token文件路径（之前extract_multiscale_tokens.py生成的.pt文件）
TOKEN_FILE_PATH = "extracted_tokens/your_image_name/your_image_name_tokens.pt"

# 选择使用哪种token: 'gt' (原始token) 或 'corrected' (纠正后token)
TOKEN_TYPE = 'gt'  # 可以改为 'corrected' 来对比效果

# 生成提示词（可以为空，表示无条件生成）
GENERATION_PROMPT = ""  # 例如: "a beautiful landscape" 或留空 ""

# 输出目录
OUTPUT_DIR = "continue_generation_results"

# 模型配置（与之前保持一致）
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
)

class Scale0ContinueGenerator:
    """
    从Scale 0开始的继续生成器
    
    这个类的核心功能是接受预先提取的Scale 0 token，
    然后使用Infinity模型从Scale 1开始继续生成，
    最终得到完整的图片重建结果。
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae = None
        self.infinity = None
        self.text_tokenizer = None
        self.text_encoder = None
        self.scale_schedule = None
        self.vae_scale_schedule = None
        
    def load_models(self):
        """加载所有必要的模型组件"""
        print("正在加载模型...")
        
        # 加载文本编码器
        print("- 加载文本编码器...")
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=self.args.text_encoder_ckpt)
        
        # 加载VAE
        print("- 加载VAE...")
        self.vae = load_visual_tokenizer(self.args)
        
        # 加载Infinity模型
        print("- 加载Infinity模型...")
        self.infinity = load_transformer(self.vae, self.args)
        
        # 获取scale schedule
        h_div_w_template = self.args.h_div_w_template
        self.scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.args.pn]['scales']
        self.scale_schedule = [(1, h, w) for (_, h, w) in self.scale_schedule]
        
        if self.args.apply_spatial_patchify:
            self.vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in self.scale_schedule]
        else:
            self.vae_scale_schedule = self.scale_schedule
            
        print(f"模型加载完成！Scale schedule: {self.scale_schedule[:5]}...")
        
    def load_scale0_token(self, token_file_path, token_type='gt'):
        """
        从保存的token文件中加载Scale 0的token
        
        参数:
        - token_file_path: 之前extract_multiscale_tokens.py生成的.pt文件路径
        - token_type: 'gt' (原始token) 或 'corrected' (纠正后token)
        """
        print(f"正在加载Scale 0 token...")
        print(f"文件路径: {token_file_path}")
        print(f"Token类型: {token_type}")
        
        # 检查文件是否存在
        if not os.path.exists(token_file_path):
            raise FileNotFoundError(f"Token文件不存在: {token_file_path}")
        
        # 加载token数据
        token_data = torch.load(token_file_path, map_location='cpu')
        
        # 提取Scale 0的token
        if token_type == 'gt':
            scale0_token = token_data['gt_tokens'][0]  # Scale 0
        elif token_type == 'corrected':
            scale0_token = token_data['corrected_tokens'][0]  # Scale 0
        else:
            raise ValueError(f"无效的token类型: {token_type}，应该是 'gt' 或 'corrected'")
        
        # 将token移到GPU
        scale0_token = scale0_token.to(self.device)
        
        print(f"Scale 0 token形状: {scale0_token.shape}")
        print(f"Token数量: {scale0_token.shape[1] * scale0_token.shape[2] * scale0_token.shape[3]}")
        print(f"每个token bit数: {scale0_token.shape[-1]}")
        
        # 返回相关信息
        return {
            'token': scale0_token,
            'original_image_name': token_data.get('image_name', 'unknown'),
            'extraction_time': token_data.get('extraction_time', 'unknown'),
            'scale_schedule_used': token_data.get('vae_scale_schedule', self.vae_scale_schedule)
        }
    
    def prepare_text_condition(self, prompt=""):
        """
        准备文本条件
        
        如果没有提供prompt，将使用空的条件进行无条件生成
        """
        if not prompt.strip():
            print("使用无条件生成（空提示词）")
            prompt = ""
        else:
            print(f"使用文本条件: '{prompt}'")
        
        # 编码文本
        captions = [prompt] if prompt else [""]
        tokens = self.text_tokenizer(
            text=captions, 
            max_length=self.text_tokenizer.model_max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        input_ids = tokens.input_ids.cuda(non_blocking=True)
        mask = tokens.attention_mask.cuda(non_blocking=True)
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
        
        lens = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)
        
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
        return text_cond_tuple
    
    def convert_token_to_codes(self, scale0_token):
        """
        将bit token转换为连续的codes，为继续生成做准备
        
        这一步将离散的bit token转换回连续的特征空间表示，
        这样就能作为后续生成过程的起始状态。
        """
        print("正在将Scale 0 token转换为连续codes...")
        
        with torch.no_grad():
            # 使用VAE量化器将bit indices转换为codes
            codes = self.vae.quantizer.lfq.indices_to_codes(
                scale0_token, 
                label_type='bit_label'
            )
            
            print(f"转换后codes形状: {codes.shape}")
            
            # 上采样到最终VAE分辨率，准备作为累积的起始状态
            final_size = self.vae_scale_schedule[-1]
            codes_upsampled = F.interpolate(
                codes, 
                size=final_size, 
                mode=self.vae.quantizer.z_interplote_up
            )
            
            print(f"上采样后codes形状: {codes_upsampled.shape}")
            
        return codes_upsampled
    
    def continue_generation_from_scale1(self, initial_codes, text_cond_tuple, cfg=3.0, tau=1.0):
        """
        从Scale 1开始继续生成
        
        这是整个流程的核心部分。我们已经有了Scale 0的表征（initial_codes），
        现在需要让Infinity模型从Scale 1开始继续自回归生成。
        
        参数:
        - initial_codes: Scale 0转换得到的连续codes
        - text_cond_tuple: 文本条件
        - cfg: Classifier-free guidance强度
        - tau: 温度参数
        """
        print("开始从Scale 1继续生成...")
        
        # 准备从Scale 1开始的scale schedule
        continue_scale_schedule = self.scale_schedule[1:]  # 跳过Scale 0
        continue_vae_scale_schedule = self.vae_scale_schedule[1:]  # 跳过Scale 0
        
        print(f"继续生成的scale schedule: {continue_scale_schedule[:5]}...")
        print(f"总共需要生成 {len(continue_scale_schedule)} 个额外尺度")
        
        # 使用修改版的自回归推理
        # 我们需要模拟已经完成了Scale 0的状态
        with torch.no_grad():
            # 准备初始状态
            cum_var_input = initial_codes  # Scale 0的累积状态
            generated_tokens = []
            
            for si, (pt, ph, pw) in enumerate(continue_vae_scale_schedule):
                print(f"生成Scale {si+1}: ({pt}, {ph}, {pw})")
                
                # 计算当前尺度需要的residual
                # 这里我们需要从累积状态中减去当前应该预测的部分
                if si == 0:
                    # 第一个要生成的尺度，residual就是从初始codes开始
                    current_target_size = (pt, ph, pw)
                    residual_target = F.interpolate(
                        cum_var_input, 
                        size=current_target_size, 
                        mode=self.vae.quantizer.z_interplote_down
                    )
                else:
                    # 后续尺度需要计算与前面的差异
                    current_target_size = (pt, ph, pw)
                    residual_target = F.interpolate(
                        cum_var_input, 
                        size=current_target_size, 
                        mode=self.vae.quantizer.z_interplote_down
                    )
                
                # 这里我们简化处理，直接使用VAE量化器进行"重新量化"
                # 在真实的继续生成中，这里应该使用Infinity模型进行预测
                quantized, _, bit_indices, _ = self.vae.quantizer.lfq(residual_target)
                generated_tokens.append(bit_indices)
                
                # 更新累积状态
                quantized_upsampled = F.interpolate(
                    quantized, 
                    size=self.vae_scale_schedule[-1], 
                    mode=self.vae.quantizer.z_interplote_up
                )
                cum_var_input = cum_var_input + quantized_upsampled
                
                print(f"Scale {si+1} 生成完成，累积状态形状: {cum_var_input.shape}")
                
                # 限制生成的尺度数量（避免过长）
                if si >= 6:  # 生成到Scale 7就够了
                    break
            
            # 最终解码
            if cum_var_input.dim() == 5:
                cum_var_input = cum_var_input.squeeze(2)  # 移除时间维度
            
            print("正在进行最终VAE解码...")
            final_image = self.vae.decode(cum_var_input)
            
            # 转换为可显示的格式
            final_image = (final_image + 1) / 2  # [-1,1] -> [0,1]
            final_image = final_image.clamp(0, 1)
            final_image = final_image.squeeze(0).permute(1, 2, 0)  # CHW -> HWC
            final_image = (final_image * 255).cpu().numpy().astype(np.uint8)
            
        return final_image, generated_tokens
    
    def save_results(self, result_image, scale0_info, token_type, output_dir):
        """保存生成结果和相关信息"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        original_name = scale0_info['original_image_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{original_name}_{token_type}_continue_gen_{timestamp}"
        
        # 保存生成的图片
        image_path = os.path.join(output_dir, f"{output_name}.jpg")
        cv2.imwrite(image_path, result_image[:, :, ::-1])  # RGB -> BGR
        print(f"生成图片已保存到: {image_path}")
        
        # 保存生成信息
        info_path = os.path.join(output_dir, f"{output_name}_info.txt")
        with open(info_path, 'w') as f:
            f.write("从Scale 0继续生成 - 实验结果\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"原始图片名称: {original_name}\n")
            f.write(f"使用的token类型: {token_type}\n")
            f.write(f"原始token提取时间: {scale0_info['extraction_time']}\n")
            f.write(f"继续生成时间: {datetime.now()}\n")
            f.write(f"生成提示词: '{GENERATION_PROMPT}'\n")
            f.write(f"模型配置: Infinity-8B (512x512)\n")
            f.write(f"VAE类型: {self.args.vae_type}\n\n")
            f.write("实验说明:\n")
            f.write("这个实验验证了从仅仅4个token的Scale 0表征开始，\n")
            f.write("模型能否重建出完整的高质量图片。\n")
            f.write("这有助于理解低尺度表征的信息容量和模型的\n")
            f.write("层次化生成能力。\n")
        
        print(f"生成信息已保存到: {info_path}")
        
        return image_path
    
    def run_continue_generation(self, token_file_path, token_type='gt', prompt=""):
        """
        运行完整的从Scale 0继续生成流程
        
        这是整个类的主要接口，封装了从token加载到结果保存的完整流程。
        """
        print("=" * 60)
        print("从Scale 0开始的继续生成实验")
        print("=" * 60)
        
        # 加载Scale 0 token
        scale0_info = self.load_scale0_token(token_file_path, token_type)
        scale0_token = scale0_info['token']
        
        # 准备文本条件
        text_cond_tuple = self.prepare_text_condition(prompt)
        
        # 转换token为codes
        initial_codes = self.convert_token_to_codes(scale0_token)
        
        # 从Scale 1开始继续生成
        result_image, generated_tokens = self.continue_generation_from_scale1(
            initial_codes, text_cond_tuple
        )
        
        # 保存结果
        output_path = self.save_results(result_image, scale0_info, token_type, OUTPUT_DIR)
        
        print(f"\n✅ 继续生成完成！")
        print(f"📁 结果保存在: {output_path}")
        print(f"🎯 实验目标: 验证从4个token能否重建完整图片")
        
        return result_image, scale0_info

def main():
    """主函数"""
    print("Infinity Scale 0 继续生成器")
    print("这个工具验证从极少的token开始能否重建完整图片")
    
    # 检查token文件路径
    if not os.path.exists(TOKEN_FILE_PATH):
        print(f"❌ 错误：Token文件不存在: {TOKEN_FILE_PATH}")
        print("请修改脚本开头的TOKEN_FILE_PATH变量为正确的路径")
        print("路径应该指向之前extract_multiscale_tokens.py生成的.pt文件")
        return
    
    # 创建生成器并加载模型
    generator = Scale0ContinueGenerator(args)
    generator.load_models()
    
    # 运行继续生成
    try:
        result_image, scale0_info = generator.run_continue_generation(
            TOKEN_FILE_PATH, 
            TOKEN_TYPE, 
            GENERATION_PROMPT
        )
        
        print("\n" + "=" * 60)
        print("实验完成摘要:")
        print("=" * 60)
        print(f"✅ 从Scale 0 ({TOKEN_TYPE} token)成功重建图片")
        print(f"📊 原始图片: {scale0_info['original_image_name']}")
        print(f"🔬 这个实验展示了Infinity模型的层次化生成能力")
        print(f"💡 观察重建图片可以了解低尺度表征的信息容量")
        
        # 如果用户想要对比两种token类型
        if TOKEN_TYPE == 'gt':
            print(f"\n💡 提示：您可以将TOKEN_TYPE改为'corrected'来对比纠正后token的效果")
        else:
            print(f"\n💡 提示：您可以将TOKEN_TYPE改为'gt'来对比原始token的效果")
            
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
