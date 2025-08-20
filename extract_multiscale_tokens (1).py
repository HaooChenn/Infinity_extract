(inf) /mnt/nas-data-1/chenhao/Infinity> python /mnt/nas-data-1/chenhao/Infinity/shape.py
开始形状测试...
=== 测试图片预处理 ===
原始图片尺寸: (640, 480)
预处理后tensor形状: torch.Size([3, 512, 512])
数值范围: [-1.000, 1.000]
添加batch维度后: torch.Size([1, 3, 512, 512])

=== 加载VAE ===
VAE codebook_dim: 14
VAE apply_spatial_patchify: 1

=== 测试VAE编码 ===
Scale schedule: [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 6, 6), (1, 8, 8)]...
VAE scale schedule (patchify): [(1, 2, 2), (1, 4, 4), (1, 8, 8), (1, 12, 12), (1, 16, 16)]...

--- VAE encode_for_raw_features ---
Raw features形状: torch.Size([1, 14, 64, 64])
Raw features数值范围: [-4.140, 5.670]

--- VAE encode (with quantization) ---
编码后h形状: torch.Size([1, 14, 64, 64])
量化后z形状: torch.Size([1, 14, 64, 64])
all_bit_indices数量: 10
Scale 0 bit_indices形状: torch.Size([1, 1, 2, 2, 14])
Scale 1 bit_indices形状: torch.Size([1, 1, 4, 4, 14])
Scale 2 bit_indices形状: torch.Size([1, 1, 8, 8, 14])
Scale 3 bit_indices形状: torch.Size([1, 1, 12, 12, 14])

=== 测试Bitwise Self-Correction ===
开始逐scale处理...
初始codes_out形状: torch.Size([1, 14, 1, 64, 64])

--- Scale 0: (1, 2, 2) ---
Residual形状 (before interpolate): torch.Size([1, 14, 1, 64, 64])
Residual形状 (after interpolate): torch.Size([1, 14, 1, 2, 2])
Quantized形状: torch.Size([1, 14, 1, 2, 2])
Bit indices形状: torch.Size([1, 1, 2, 2, 14])
添加噪声，翻转了 6 个bits
累积后cum_var_input形状: torch.Size([1, 14, 1, 64, 64])

--- Scale 1: (1, 4, 4) ---
Residual形状 (before interpolate): torch.Size([1, 14, 1, 64, 64])
Residual形状 (after interpolate): torch.Size([1, 14, 1, 4, 4])
Quantized形状: torch.Size([1, 14, 1, 4, 4])
Bit indices形状: torch.Size([1, 1, 4, 4, 14])
添加噪声，翻转了 11 个bits
累积后cum_var_input形状: torch.Size([1, 14, 1, 64, 64])

--- Scale 2: (1, 8, 8) ---
Residual形状 (before interpolate): torch.Size([1, 14, 1, 64, 64])
Residual形状 (after interpolate): torch.Size([1, 14, 1, 8, 8])
Quantized形状: torch.Size([1, 14, 1, 8, 8])
Bit indices形状: torch.Size([1, 1, 8, 8, 14])
未添加噪声
累积后cum_var_input形状: torch.Size([1, 14, 1, 64, 64])

--- Scale 3: (1, 12, 12) ---
Residual形状 (before interpolate): torch.Size([1, 14, 1, 64, 64])
Residual形状 (after interpolate): torch.Size([1, 14, 1, 12, 12])
Quantized形状: torch.Size([1, 14, 1, 12, 12])
Bit indices形状: torch.Size([1, 1, 12, 12, 14])
未添加噪声
累积后cum_var_input形状: torch.Size([1, 14, 1, 64, 64])

最终结果:
真实indices数量: 4
预测indices数量: 4

=== 测试完成 ===
请检查以上形状输出是否符合预期
