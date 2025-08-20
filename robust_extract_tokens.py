原始图像尺寸: 640 x 480
缩放后尺寸: 682 x 512
最终预处理后图像: 512 x 512
tensor形状: torch.Size([3, 512, 512]), 数值范围: [-1.000, 1.000]

第一步：VAE编码获取原始特征...
原始特征形状: torch.Size([1, 14, 64, 64])

第二步：多尺度量化处理...
正在进行健壮的多尺度量化...
量化输入特征形状: torch.Size([1, 14, 1, 64, 64])
处理尺度 1/10: (1, 2, 2)
  - 量化后bit_indices形状: torch.Size([1, 1, 2, 2, 14])
    处理bit_indices: torch.Size([1, 1, 2, 2, 14])
    最终token形状: torch.Size([1, 1, 56])
    处理bit_indices: torch.Size([1, 1, 2, 2, 14])
    最终token形状: torch.Size([1, 1, 56])
处理尺度 2/10: (2, 4, 4)
  - 量化后bit_indices形状: torch.Size([1, 2, 4, 4, 14])
    处理bit_indices: torch.Size([1, 2, 4, 4, 14])
    最终token形状: torch.Size([1, 2, 224])
    处理bit_indices: torch.Size([1, 2, 4, 4, 14])
    最终token形状: torch.Size([1, 2, 224])
处理尺度 3/10: (3, 8, 8)
  - 量化后bit_indices形状: torch.Size([1, 3, 8, 8, 14])
    处理bit_indices: torch.Size([1, 3, 8, 8, 14])
Traceback (most recent call last):
  File "/mnt/nas-data-1/chenhao/Infinity/extract_multiscale_tokens.py", line 677, in <module>
    main()
  File "/mnt/nas-data-1/chenhao/Infinity/extract_multiscale_tokens.py", line 642, in main
    extraction_result = extractor.extract_multiscale_tokens(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/nas-data-1/chenhao/Infinity/extract_multiscale_tokens.py", line 416, in extract_multiscale_tokens
    original_tokens, corrected_tokens = self._robust_multiscale_quantization(
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/nas-data-1/chenhao/Infinity/extract_multiscale_tokens.py", line 233, in _robust_multiscale_quantization
    original_token = self._process_bit_indices(bit_indices, si, B)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/nas-data-1/chenhao/Infinity/extract_multiscale_tokens.py", line 323, in _process_bit_indices
    processed = torch.nn.functional.pixel_unshuffle(processed, 2)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)=3 is not divisible by 2
(inf) /mnt/nas-data-1/chenhao/Infinity> 
