#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 h5py 字符串解码兼容性问题
"""

import h5py

def patch_h5py():
    """
    修复 h5py 的字符串解码问题
    将字符串的 decode 方法替换为直接返回自身
    """
    def return_self(self, *args, **kwargs):
        return self
    
    # 为字符串类型添加一个 decode 方法，直接返回自身
    if not hasattr(str, 'decode'):
        str.decode = return_self

# 在导入 keras 之前应用补丁
patch_h5py()

print("H5py 兼容性补丁已应用")