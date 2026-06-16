def interpolate_rewards(rewards):
    """
    对列表 rewards 中的 None 值进行线性插值，要求两端都有非 None 值。
    如果缺失值位于开头或结尾，则用最近的非 None 值填充。
    """
    n = len(rewards)
    # 创建一个新的列表，避免直接修改原列表
    result = rewards.copy()
    i = 0
    while i < n:
        if result[i] is None:
            # 记录缺失值段的起始索引
            start = i
            # 计算连续 None 的个数
            while i < n and result[i] is None:
                i += 1
            end = i  # end 是连续缺失值后第一个非 None 的索引
            left_index = start - 1
            right_index = end

            # 判断两侧是否有非 None 的值
            left_value = result[left_index] if left_index >= 0 else None
            right_value = result[right_index] if right_index < n else None

            # 如果两侧都有非 None 值，则进行线性插值
            if left_value is not None and right_value is not None:
                gap_length = right_index - left_index - 1  # 缺失值的数量
                for j in range(1, gap_length + 1):
                    result[left_index + j] = left_value + (right_value - left_value) * j / (gap_length + 1)
            # 如果缺失值在开头（左侧没有值），则全部填充为右侧的值
            elif left_value is None and right_value is not None:
                for j in range(start, i):
                    result[j] = right_value
            # 如果缺失值在结尾（右侧没有值），则全部填充为左侧的值
            elif left_value is not None and right_value is None:
                for j in range(start, i):
                    result[j] = left_value
            # 如果两侧都没有非 None 值，则无法插值，保持原样
        else:
            i += 1
    return result