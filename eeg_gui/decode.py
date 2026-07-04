def encode24BitSigned(b0 : int, b1 : int, b2 : int) -> int:
    """
    将3个字节(大端序/MSB First)转换为24位有符号整数
    """

    rawVal = (b0 << 16) | (b1 << 8) | b2
    
    # 判断是否为负数：如果24位二进制的最高位（符号位）是1，即 >= 0x800000
    if rawVal >= 0x800000:
        rawVal -= 0x1000000 # 补码转换，求出实际的负数值
        
    return rawVal

def parseEEGPacket(packetBytes : bytes) -> tuple:
    """
    解析完整的 33 字节 EEG 数据包，返回样本编号和各通道电压值 (uV)
    """
    # 1. 基础校验
    if len(packetBytes) != 33:
        raise ValueError(f"数据包长度错误，期望 33 字节，实际得到 {len(packetBytes)} 字节")
    if packetBytes[0] != 0xA0:
        raise ValueError(f"帧头错误, 期望 0xA0, 实际得到 {hex(packetBytes[0])}")
    if packetBytes[32] != 0xC0:
        raise ValueError(f"帧尾错误, 期望 0xC0, 实际得到 {hex(packetBytes[32])}")
        
    # 2. 提取样本编号
    sampleNumber = packetBytes[1]
    
    # 3. 说明书提供的比例因子 (默认增益 x24)
    SCALE_FACTOR = 0.022351744455307063
    
    # 4. 循环解析 8 个通道的数据
    channels_uV = []
    rawCounts = [] # 用于测试打印原始整数值
    
    for i in range(8):
        startIdx = 2 + i * 3
        b0 = packetBytes[startIdx]
        b1 = packetBytes[startIdx + 1]
        b2 = packetBytes[startIdx + 2]
        
        # 将字节转为 24位整数
        adcCount = encode24BitSigned(b0, b1, b2)
        rawCounts.append(adcCount)
        
        # 乘以比例因子得到实际电压 (微伏)
        voltage_uV = adcCount * SCALE_FACTOR
        channels_uV.append(voltage_uV)
        
    return sampleNumber, rawCounts, channels_uV

if __name__ == "__main__":
    test_packet = bytearray([
        0xA0,             # Byte 1: 帧头
        0x45,             # Byte 2: 样本编号 (例如 69)
        
        0x29, 0x96, 0x49, # Bytes 3-5: 通道1 (说明书正数例子)
        0xE1, 0x96, 0x49, # Bytes 6-8: 通道2 (说明书负数例子)
        0x00, 0x00, 0x01, # Bytes 9-11: 通道3 (微小正数)
        0xFF, 0xFF, 0xFF, # Bytes 12-14: 通道4 (微小负数)
        0x00, 0x00, 0x00, # Bytes 15-17: 通道5 
        0x00, 0x00, 0x00, # Bytes 18-20: 通道6 
        0x00, 0x00, 0x00, # Bytes 21-23: 通道7 
        0x00, 0x00, 0x00, # Bytes 24-26: 通道8 
        
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, # Bytes 27-32: Aux Data (保留位)
        0xC0              # Byte 33: 帧尾
    ])
    
    print("开始解析测试数据包...")
    try:
        sample_num, raw_data, uV_data = parseEEGPacket(test_packet)
        print("-" * 40)
        print(f"样本编号 (Sample Number): {sample_num}")
        print("-" * 40)
        
        for i in range(8):
            print(f"通道 {i + 1}:")
            print(f"  -> 解析整数值: {raw_data[i]}")
            print(f"  -> 转换电压值: {uV_data[i]:.4f} uV")
            
    except Exception as e:
        print(f"解析失败: {e}")