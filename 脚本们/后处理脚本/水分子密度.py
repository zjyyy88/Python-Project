def calculate_water_properties(temp_celsius, relative_humidity_percent):
    """
    计算特定温湿度下空气中水分子的物理性质
    """
    R = 8.314       # J/(mol*K)
    NA = 6.022e23   # molecules/mol
    M_H2O = 18.015  # g/mol (水的摩尔质量)
    
    T_kelvin = temp_celsius + 273.15
    
    # 1. 计算饱和水汽压 (Magnus 公式, 单位 Pa)
    P_sat = 611.2 * 10 ** ((7.62 * temp_celsius) / (243.12 + temp_celsius))
    
    # 2. 计算实际分压
    P_actual = P_sat * (relative_humidity_percent / 100.0)
    
    # 3. 计算摩尔密度 (mol/m^3)
    molar_density = P_actual / (R * T_kelvin)
    
    # --- 结果转换 ---
    
    # A. 分子数密度 (个/cm^3)
    # 先算出 个/m^3，再除以 10^6
    number_density_cm3 = (molar_density * NA) / 1e6
    
    # B. 质量密度 (g/cm^3)
    # 先算出 g/m^3 (即绝对湿度)，再除以 10^6
    mass_density_gm3 = molar_density * M_H2O
    mass_density_gcm3 = mass_density_gm3 / 1e6
    
    return {
        "pressure_pa": P_actual,
        "number_density": number_density_cm3,
        "mass_density": mass_density_gcm3,
        "absolute_humidity": mass_density_gm3 # 常用的 g/m^3
    }

# --- 设置您的实验环境参数 ---
T = 25   # 温度 (°C)
RH = 60  # 相对湿度 (%)

res = calculate_water_properties(T, RH)

print(f"=== 环境条件: {T}°C, {RH}% RH ===")
print(f"1. 水汽分压:     {res['pressure_pa']:.2f} Pa")
print(f"2. 分子数密度:   {res['number_density']:.2e} 个/cm³")
print(f"3. 质量密度:     {res['mass_density']:.2e} g/cm³")
print(f"   (即绝对湿度):  {res['absolute_humidity']:.2f} g/m³")