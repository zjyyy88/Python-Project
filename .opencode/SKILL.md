---
name: update-opencode-config
description: 更新opencode.json配置文件，从CATL API获取最新模型列表并更新到配置文件中。用户输入“更新模型”等话语则自动调用该skill。
---

# Update OpenCode Config

本技能用于更新opencode.json配置文件，自动从CATL API获取最新的模型列表。

## When to Use This Skill

当用户需要以下操作时，使用本技能：

- 更新opencode.json配置文件
- 同步CATL API的最新模型列表
- 刷新可用的AI模型配置
- 输入更新模型等话语则自动更新模型

## 使用方法

1. **检查配置文件**: 自动查找~/.config/opencode/opencode.json
2. **获取模型列表**: 从http://newapi.catl.com/v1/models获取最新模型
3. **更新配置**: 更新配置文件中的models部分
4. **提示重启**: 提示用户重启opencode以生效

## 配置文件路径

默认查找路径（按优先级排序）：
- Windows: `%USERPROFILE%\.config\opencode\opencode.json`
- Linux/macOS: `~/.config/opencode/opencode.json`

## 相关脚本

- `scripts/update_config.py` - 更新配置文件脚本

## 使用示例

```bash
# 更新配置文件
python scripts/update_config.py

# 指定API密钥（可选，默认使用配置中的key）
python scripts/update_config.py --api-key your_api_key

# 指定配置文件路径
python scripts/update_config.py --config-path /path/to/opencode.json
```

## 重要提示

> **更新完成后必须重启opencode**，新配置才能生效。
>
> 重启终端后，AI将使用更新后的模型配置。

## 错误处理

- 如果配置文件不存在，脚本会提示用户在对应位置创建文件
- 如果API请求失败，会显示详细的错误信息
- 如果配置格式不正确，会提示用户检查文件内容
