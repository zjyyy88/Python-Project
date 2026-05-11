#!/usr/bin/env python3
"""
更新opencode.json配置文件，从CATL API获取最新模型列表。
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
import ssl
from pathlib import Path


def get_default_config_path():
    """获取默认的配置文件路径"""
    home = Path.home()
    
    # 根据操作系统确定路径
    if sys.platform == 'win32':
        config_dir = home / '.config' / 'opencode'
    else:
        config_dir = home / '.config' / 'opencode'
    
    return config_dir / 'opencode.json'


def read_config(config_path):
    """读取配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件JSON格式错误 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取配置文件失败 - {e}", file=sys.stderr)
        sys.exit(1)


def fetch_models(api_key, base_url="http://newapi.catl.com/v1"):
    """从API获取模型列表"""
    url = f"{base_url}/models"
    
    # 创建请求
    req = urllib.request.Request(url, method='GET')
    req.add_header('Authorization', f'Bearer {api_key}')
    
    # 创建SSL上下文
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except urllib.error.HTTPError as e:
        print(f"HTTP错误: {e.code} - {e.reason}", file=sys.stderr)
        try:
            error_body = e.read().decode('utf-8')
            print(f"错误详情: {error_body}", file=sys.stderr)
        except:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"请求失败: {e}", file=sys.stderr)
        sys.exit(1)


def update_config_with_models(config, models_data):
    """使用获取的模型列表更新配置"""
    if 'data' not in models_data or not models_data['data']:
        print("警告: API返回的模型列表为空", file=sys.stderr)
        return config
    
    models = {}
    for model in models_data['data']:
        model_id = model.get('id', '')
        if not model_id:
            continue
        
        # 构建模型配置
        model_config = {
            "name": model_id,
            "limit": {
                "context": 100000,
                "output": 65536
            },
            "modalities": {
                "input": ["text", "image"],
                "output": ["text"]
            },
            "attachment": True
        }
        
        models[model_id] = model_config
    
    # 更新配置中的models
    if 'provider' not in config:
        config['provider'] = {}
    
    if 'catl' not in config['provider']:
        config['provider']['catl'] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "CATL AI",
            "options": {
                "baseURL": "http://newapi.catl.com/v1",
                "apiKey": ""
            },
            "models": {}
        }
    
    config['provider']['catl']['models'] = models
    
    return config


def save_config(config_path, config):
    """保存配置文件"""
    try:
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"错误: 保存配置文件失败 - {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='更新opencode.json配置文件')
    parser.add_argument('--config-path', type=str, help='配置文件路径（可选）')
    parser.add_argument('--api-key', type=str, help='API密钥（可选，默认使用配置中的key）')
    
    args = parser.parse_args()
    
    # 确定配置文件路径
    if args.config_path:
        config_path = Path(args.config_path)
    else:
        config_path = get_default_config_path()
    
    print(f"配置文件路径: {config_path}")
    
    # 读取现有配置
    config = read_config(config_path)
    
    if config is None:
        print(f"\n错误: 配置文件不存在于 {config_path}")
        print("\n请在对应位置创建配置文件，或使用 --config-path 指定路径。")
        print("\n配置文件示例格式:")
        example_config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "catl": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "CATL AI",
                    "options": {
                        "baseURL": "http://newapi.catl.com/v1",
                        "apiKey": "your-api-key-here"
                    },
                    "models": {}
                }
            },
            "enabled_providers": ["catl"]
        }
        print(json.dumps(example_config, ensure_ascii=False, indent=4))
        sys.exit(1)
    
    print("成功读取配置文件")
    
    # 获取API密钥
    api_key = args.api_key
    if not api_key:
        try:
            api_key = config['provider']['catl']['options']['apiKey']
        except (KeyError, TypeError):
            print("\n错误: 未找到API密钥")
            print("请在配置文件中设置 apiKey，或使用 --api-key 参数指定")
            sys.exit(1)
    
    # 获取模型列表
    print("\n正在从API获取模型列表...")
    try:
        models_data = fetch_models(api_key)
    except SystemExit:
        raise
    except Exception as e:
        print(f"错误: 获取模型列表失败 - {e}", file=sys.stderr)
        sys.exit(1)
    
    # 检查模型列表
    if 'data' not in models_data:
        print("\n错误: API响应格式不正确")
        print(f"响应内容: {json.dumps(models_data, ensure_ascii=False, indent=2)}")
        sys.exit(1)
    
    model_count = len(models_data.get('data', []))
    print(f"成功获取 {model_count} 个模型")
    
    # 更新配置
    print("\n正在更新配置文件...")
    updated_config = update_config_with_models(config, models_data)
    
    # 保存配置
    if save_config(config_path, updated_config):
        print(f"[OK] 配置文件已成功更新: {config_path}")
        print("\n" + "="*60)
        print("重要提示:")
        print("="*60)
        print("配置文件已更新，请重启opencode以生效！")
        print("\n重启后，新的模型配置将自动加载。")
        print("="*60)
    else:
        print("\n错误: 配置文件更新失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
