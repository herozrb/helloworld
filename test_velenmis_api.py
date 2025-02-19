import requests
import json
import base64
import time

def generate_basic_auth_key(username):
    """生成Basic认证的密钥"""
    auth_str = f"{username}:"  # 不需要密码，但保留冒号
    auth_bytes = auth_str.encode('utf-8')
    encoded_auth = base64.b64encode(auth_bytes)
    return f"Basic {encoded_auth.decode('utf-8')}"

def test_api(text="给我说2个科学家的名字", prompt=None):
    """测试 velenmis API"""
    url = "https://velenmis-rc.sankuai.com/velen-container/chatgpt/chatCompletions"
    
    headers = {
        'Authorization': generate_basic_auth_key("intelligentInteraction"),
        'Content-Type': 'application/json'
    }
    
    # 基于成功的配置进行测试
    payloads = [
        # 1. 基础成功配置
        {
            "model": "LongCat-MoE-3B-32K-Chat",
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ],
            "tenant": "knowledgeGraph",
            "velenOfflineChatFlag": False,
            "velenOfflineHiveFlag": False
        },
        
        # 2. 带 system 角色的配置
        {
            "model": "LongCat-MoE-3B-32K-Chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的助手。"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "tenant": "knowledgeGraph",
            "velenOfflineChatFlag": False,
            "velenOfflineHiveFlag": False
        }
    ]
    
    # 如果提供了 prompt，添加文本分析配置
    if prompt:
        payloads.append({
            "model": "LongCat-MoE-3B-32K-Chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的文本分析助手，需要帮助分析文本并给出明确的判断结果。请只返回'是'或'否'。"
                },
                {
                    "role": "user",
                    "content": f"请按要求分析以下文本，{prompt}\n\n文本内容：{text}\n\n请只回答'是'或'否'。"
                }
            ],
            "tenant": "knowledgeGraph",
            "velenOfflineChatFlag": False,
            "velenOfflineHiveFlag": False
        })
    
    # 测试每个配置
    for i, payload in enumerate(payloads, 1):
        print(f"\n\n=== 测试配置 {i} ===")
        print("配置说明：")
        if i == 1:
            print("基础配置 (使用成功的模型)")
        elif i == 2:
            print("带 system 角色")
        elif i == 3:
            print("文本分析配置")
            
        print("\n--- 请求信息 ---")
        print(f"URL: {url}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
        print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print("\n--- 响应信息 ---")
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 200 and 'data' in result:
                    if 'choices' in result['data']:
                        content = result['data']['choices'][0]['message']['content']
                        print(f"\n结果: {content}")
                    else:
                        print("\n无内容返回")
            
        except Exception as e:
            print(f"\n错误: {str(e)}")
        
        # 等待一下，避免请求太频繁
        time.sleep(3)

def submit_offline_task(text, batch_id):
    """提交离线分析任务"""
    url = "https://velenmis-rc.sankuai.com/velen-container/chatgpt/chatCompletions"
    
    headers = {
        'Authorization': generate_basic_auth_key("intelligentInteraction"),
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "DeepSeek-R1-Distill-32B-Velen",  # 改用新模型
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的文本分析助手，需要帮助分析文本并给出明确的判断结果。"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "tenant": "intelligentInteraction",
        "velenOfflineChatFlag": True,
        "velenOfflineHiveFlag": True,
        "taskName": "testvelenmisapi_batch1"
    }
    
    # ... 其余代码保持不变 ...

if __name__ == "__main__":
    # 测试基础对话
    print("\n=== 测试基础对话 ===")
    test_api()
    
    # 测试文本分析
    print("\n=== 测试文本分析 ===")
    test_text = "话已转至语音留言，你尝试联系的用户无法接通，请在提示音后录制留言。"
    test_prompt = "判断这段文本是否表示电话无法接通"
    test_api(test_text, test_prompt) 