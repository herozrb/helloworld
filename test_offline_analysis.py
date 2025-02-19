import requests
import json
import base64
import time
from datetime import datetime

def generate_basic_auth_key(username):
    """生成Basic认证的密钥"""
    auth_str = f"{username}:"
    auth_bytes = auth_str.encode('utf-8')
    encoded_auth = base64.b64encode(auth_bytes)
    return f"Basic {encoded_auth.decode('utf-8')}"

def submit_offline_task(text, batch_id):
    """提交离线分析任务"""
    url = "https://velenmis-rc.sankuai.com/velen-container/chatgpt/chatCompletions"
    
    headers = {
        'Authorization': generate_basic_auth_key("intelligentInteraction"),
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "LongCat-MoE-3B-32K-Chat",
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
        "taskName": "testvelenmisapi_batch1"  # 使用固定的任务名
    }
    
    print(f"\n=== 提交离线任务 {batch_id} ===")
    print("请求信息:")
    print(f"URL: {url}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print("\n响应信息:")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 200:
                print(f"任务提交成功: {result}")
                return True
        
        print(f"任务提交失败: {response.text}")
        return False
        
    except Exception as e:
        print(f"提交任务出错: {str(e)}")
        return False

def main():
    # 测试数据
    test_texts = [
        "这个电话无法接通，请稍后再拨",
        "您拨打的用户已关机",
        "对方正在通话中",
        "该用户已停机",
        "电话已转至语音信箱",
        "您拨打的电话暂时无人接听",
        "号码是空号",
        "用户已欠费停机",
        "该号码已停止使用",
        "系统繁忙，请稍后再拨"
    ]
    
    print("开始提交离线分析任务...")
    
    for i, text in enumerate(test_texts, 1):
        success = submit_offline_task(text, i)
        
        if success:
            print(f"\n成功提交第 {i} 个任务")
        else:
            print(f"\n提交第 {i} 个任务失败")
        
        # 等待一段时间再提交下一个任务
        time.sleep(5)
    
    print("\n所有任务提交完成")
    print("\n请在 DW_HIVE_DB_CONNECT_URL.log.pay_rc_velen_chat_gpt_data 表中查看结果")
    print("查询示例: SELECT * FROM pay_rc_velen_chat_gpt_data WHERE task_name = 'testvelenmisapi_batch1'")

if __name__ == "__main__":
    main() 