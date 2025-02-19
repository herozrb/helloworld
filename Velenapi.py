from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import traceback
import json
from datetime import datetime
import requests
import sseclient
from tqdm import tqdm
import base64
import time

# 在文件开头，获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
    template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB限制

# 确保上传目录和导出目录都存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'exports'), exist_ok=True)

# 允许的文件类型
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def generate_basic_auth_key(username):
    """生成Basic认证的密钥"""
    auth_str = f"{username}:"  # 不需要密码，但保留冒号
    auth_bytes = auth_str.encode('utf-8')
    encoded_auth = base64.b64encode(auth_bytes)
    return f"Basic {encoded_auth.decode('utf-8')}"

def call_model(text, prompt, retry_count=3):
    """调用模型进行分析"""
    url = "https://velenmis-rc.sankuai.com/velen-container/chatgpt/chatCompletions"
    
    for attempt in range(retry_count):
        try:
            # 处理输入文本，尝试解析 JSON
            try:
                if isinstance(text, str) and text.startswith('['):
                    json_data = json.loads(text)
                    text_content = ' '.join(item.get('text', '') for item in json_data)
                else:
                    text_content = text
            except:
                text_content = text
            
            headers = {
                'Authorization': generate_basic_auth_key("intelligentInteraction"),
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": "DeepSeek-R1-Distill-32B-Velen",  # 改用新模型
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的文本分析助手，需要帮助分析文本并给出明确的判断结果。请只返回'是'或'否'。"
                    },
                    {
                        "role": "user",
                        "content": f"请按要求分析以下文本，{prompt}\n\n文本内容：{text_content}\n\n请只回答'是'或'否'。"
                    }
                ],
                "tenant": "knowledgeGraph",
                "velenOfflineChatFlag": False,
                "velenOfflineHiveFlag": False
            }
            
            print("\n=== 请求信息 ===")
            print(f"URL: {url}")
            print(f"Headers: {json.dumps(headers, indent=2)}")
            print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            # 增加超时时间
            response = requests.post(url, headers=headers, json=payload, 
                                  timeout=(30, 90))  # (连接超时, 读取超时)
            
            print("\n=== 响应信息 ===")
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 200 and 'data' in result:
                    if 'choices' in result['data']:
                        content = result['data']['choices'][0]['message']['content']
                        print(f"分析结果: {content}")
                        return content.strip()
                    else:
                        print("API返回成功但无内容")
                        time.sleep(5)  # 增加等待时间
                        continue
                elif result.get('code') == 500:
                    print(f"服务器错误: {result.get('msg')}")
                    time.sleep(10)  # 服务器错误时等待更长时间
                    continue
            
            print(f"请求失败: Status Code = {response.status_code}")
            time.sleep(5)  # 失败后等待
            if attempt == retry_count - 1:
                print("已达到最大重试次数")
                return None
                
        except requests.exceptions.ReadTimeout:
            print(f"第 {attempt + 1} 次请求读取超时")
            time.sleep(10)  # 超时后等待更长时间
            continue
        except requests.exceptions.ConnectTimeout:
            print(f"第 {attempt + 1} 次请求连接超时")
            time.sleep(5)
            continue
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {str(e)}")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"其他错误: {str(e)}")
            time.sleep(5)
            if attempt == retry_count - 1:
                return None
    
    return None

def apply_model_analysis(df, prompt):
    """使用模型分析文本"""
    total_rows = len(df)
    matched_count = 0
    sample_records = []
    
    # 使用 tqdm 创建进度条
    with tqdm(total=total_rows, desc="处理进度", unit="行") as pbar:
        for text in df['context_around_keyword'].astype(str):
            result = call_model(text, prompt)
            if result == '是':
                matched_count += 1
                if len(sample_records) < 10:  # 只保存前10个匹配的记录
                    sample_records.append(text)
            pbar.update(1)  # 更新进度条
    
    percentage = (matched_count / total_rows * 100) if total_rows > 0 else 0
    
    return {
        'total_rows': total_rows,
        'matched_count': matched_count,
        'percentage': round(percentage, 2),
        'sample_records': sample_records
    }

def process_data_with_model(text, prompt, retry_count=3):
    """使用模型处理单个数据"""
    url = "https://velenmis-rc.sankuai.com/velen-container/chatgpt/chatCompletions"
    
    for attempt in range(retry_count):
        try:
            # 处理输入文本，尝试解析 JSON
            try:
                if isinstance(text, str) and text.startswith('['):
                    json_data = json.loads(text)
                    text_content = ' '.join(item.get('text', '') for item in json_data)
                else:
                    text_content = text
            except:
                text_content = text
            
            headers = {
                'Authorization': generate_basic_auth_key("intelligentInteraction"),
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": "DeepSeek-R1-Distill-32B-Velen",  # 改用新模型
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的数据处理助手，需要按照用户的要求处理文本。"
                    },
                    {
                        "role": "user",
                        "content": f"请按照以下要求处理文本：{prompt}\n\n文本内容：{text_content}"
                    }
                ],
                "tenant": "knowledgeGraph",
                "velenOfflineChatFlag": False,
                "velenOfflineHiveFlag": False
            }
            
            print("\n=== 请求信息 ===")
            print(f"URL: {url}")
            print(f"Headers: {json.dumps(headers, indent=2)}")
            print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            # 增加超时时间
            response = requests.post(url, headers=headers, json=payload, 
                                  timeout=(30, 90))  # (连接超时, 读取超时)
            
            print("\n=== 响应信息 ===")
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 200 and 'data' in result:
                    if 'choices' in result['data']:
                        content = result['data']['choices'][0]['message']['content']
                        print(f"处理结果: {content}")
                        return content.strip()
                    else:
                        print("API返回成功但无内容")
                        continue
            
            print(f"请求失败: Status Code = {response.status_code}")
            if attempt == retry_count - 1:
                print("已达到最大重试次数")
                return None
                
        except requests.exceptions.ReadTimeout:
            print(f"第 {attempt + 1} 次请求读取超时")
            time.sleep(10)
            continue
        except requests.exceptions.ConnectTimeout:
            print(f"第 {attempt + 1} 次请求连接超时")
            time.sleep(5)
            continue
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {str(e)}")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"其他错误: {str(e)}")
            time.sleep(5)
            if attempt == retry_count - 1:
                return None
    
    return None

def process_dataframe(df, rules):
    """处理整个数据框"""
    output_file = None
    try:
        # 获取所有列名
        columns = df.columns.tolist()
        
        # 为每列创建新的处理后的列
        processed_df = pd.DataFrame()  # 创建新的数据框
        processed_count = 0  # 记录成功处理的行数
        
        for col in columns:
            # 复制原始列
            processed_df[col] = df[col]
            new_col_name = f"{col}_processed"
            processed_df[new_col_name] = None
            
            # 使用 tqdm 显示处理进度
            with tqdm(total=len(df), desc=f"处理 {col} 列", unit="行") as pbar:
                for idx, value in df[col].astype(str).items():
                    try:
                        processed_value = process_data_with_model(value, rules)
                        if processed_value is not None:
                            processed_df.at[idx, new_col_name] = processed_value
                            processed_count += 1
                    except Exception as e:
                        print(f"处理第 {idx} 行时出错: {str(e)}")
                    finally:
                        pbar.update(1)
                    time.sleep(1)  # 每条记录处理后等待1秒
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'exports')
        os.makedirs(export_dir, exist_ok=True)  # 确保导出目录存在
        output_file = os.path.join(export_dir, f'processed_{timestamp}.xlsx')
        
        # 保存处理后的数据
        processed_df.to_excel(output_file, index=False)
        
        # 如果没有成功处理任何数据，视为失败
        if processed_count == 0:
            if os.path.exists(output_file):
                os.remove(output_file)
            return {
                'success': False,
                'error': '没有成功处理任何数据'
            }
        
        return {
            'success': True,
            'message': f'数据处理完成，成功处理 {processed_count} 行数据',
            'file': os.path.basename(output_file)
        }
        
    except Exception as e:
        error_msg = f"数据处理错误: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        # 如果文件已经生成，返回文件信息
        if output_file and os.path.exists(output_file):
            return {
                'success': True,  # 虽然有错误但仍返回成功
                'message': f'处理过程中出现错误，但已保存部分结果: {error_msg}',
                'file': os.path.basename(output_file)
            }
        
        return {
            'success': False,
            'error': error_msg
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # 检查文件是否存在
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400
            
        try:
            # 先读取文件内容
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-8')
            else:
                df = pd.read_excel(file)
            
            if df.empty:
                return jsonify({'error': '文件内容为空'}), 400
            
            content_column = 'context_around_keyword'
            if content_column not in df.columns:
                return jsonify({'error': f'文件中缺少"{content_column}"列'}), 400
            
            # 获取分析参数
            analysis_type = request.form.get('analysisType', 'keyword')
            rules = request.form.get('rules', '').strip()
            
            if not rules:
                return jsonify({'error': '请输入规则'}), 400
            
            # 保存文件（只有在验证通过后才保存）
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # 处理数据
            try:
                if analysis_type == 'keyword':
                    stats = apply_rules(df, rules)
                elif analysis_type == 'model':
                    stats = apply_model_analysis(df, rules)
                else:  # process
                    result = process_dataframe(df, rules)
                    if result['success']:
                        return jsonify({
                            'success': True,
                            'message': result['message'],
                            'download_url': f"/download/{result['file']}"
                        })
                    else:
                        # 处理失败时删除上传的文件
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        return jsonify({'error': result['error']}), 400
                
                return jsonify({
                    'success': True,
                    'stats': stats
                })
                
            except Exception as e:
                # 处理失败时删除上传的文件
                if os.path.exists(filepath):
                    os.remove(filepath)
                print(f"数据处理错误: {str(e)}")
                print(traceback.format_exc())  # 打印详细错误堆栈
                return jsonify({'error': f'数据处理错误: {str(e)}'}), 400
                
        except Exception as e:
            print(f"文件读取错误: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'文件读取错误: {str(e)}'}), 400
            
    except Exception as e:
        print(f"上传过程错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'上传过程错误: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        export_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'exports')
        file_path = os.path.join(export_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'文件不存在: {filename}'}), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003) 