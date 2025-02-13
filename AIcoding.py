from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import traceback
import json
from datetime import datetime
import requests
import sseclient
from tqdm import tqdm  # 导入 tqdm 库

app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 允许的文件类型
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def call_model(text, prompt):
    """调用第三方 API 进行分析"""
    try:
        # 格式化 prompt
        formatted_prompt = f"请分析以下文本，{prompt}\n\n文本内容：{text}\n\n请只回答'是'或'否'。"
        
        headers = {
            'Authorization': '1729335449824772151',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的文本分析助手，需要帮助分析文本并给出明确的判断结果。请只返回'是'或'否'。"
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            "stream": True
        }
        
        response = requests.post(
            'https://aigc.sankuai.com/v1/openai/native/chat/completions',
            headers=headers,
            json=data,
            stream=True
        )
        
        client = sseclient.SSEClient(response)
        full_response = ""
        for event in client.events():
            if event.data != "[DONE]":
                chunk = json.loads(event.data)
                if 'choices' in chunk and chunk['choices']:
                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                    full_response += content
        
        # 清理结果，只保留"是"或"否"
        result = '是' if '是' in full_response else '否'
        return result
    except Exception as e:
        print(f"模型调用错误: {str(e)}")
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

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath, encoding='utf-8')
                else:
                    df = pd.read_excel(filepath)
                
                if df.empty:
                    return jsonify({'error': '文件内容为空'}), 400
                
                content_column = 'context_around_keyword'
                if content_column not in df.columns:
                    return jsonify({'error': f'文件中缺少"{content_column}"列'}), 400
                
                analysis_type = request.form.get('analysisType', 'keyword')
                rules = request.form.get('rules', '').strip()
                
                if not rules:
                    return jsonify({'error': '请输入规则'}), 400
                
                if analysis_type == 'keyword':
                    stats = apply_rules(df, rules)
                else:  # model
                    stats = apply_model_analysis(df, rules)
                
                return jsonify({
                    'success': True,
                    'stats': stats
                })
            except Exception as e:
                return jsonify({'error': f'文件处理错误: {str(e)}'}), 400
            
        return jsonify({'error': '不支持的文件类型'}), 400
    
    except Exception as e:
        return jsonify({'error': f'上传过程错误: {str(e)}'}), 500

def apply_rules(df, keyword):
    """使用关键词匹配分析"""
    try:
        content_column = 'context_around_keyword'
        
        # 清理文本内容
        df['cleaned_content'] = df[content_column].astype(str)
        
        # 总行数
        total_rows = len(df)
        
        # 找到包含关键词的行
        matched_rows = df[df['cleaned_content'].str.contains(keyword, na=False)]
        matched_count = len(matched_rows)
        
        # 计算占比
        percentage = (matched_count / total_rows * 100) if total_rows > 0 else 0
        
        # 获取前10个匹配的记录
        sample_records = matched_rows['cleaned_content'].head(10).tolist()
        
        return {
            'total_rows': total_rows,
            'matched_count': matched_count,
            'percentage': round(percentage, 2),
            'sample_records': sample_records
        }
    except Exception as e:
        print(f"规则应用错误: {str(e)}")
        raise

@app.route('/export', methods=['POST'])
def export_results():
    try:
        data = request.json
        if not data or 'stats' not in data:
            return jsonify({'error': '没有数据可导出'}), 400
        
        # 创建导出目录
        export_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        # 生成导出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_file = os.path.join(export_dir, f'分析结果_{timestamp}.xlsx')
        
        # 创建数据框
        stats = data['stats']
        summary_df = pd.DataFrame([{
            '总数据量': stats['total_rows'],
            '匹配数量': stats['matched_count'],
            '匹配占比': f"{stats['percentage']}%",
            '搜索关键词': data['keyword']
        }])
        
        records_df = pd.DataFrame(stats['sample_records'], columns=['匹配记录'])
        
        # 使用ExcelWriter保存到不同的sheet
        with pd.ExcelWriter(export_file) as writer:
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
            records_df.to_excel(writer, sheet_name='匹配记录', index=True)
        
        return jsonify({
            'success': True,
            'file': os.path.basename(export_file)
        })
    except Exception as e:
        print(f"导出错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'导出失败: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        export_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'exports')
        return send_file(
            os.path.join(export_dir, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)