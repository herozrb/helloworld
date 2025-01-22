from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import traceback
import json  # 添加json模块来处理文本内容
from datetime import datetime

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

@app.route('/upload', methods=['POST'])
def upload_file():
    print("\n=== 开始处理新的上传请求 ===")  # 新增
    try:
        if 'file' not in request.files:
            print("没有文件被上传")
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        print(f"接收到文件: {file.filename}")  # 新增
        
        if file.filename == '':
            print("文件名为空")
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(f"文件已保存到: {filepath}")
            
            try:
                print("开始读取文件")  # 新增
                # 读取文件内容
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath, encoding='utf-8')
                else:
                    df = pd.read_excel(filepath)
                
                print(f"文件读取成功，数据行数: {len(df)}")
                print(f"列名: {df.columns.tolist()}")
                
                # 检查数据框是否为空
                if df.empty:
                    print("数据框为空")  # 调试信息
                    return jsonify({'error': '文件内容为空'}), 400
                
                # 检查是否存在必要的列
                content_column = 'context_around_keyword'
                if content_column not in df.columns:
                    print(f"缺少列: {content_column}")  # 调试信息
                    return jsonify({'error': f'文件中缺少"{content_column}"列'}), 400
                
                # 获取规则配置
                keyword = request.form.get('rules', '').strip()
                print(f"搜索关键词: {keyword}")  # 调试信息
                
                if not keyword:
                    print("关键词为空")  # 调试信息
                    return jsonify({'error': '请输入标注规则关键词'}), 400
                
                # 应用标注规则并获取统计结果
                stats = apply_rules(df, keyword)
                print(f"处理结果: {stats}")  # 调试信息
                
                # 在返回结果前打印
                print("=== 处理完成，返回结果 ===")  # 新增
                return jsonify({
                    'success': True,
                    'stats': stats
                })
            except Exception as e:
                print(f"文件处理错误: {str(e)}")
                print(traceback.format_exc())
                return jsonify({'error': f'文件处理错误: {str(e)}'}), 400
            finally:
                pass
        
        print("不支持的文件类型")  # 调试信息
        return jsonify({'error': '不支持的文件类型'}), 400
    
    except Exception as e:
        print(f"上传过程错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'上传过程错误: {str(e)}'}), 500

def clean_text(text):
    """清理文本内容，提取实际对话内容"""
    try:
        # 尝试提取text字段的内容
        if '"text":"' in text:
            # 提取所有text字段的内容并合并
            texts = []
            parts = text.split('"text":"')
            for part in parts[1:]:  # 跳过第一部分
                end = part.find('"')
                if end != -1:
                    texts.append(part[:end])
            return ' '.join(texts)
        return text
    except:
        return text

def apply_rules(df, keyword):
    try:
        content_column = 'context_around_keyword'
        
        # 清理文本内容
        df['cleaned_content'] = df[content_column].astype(str).apply(clean_text)
        
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
        print(traceback.format_exc())
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
