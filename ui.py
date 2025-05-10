# 提供交互界面进行关键词分析
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets


def keyword_analysis_ui():
    """提供交互式界面进行关键词分析"""

    # 文件选择控件
    file_selector = widgets.Text(
        value='',
        placeholder='请输入CSV文件路径',
        description='数据文件:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )

    # 关键词输入控件
    keyword_input = widgets.Textarea(
        value='',
        placeholder='请输入关键词，每行一个，最多10个',
        description='关键词:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%', height='150px')
    )

    # 输出目录控件
    output_dir = widgets.Text(
        value='output',
        description='输出目录:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )

    # 分析按钮
    analyze_button = widgets.Button(
        description='开始分析',
        button_style='primary',
        tooltip='点击开始分析选定关键词',
        icon='search'
    )

    # 输出区域
    output = widgets.Output()

    # 分析按钮点击事件处理函数
    def on_analyze_button_clicked(b):
        with output:
            clear_output()

            # 获取文件路径
            file_path = file_selector.value.strip()
            if not file_path:
                print("错误: 请提供数据文件路径")
                return

            # 获取关键词列表
            keywords_text = keyword_input.value.strip()
            if not keywords_text:
                print("错误: 请输入至少一个关键词")
                return

            # 拆分关键词并限制数量
            keywords = [kw.strip() for kw in keywords_text.split('\n') if kw.strip()]
            if len(keywords) > 10:
                print("警告: 输入了超过10个关键词，将只分析前10个")
                keywords = keywords[:10]

            # 获取输出目录
            out_dir = output_dir.value.strip()
            if not out_dir:
                out_dir = "output"

            # 执行分析
            try:
                print(f"开始分析 {len(keywords)} 个关键词...")
                for i, keyword in enumerate(keywords):
                    print(f"\n[{i + 1}/{len(keywords)}] 正在分析关键词: {keyword}")

                results = analyze_keywords(file_path, keywords, out_dir)

                # 显示结果链接
                print("\n分析完成! 报告文件生成于:")
                for keyword, res in results.items():
                    report_file = res['report_file']
                    print(f"- {keyword}: {report_file}")

                # 尝试显示第一个关键词的可视化结果
                if keywords:
                    first_keyword = keywords[0]
                    cooccurrence_file = f"{out_dir}/{first_keyword}_cooccurrence_bar.html"
                    inst_file = f"{out_dir}/{first_keyword}_institutions_bar.html"

                    print("\n预览可视化结果 (以第一个关键词为例):")

                    try:
                        with open(cooccurrence_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            display(HTML(f"<iframe src='{cooccurrence_file}' width='100%' height='500px'></iframe>"))
                    except:
                        print(f"无法显示共现关键词可视化: {cooccurrence_file}")

                    try:
                        with open(inst_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            display(HTML(f"<iframe src='{inst_file}' width='100%' height='500px'></iframe>"))
                    except:
                        print(f"无法显示机构分析可视化: {inst_file}")

            except Exception as e:
                print(f"分析过程中出错: {str(e)}")

    # 注册按钮点击事件
    analyze_button.on_click(on_analyze_button_clicked)

    # 创建UI布局
    file_section = widgets.VBox([widgets.HTML("<h3>1. 选择数据文件</h3>"), file_selector])
    keyword_section = widgets.VBox([widgets.HTML("<h3>2. 输入关键词(最多10个)</h3>"), keyword_input])
    output_section = widgets.VBox([widgets.HTML("<h3>3. 设置输出目录</h3>"), output_dir])
    button_section = widgets.VBox([widgets.HTML("<h3>4. 开始分析</h3>"), analyze_button])

    # 组装UI
    ui = widgets.VBox([
        widgets.HTML("<h1>文献关键词探究分析工具</h1>"),
        widgets.HTML("<p>本工具可以分析文献中的关键词共现关系、高引论文和发表机构分布。</p>"),
        file_section,
        keyword_section,
        output_section,
        button_section,
        output
    ])

    return ui
