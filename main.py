"""
PDF文件解析-上传-切片程序主界面

"""
import os
import pathlib

import gradio as gr

from dots_mocr.parser import do_parse
from milvus_db.db_operator import embedding_to_save
from splitters.splitter_md import MarkdownDirSplitter
from utils.common_utils import get_filename, delete_directory_if_non_empty, get_sorted_md_files
from utils.log_utils import log

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
log.info('切片程序主界面启动')


class FileEmbeddingWebUI:
    def __init__(self):
        self.pdf_path = None
        self.md_dir = None
        self.md_files = None
        self.file_contents = {}

    def upload_pdf(self, pdf_file):
        """文件上传控件内部方法，获取文件路径，保存到pdf_path，更新解析按钮状态"""
        log.info(f"上传pdf文件：{pdf_file}")
        self.pdf_path = pdf_file if pdf_file else None
        if self.pdf_path:
            return [
                f"PDF已上传: {os.path.basename(self.pdf_path)}",  # status
                gr.Button(interactive=True)
            ]
        else:
            return [
                "上传文件没有成功，请重新上传PDF文件",  # status
                gr.Button(interactive=False)
            ]

    def parse_pdf(self):
        """解析pdf文件，变成多个md文件"""
        # 创建解析后输出目录OUTPUT_DIR+上传pdf文件名
        parse_md_dir = os.path.join(OUTPUT_DIR, get_filename(self.pdf_path, False))
        # 创建前先删除同名目录
        delete_directory_if_non_empty(parse_md_dir)
        # 开始解析
        do_parse(input_path=self.pdf_path, num_thread=32, no_fitz_preprocess=True)
        # 解析完成后，保存类属性,默认展示第一个MarkDown，更新状态
        if os.path.isdir(parse_md_dir):
            self.md_dir = parse_md_dir
            log.info(f"PDF已解析，生成了{len(os.listdir(parse_md_dir))}个md文件")
            self.md_files = get_sorted_md_files(self.md_dir)
            log.info(f"PDF已解析，生成的MD文件列表：{self.md_files},存放位置：{self.md_dir}")
            # 把第一个文件的内容展示出来
            # 读取所有的md文件内容
            for f in self.md_files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        self.file_contents[f] = file.read()
                except Exception as e:
                    print(f"读取文件 {f} 时出错: {e}")
                    self.file_contents[f] = f"读取文件内容时出错: {e}"
            file_names = [os.path.basename(f) for f in self.md_files]
            return [
                f"解析完成，共 {len(self.md_files)} 个MD文件",  # status
                gr.Dropdown(choices=file_names, label="MD文件列表", interactive=True),  # file_dropdown
                gr.Button(interactive=False),  # parse_btn
                gr.update(interactive=True)  # save_btn - 使用 gr.update
            ]
        else:
            return [
                f"解析失败！",  # status
                gr.Dropdown(interactive=False),  # file_dropdown
                gr.Button(interactive=True),  # parse_btn
                gr.update(interactive=False)  # save_btn - 使用 gr.update
            ]

    def select_md_file(self, selected_file):
        """选择一个md文件，并展示它的内容"""
        log.info(f"选择文件：{selected_file}")
        if selected_file:
            show_file = None
            for f in self.md_files:
                if os.path.basename(f) == selected_file:
                    show_file = f
                    break
            if show_file and show_file in self.file_contents:
                return self.file_contents[show_file]
            else:
                return "没有找到该文件"
        else:
            return "文件内容加载失败,选择的文件不对"

    def embedding_save(self):
        splitter = MarkdownDirSplitter(images_output_dir=str(OUTPUT_DIR / 'images'))
        docs = splitter.process_md_dir(str(self.md_dir), source_filename=self.pdf_path)
        res = embedding_to_save(docs)
        for i, doc in enumerate(res):
            print(f"\n文档 #{i + 1}:")
            print(doc['text'], doc['image_path'])
        return f"成功存入 {len(res)} 个文档到Milvus"

    def create_web_ui(self):
        """创建一个构建多模态知识库的界面"""

        with gr.Blocks() as app:
            gr.Markdown("## PDF解析与知识库存储和构建")

            with gr.Row():
                pdf_upload = gr.File(label="上传PDF文件")
                parse_btn = gr.Button("解析PDF", variant="primary", interactive=False)

            status = gr.Textbox(label="状态", value="等待操作...", interactive=False)

            with gr.Row():
                # MD文件列表
                file_dropdown = gr.Dropdown(choices=[], label="MD文件列表", interactive=False)
                # MD文件中的内容
                content = gr.Textbox(label="文件内容", lines=20, interactive=False)

            save_btn = gr.Button("存入知识库", variant="huggingface", interactive=False)

            # 绑定按钮点击事件
            pdf_upload.change(
                fn=self.upload_pdf,
                inputs=pdf_upload,
                outputs=[status, parse_btn]
            )

            parse_btn.click(
                fn=self.parse_pdf,
                inputs=[],
                outputs=[status, file_dropdown, parse_btn, save_btn]
            )

            file_dropdown.change(
                fn=self.select_md_file,
                inputs=file_dropdown,
                outputs=content
            )
            save_btn.click(
                fn=self.embedding_save,
                inputs=[],
                outputs=status
            )

        return app


if __name__ == '__main__':
    app = FileEmbeddingWebUI()
    webui_app = app.create_web_ui()
    webui_app.launch()
