import glob
import os
import asyncio

os.environ.setdefault("MINERU_LOG_LEVEL", "DEBUG")

from mineru.cli.gradio_app import to_markdown

async def convert_pdf_to_md(pdf_file):
    # 只转换前10页，可根据需要调整
    try:
        await to_markdown(pdf_file, end_pages=10)
        return True
    except TypeError as exc:
        print(f"{pdf_file} 转换失败: {exc}。请查看上方 mineru 日志。")
        return False
    except Exception as exc:
        print(f"{pdf_file} 转换失败: {exc}")
        return False

pdf_files = glob.glob("*.pdf")
for pdf_file in pdf_files:
    success = asyncio.run(convert_pdf_to_md(pdf_file))
    if success:
        print(f"{pdf_file} 已转换为 md（请在 output 目录下查找）")