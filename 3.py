from graphviz import Digraph
import subprocess

dot = Digraph('测试')
dot.node("1", "Life's too short")
dot.node("2", "I learn Python")
dot.edge('1', '2')

try:
    dot.view()
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
    print("Trying to capture Graphviz output for debugging:")
    try:
        result = subprocess.run(['dot', '-Tpdf', '-o', '测试.pdf'], input=dot.source, capture_output=True, text=True,
                                encoding='gbk')
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
    except Exception as e:
        print(f"Error capturing output: {e}")
