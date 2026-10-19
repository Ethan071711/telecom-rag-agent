import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=SyntaxWarning)  # jieba 在 Python 3.12+ 下的转义序列警告
