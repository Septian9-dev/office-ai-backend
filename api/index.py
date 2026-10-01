import sys
import os

# Tambahkan root directory ke sys.path Vercel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
