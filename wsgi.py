import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from webapp.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
