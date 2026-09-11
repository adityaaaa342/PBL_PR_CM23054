import os
import sys

# Add root folder to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# WSGI Middleware to fix Vercel's rewrite pathing
class VercelPathFix:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # Check if Vercel set matched path header
        matched_path = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_VERCEL_MATCHED_PATH')
        
        path_info = environ.get('PATH_INFO', '')
        
        # If visiting root, Vercel often rewrites path_info to /api/index
        if path_info in ('/api/index', '/api/index.py', '/api'):
            environ['PATH_INFO'] = '/'
        elif matched_path and matched_path not in ('/api/index', '/api/index.py', '/api'):
            environ['PATH_INFO'] = matched_path
            
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFix(app.wsgi_app)
