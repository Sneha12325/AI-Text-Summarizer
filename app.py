# app.py - Production-ready Flask Application
from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline
import hashlib
import redis
import logging
from functools import wraps
import time
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Redis configuration (use environment variables in production)
try:
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True,
        socket_timeout=5
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - caching disabled")


# Singleton Model Manager
class ModelManager:
    """Ensures model is loaded once and reused across requests"""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        if self._model is None:
            logger.info("Loading BART model...")
            start = time.time()
            self._model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU, change to 0 for GPU
            )
            logger.info(f"Model loaded in {time.time() - start:.2f}s")
        return self._model


model_manager = ModelManager()


# Rate Limiting Decorator
def rate_limit(max_requests=10, window=60):
    """Rate limit: max_requests per window (seconds)"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not REDIS_AVAILABLE:
                return f(*args, **kwargs)
            
            ip = request.remote_addr
            key = f"rate_limit:{ip}"
            
            try:
                current = redis_client.get(key)
                if current and int(current) >= max_requests:
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'message': f'Max {max_requests} requests per {window}s'
                    }), 429
                
                pipe = redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window)
                pipe.execute()
            except Exception as e:
                logger.error(f"Rate limit error: {e}")
            
            return f(*args, **kwargs)
        return wrapped
    return decorator


# Caching utilities
def get_cache_key(text, length):
    """Generate cache key from text and summary length"""
    content = f"{text[:1000]}:{length}"  # Use first 1000 chars for key
    return f"summary:{hashlib.md5(content.encode()).hexdigest()}"


def get_cached_summary(text, length):
    """Retrieve cached summary if available"""
    if not REDIS_AVAILABLE:
        return None
    
    try:
        key = get_cache_key(text, length)
        cached = redis_client.get(key)
        if cached:
            logger.info("Cache hit")
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Cache retrieval error: {e}")
    
    return None


def cache_summary(text, length, summary_data):
    """Cache summary for 24 hours"""
    if not REDIS_AVAILABLE:
        return
    
    try:
        key = get_cache_key(text, length)
        redis_client.setex(
            key,
            86400,  # 24 hours
            json.dumps(summary_data)
        )
        logger.info("Summary cached")
    except Exception as e:
        logger.error(f"Cache storage error: {e}")


# Input validation
def validate_input(text, length):
    """Validate and sanitize input"""
    errors = []
    
    if not text or not text.strip():
        errors.append("Text cannot be empty")
    
    if len(text) > 10000:
        errors.append("Text too long (max 10,000 characters)")
    
    if len(text.split()) < 30:
        errors.append("Text too short (minimum 30 words)")
    
    if length not in ['short', 'medium', 'long']:
        errors.append("Invalid length option")
    
    return errors


# Summarization logic
def generate_summary(text, length='medium'):
    """Generate summary with error handling"""
    try:
        # Check cache first
        cached = get_cached_summary(text, length)
        if cached:
            cached['cached'] = True
            return cached
        
        # Configure summary parameters
        length_config = {
            'short': {'max_length': 50, 'min_length': 20},
            'medium': {'max_length': 130, 'min_length': 30},
            'long': {'max_length': 250, 'min_length': 50}
        }
        
        config = length_config.get(length, length_config['medium'])
        
        # Get model and generate
        model = model_manager.get_model()
        start_time = time.time()
        
        result = model(
            text,
            max_length=config['max_length'],
            min_length=config['min_length'],
            do_sample=False,
            truncation=True
        )
        
        inference_time = time.time() - start_time
        summary_text = result[0]['summary_text']
        
        # Prepare response
        response = {
            'summary': summary_text,
            'original_length': len(text.split()),
            'summary_length': len(summary_text.split()),
            'compression_ratio': round(len(summary_text) / len(text) * 100, 1),
            'inference_time': round(inference_time, 2),
            'cached': False,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache the result
        cache_summary(text, length, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise


# Routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/summarize', methods=['POST'])
@rate_limit(max_requests=10, window=60)
def summarize():
    """Main API endpoint for text summarization"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        length = data.get('length', 'medium').lower()
        
        # Validate input
        errors = validate_input(text, length)
        if errors:
            return jsonify({'error': errors[0]}), 400
        
        # Generate summary
        result = generate_summary(text, length)
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        model = model_manager.get_model()
        redis_status = "connected" if REDIS_AVAILABLE else "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'redis': redis_status,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get cache statistics"""
    if not REDIS_AVAILABLE:
        return jsonify({'error': 'Redis not available'}), 503
    
    try:
        info = redis_client.info()
        return jsonify({
            'cache_keys': redis_client.dbsize(),
            'memory_used': info.get('used_memory_human', 'N/A'),
            'connected_clients': info.get('connected_clients', 0)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# HTML Template with modern UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text Summarizer - Production Ready</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            margin: 10px 5px;
            font-size: 0.9em;
        }
        .content { padding: 40px; }
        .form-group { margin-bottom: 25px; }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 200px;
            transition: border 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .length-options {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .length-option {
            flex: 1;
            min-width: 150px;
        }
        .length-option input { display: none; }
        .length-option label {
            display: block;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .length-option input:checked + label {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            display: none;
        }
        .result.show { display: block; }
        .result h3 { color: #667eea; margin-bottom: 15px; }
        .result-text {
            line-height: 1.8;
            color: #333;
            font-size: 16px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .metric {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            display: none;
        }
        .error.show { display: block; }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        .loader.show { display: block; }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .char-count {
            text-align: right;
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 AI Text Summarizer</h1>
            <p>Production-ready NLP application with caching & rate limiting</p>
            <div>
                <span class="badge">BART Model</span>
                <span class="badge">Redis Cache</span>
                <span class="badge">Rate Limited</span>
            </div>
        </div>
        
        <div class="content">
            <form id="summarizerForm">
                <div class="form-group">
                    <label for="text">Enter Text to Summarize</label>
                    <textarea id="text" placeholder="Paste your text here (minimum 30 words, maximum 10,000 characters)..." required></textarea>
                    <div class="char-count" id="charCount">0 / 10,000 characters</div>
                </div>
                
                <div class="form-group">
                    <label>Summary Length</label>
                    <div class="length-options">
                        <div class="length-option">
                            <input type="radio" name="length" id="short" value="short">
                            <label for="short">
                                <strong>Short</strong><br>
                                <small>~20-50 words</small>
                            </label>
                        </div>
                        <div class="length-option">
                            <input type="radio" name="length" id="medium" value="medium" checked>
                            <label for="medium">
                                <strong>Medium</strong><br>
                                <small>~30-130 words</small>
                            </label>
                        </div>
                        <div class="length-option">
                            <input type="radio" name="length" id="long" value="long">
                            <label for="long">
                                <strong>Long</strong><br>
                                <small>~50-250 words</small>
                            </label>
                        </div>
                    </div>
                </div>
                
                <button type="submit" id="submitBtn">Generate Summary</button>
            </form>
            
            <div class="loader" id="loader"></div>
            <div class="error" id="error"></div>
            
            <div class="result" id="result">
                <h3>📝 Summary</h3>
                <div class="result-text" id="summaryText"></div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="originalWords">-</div>
                        <div class="metric-label">Original Words</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="summaryWords">-</div>
                        <div class="metric-label">Summary Words</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="compression">-</div>
                        <div class="metric-label">Compression</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="inferenceTime">-</div>
                        <div class="metric-label">Time (s)</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('summarizerForm');
        const textArea = document.getElementById('text');
        const charCount = document.getElementById('charCount');
        const loader = document.getElementById('loader');
        const error = document.getElementById('error');
        const result = document.getElementById('result');
        const submitBtn = document.getElementById('submitBtn');

        // Character counter
        textArea.addEventListener('input', () => {
            const count = textArea.value.length;
            charCount.textContent = `${count} / 10,000 characters`;
            charCount.style.color = count > 10000 ? '#c33' : '#666';
        });

        // Form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const text = textArea.value.trim();
            const length = document.querySelector('input[name="length"]:checked').value;
            
            // Reset UI
            error.classList.remove('show');
            result.classList.remove('show');
            loader.classList.add('show');
            submitBtn.disabled = true;
            
            try {
                const response = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text, length })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Failed to generate summary');
                }
                
                // Display results
                const summary = data.data;
                document.getElementById('summaryText').textContent = summary.summary;
                document.getElementById('originalWords').textContent = summary.original_length;
                document.getElementById('summaryWords').textContent = summary.summary_length;
                document.getElementById('compression').textContent = summary.compression_ratio + '%';
                document.getElementById('inferenceTime').textContent = summary.inference_time;
                
                result.classList.add('show');
                
                // Show cache indicator
                if (summary.cached) {
                    document.getElementById('inferenceTime').innerHTML = 
                        summary.inference_time + '<br><small style="color: #28a745;">✓ Cached</small>';
                }
                
            } catch (err) {
                error.textContent = err.message;
                error.classList.add('show');
            } finally {
                loader.classList.remove('show');
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
'''


# ...existing code...
if __name__ == '__main__':
    # Do not preload the model at startup — it will load on first request.
    logger.info("Application ready! (model will load on first request)")
    app.run(debug=False, host='0.0.0.0', port=5000)
# ...existing code...