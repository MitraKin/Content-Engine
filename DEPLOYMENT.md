# Production Deployment Guide

## Security Considerations

### 1. Secret Key
Change the default secret key in production:
```bash
export SECRET_KEY='your-secure-random-key-here'
```

Or set it in your production configuration.

### 2. Debug Mode
Debug mode is disabled by default. Only enable for development:
```bash
export FLASK_DEBUG=true  # Only for development
```

### 3. WSGI Server
Do not use the built-in Flask server in production. Use a production-grade WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 4. Environment Variables
Recommended environment variables for production:
- `SECRET_KEY`: Strong secret key for session encryption
- `FLASK_DEBUG`: Should be false or unset in production
- `DOCUMENTS_PATH`: Optional path to documents directory (defaults to ./Documents)

## Systemd Service Example

Create a systemd service file for automatic startup:

```ini
[Unit]
Description=AI PDF Reader Flask Application
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/Content-Engine
Environment="SECRET_KEY=your-secret-key"
Environment="FLASK_DEBUG=false"
ExecStart=/usr/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

## Nginx Reverse Proxy

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static {
        alias /path/to/Content-Engine/static;
    }
}
```

## Docker Deployment (Optional)

You can also containerize the application using Docker for easier deployment.
