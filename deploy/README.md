# Blackwell Server Deployment

This directory contains deployment files for the Blackwell inference server.

## Files

- `blackwell.service` - systemd service unit
- `nginx.conf` - nginx reverse proxy configuration
- `monitor.sh` - server monitoring script

## Quick Setup

### 1. Install Service

```bash
# Copy service file
sudo cp blackwell.service /etc/systemd/system/

# Edit to set correct user
sudo nano /etc/systemd/system/blackwell.service

# Reload and start
sudo systemctl daemon-reload
sudo systemctl enable blackwell
sudo systemctl start blackwell

# Check status
sudo systemctl status blackwell
```

### 2. Nginx Proxy (Optional)

```bash
# Copy nginx config
sudo cp nginx.conf /etc/nginx/sites-available/blackwell

# Enable site
sudo ln -s /etc/nginx/sites-available/blackwell /etc/nginx/sites-enabled/

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

Now server accessible at port 8080 instead of 8123.

### 3. Monitoring

```bash
# Single check
./deploy/monitor.sh

# Continuous monitoring
./deploy/monitor.sh --continuous

# With custom interval (10 seconds)
INTERVAL=10 ./deploy/monitor.sh --continuous
```

## Production Checklist

- [ ] GPU memory sufficient (16 GB recommended)
- [ ] No other GPU processes running
- [ ] CUDA drivers installed
- [ ] Weights in correct location
- [ ] Firewall allows port 8123 (or 8080 with nginx)
- [ ] systemd service auto-start enabled
- [ ] Monitoring configured
- [ ] Log rotation configured

## Troubleshooting

### Server Won't Start

```bash
# Check GPU availability
nvidia-smi

# Check logs
journalctl -u blackwell -n 50

# Check port
lsof -i :8123
```

### Out of Memory

```bash
# Kill GPU processes
killall -9 hashcat
pkill -9 inference
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

### Slow Performance

```bash
# Check GPU utilization
nvidia-smi dmon

# Monitor server
./deploy/monitor.sh --continuous
```

## Security

- Use nginx for TLS termination
- Configure firewall
- Limit rate with nginx rate limiting
- Monitor error rates

## Backup

```bash
# Backup weights
tar -czf weights_backup.tar.gz weights_int4_qwen3_8b/

# Backup config
cp /etc/systemd/system/blackwell.service ~/blackwell.service.bak
```