# 🦌 Wildlife Camera AI

**Real-time AI-powered wildlife monitoring system with edge deployment, smart alerts, and web dashboard.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00D4FF.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Overview

A complete full-stack wildlife monitoring system that uses computer vision to detect and log wildlife activity in real-time. The system features automatic species identification, intelligent alerting with cooldown mechanisms, Telegram bot notifications, persistent database logging, and a beautiful web dashboard for data visualization.

**Live Demo**: [View Dashboard Screenshot](#screenshots)

---

## ✨ Key Features

### 🤖 AI Detection
- **Real-time YOLOv8 inference** at 15-20 FPS on CPU
- **10 wildlife species** recognized (expandable)
- **Confidence-based filtering** with adjustable thresholds
- **Rarity classification** (Common/Interesting/Rare)

### 💾 Data Management
- **SQLite database** with indexed queries
- **Automatic snapshot saving** organized by species
- **Session tracking** with unique IDs
- **Advanced queries** (by date, species, rarity)
- **CSV export** capabilities

### 🔔 Smart Alerts
- **Multi-channel notifications** (Console, File, Telegram)
- **Cooldown system** prevents alert spam
- **Rarity-based triggering** (alert only on rare animals)
- **Rate limiting** with configurable thresholds
- **Photo attachments** via Telegram

### 🌐 Web Dashboard
- **Live camera feed** with real-time updates
- **Interactive charts** (hourly activity patterns)
- **Species statistics** cards with drill-down
- **Recent detections** gallery
- **Auto-refreshing** data (3-second intervals)

### ⚙️ Configuration
- **Preset configs** for different scenarios
- **JSON-based settings** with save/load
- **CLI overrides** for quick adjustments
- **Type-safe** dataclass configuration

---

## 📊 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Framework** | YOLOv8 (Ultralytics) | Object detection |
| **Computer Vision** | OpenCV | Video processing |
| **Backend API** | FastAPI | REST API + WebSocket |
| **Database** | SQLite3 | Persistent storage |
| **Bot Integration** | Telegram Bot API | Mobile notifications |
| **Frontend** | React + Tailwind CSS | Web dashboard |
| **Config** | Python Dataclasses | Type-safe settings |

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Webcam or video files
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/wildlife-camera-ai.git
cd wildlife-camera-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the system**
```bash
# Basic webcam detection
python main.py

# With Telegram alerts
python main.py --telegram-token YOUR_TOKEN --telegram-chat YOUR_CHAT_ID

# Process video file
python main.py --source wildlife_video.mp4
```

4. **Open the dashboard**
- Open `dashboard.html` in your browser
- View live detections at http://localhost:8000

---

## 📁 Project Structure

```
wildlife-camera/
├── detector.py              # YOLOv8 detection engine
├── config.py                # Configuration management
├── database.py              # SQLite logging & queries
├── alert_manager.py         # Smart alert system
├── telegram_bot.py          # Telegram integration
├── api_server.py            # FastAPI backend
├── main.py                  # Main application
├── dashboard.html           # Web dashboard
├── requirements.txt         # Python dependencies
└── data/
    ├── wildlife.db          # SQLite database
    ├── alerts.log           # Alert history
    └── snapshots/           # Detection images
        ├── cat/
        ├── dog/
        ├── bird/
        └── deer/
```

---

## 🎮 Usage Examples

### Command Line Options

```bash
# Higher confidence threshold
python main.py --confidence 0.7

# Use larger model (more accurate, slower)
python main.py --model s

# Disable web API/dashboard
python main.py --no-api

# Load custom configuration
python main.py --config my_settings.json

# Save current configuration
python main.py --save-config production.json
```

### Interactive Controls

While running:
- **`q`** - Quit application
- **`s`** - Save manual snapshot
- **`p`** - Pause/resume processing
- **`d`** - Display live statistics

---

## 📸 Screenshots

### Dashboard
![Dashboard](docs/dashboard.png)
*Real-time web dashboard with live feed, statistics, and activity charts*

### Detection Example
![Detection](docs/detection.png)
*Live camera feed with bounding boxes and species identification*

### Telegram Alerts
![Telegram](docs/telegram.png)
*Instant notifications with photos sent to your phone*

---

## 🔧 Configuration

### Config File Example

```json
{
  "detection": {
    "model_size": "n",
    "confidence_threshold": 0.5
  },
  "alert": {
    "enabled": true,
    "cooldown_seconds": 30,
    "alert_on_rare": true,
    "telegram_enabled": true,
    "telegram_bot_token": "YOUR_TOKEN",
    "telegram_chat_id": "YOUR_CHAT_ID"
  },
  "recording": {
    "save_snapshots": true,
    "clips_dir": "data/clips",
    "snapshots_dir": "data/snapshots"
  }
}
```

### Environment Variables

```bash
WILDLIFE_DB_PATH=data/wildlife.db
WILDLIFE_API_PORT=8000
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## 📊 API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Overall statistics |
| `/api/detections/recent` | GET | Recent detections |
| `/api/species` | GET | All species list |
| `/api/species/{name}` | GET | Species details |
| `/api/activity/hourly` | GET | Hourly activity data |
| `/api/activity/daily` | GET | Daily summary |
| `/api/rare-sightings` | GET | Rare animals only |
| `/ws/stream` | WebSocket | Live video feed |

### API Documentation

Interactive API docs available at: `http://localhost:8000/docs`

---

## 🤖 Telegram Bot Setup

1. **Create bot via @BotFather**
```
/newbot
Choose name: My Wildlife Camera
Choose username: my_wildlife_bot
Copy the bot token
```

2. **Get your Chat ID**
```python
from telegram_bot import get_chat_id
chat_id = get_chat_id('YOUR_BOT_TOKEN')
print(f"Your chat ID: {chat_id}")
```

3. **Configure**
```bash
python main.py --telegram-token YOUR_TOKEN --telegram-chat YOUR_CHAT_ID
```

---

## 🎓 Learning Outcomes

This project demonstrates expertise in:

- ✅ **Machine Learning**: YOLOv8 deployment, real-time inference
- ✅ **Computer Vision**: Video processing, object detection
- ✅ **Backend Development**: FastAPI, WebSocket, REST APIs
- ✅ **Database Design**: SQLite schema, indexed queries
- ✅ **API Integration**: Telegram Bot API, webhooks
- ✅ **Frontend Development**: React, real-time updates
- ✅ **System Design**: Modular architecture, state management
- ✅ **DevOps**: Configuration management, CLI tools

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Inference Speed** | 15-20 FPS (CPU) |
| **Latency** | 50-70ms per frame |
| **Accuracy** | 85-95% (clear images) |
| **Memory Usage** | ~500MB |
| **Database Queries** | <1ms (indexed) |

---

## 🚀 Deployment

### Raspberry Pi

```bash
# Install on Raspberry Pi
sudo apt-get update
sudo apt-get install python3-opencv
pip3 install -r requirements.txt

# Run on boot (systemd)
sudo cp wildlife-camera.service /etc/systemd/system/
sudo systemctl enable wildlife-camera
sudo systemctl start wildlife-camera
```

### Docker

```bash
# Build image
docker build -t wildlife-camera .

# Run container
docker run -d \
  --device=/dev/video0 \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  wildlife-camera
```

---

## 🔮 Future Enhancements

- [ ] Video clip recording (15-sec triggered recordings)
- [ ] ONNX model optimization (2x faster inference)
- [ ] Custom model training on specific wildlife
- [ ] Multi-camera support
- [ ] Cloud sync (optional)
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Species identification fine-tuning

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**SANUSI MB**
- Portfolio: [digitalmustiii.vercel.app](https://digitalmustiii.vercel.app)
- GitHub: [@Digitalmustiii](https://github.com/Digitalmustiii)
- LinkedIn: [Mustapha Sanusi B](https://linkedin.com/in/sbmustapha)

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [OpenCV](https://opencv.org/) for computer vision tools
- [Telegram Bot API](https://core.telegram.org/bots/api) for notifications

---

## 📧 Contact

For questions or feedback, please open an issue or contact me directly.

**Project Link**: [https://github.com/yourusername/wildlife-camera-ai](https://github.com/Digitalmustiii/wildlife-camera-ai)

---

⭐ If you found this project helpful, please consider giving it a star!
