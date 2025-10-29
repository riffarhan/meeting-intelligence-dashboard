# 🎯 Meeting Intelligence Dashboard

AI-powered analytics dashboard for analyzing 1:1 and team meetings using the CHAT + ACT frameworks.

![Dashboard Preview](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)

## 🌟 Features

### 1:1 Meeting Analysis
- **CHAT Framework**: Measures Connect, Hear, Ask, Transform dimensions
- **ACT Predictions**: Predicts Engagement, Readiness, Ownership outcomes
- **Psychological Safety**: Analyzes vulnerability, idea sharing, and challenges
- **Question Quality**: Evaluates open-ended vs closed questions
- **Sentiment Analysis**: Tracks emotional arc throughout conversations
- **Power Dynamics**: Detects hedging vs certainty patterns
- **Action Items**: Automatically extracts commitments

### Team Meeting Analysis
- **Participation Balance**: Pie charts showing word distribution
- **Inclusion Score**: Gauge measuring balanced participation
- **Speaker Interactions**: Network diagram of conversation flow
- **Decision Quality**: Tracks action items and commitments
- **Topic Coverage**: Identifies main discussion themes

### Advanced Analytics
- **Mastery Levels**: 4-level CHAT proficiency tracking
- **Trend Analysis**: Time-series visualization of metrics
- **Comparative Analysis**: Side-by-side person comparison
- **AI Coaching**: Personalized recommendations for improvement
- **Linguistic Diversity**: Vocabulary richness metrics

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/meeting-intelligence-dashboard.git
cd meeting-intelligence-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run bonus_quest_complete.py
```

4. **Open your browser** to `http://localhost:8501`

### Option 2: Use Streamlit Cloud

Click here to view the live dashboard: [Live Demo]([https://your-app-name.streamlit.app](https://meeting-intelligence-dashboard-w4rz4ctwfxlpr75bgvwwjf.streamlit.app/))

## 📁 Meeting File Format

### 1:1 Meetings
- **Filename**: `FirstName_LastName_DDMmmYY.docx` (e.g., `Alex_Rodriguez_15Oct24.docx`)
- **Format**: 
```
Sarah: How are you doing this week?
Alex: I'm doing well, thanks for asking...
Sarah: That's great to hear. What are your priorities?
Alex: My main focus is...
```

### Team Meetings
- **Filename**: Any `.docx` file with 3+ speakers
- **Format**:
```
[Speaker 1] (0:00 - 0:15)
Let's start with project updates...

[Speaker 2] (0:15 - 0:45)
I've completed the backend integration...

[Speaker 3] (0:45 - 1:20)
From a design perspective...
```

## 📊 Usage

1. **Upload Meetings**: Place `.docx` files in the `meetings/` folder
2. **Select Meeting Type**: Dashboard auto-detects 1:1 vs team meetings
3. **Explore Analytics**: Navigate through tabs for different insights
4. **Get Recommendations**: View AI-powered coaching suggestions

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **NLP**: Transformers (HuggingFace), PyTorch
- **Document Parsing**: python-docx

## 📈 Key Metrics Explained

### CHAT Framework
- **Connect** (0-100%): Trust, empathy, goal clarity
- **Hear** (0-100%): Active listening, validation
- **Ask** (0-100%): Question quality and depth
- **Transform** (0-100%): Action, empowerment, accountability

### ACT Predictions
- **Engagement** (0-100%): Emotional investment
- **Readiness** (0-100%): Preparedness to act
- **Ownership** (0-100%): Personal accountability

### Team Metrics
- **Inclusion Score** (0-100%): Participation balance
- **Decision Quality** (0-100%): Action item clarity

## 🔧 Configuration

### Customize Settings

Edit sidebar settings in the app:
- Show/hide detailed analysis
- Show/hide action items
- Filter by person
- Select metrics to track

### Advanced Configuration

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
```

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - How to deploy to GitHub & Streamlit Cloud
- [API Reference](docs/API.md) - Function documentation
- [User Guide](docs/USER_GUIDE.md) - Detailed usage instructions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@riffarhan]((https://github.com/riffarhan))
- LinkedIn: [Arif Farhan Bukhori](https://www.linkedin.com/in/farhanbukhori/)

## 🙏 Acknowledgments

- CHAT Framework for conversational leadership
- ACT Framework for outcome prediction
- Streamlit for the amazing framework
- HuggingFace for NLP models

## 🔮 Roadmap

- [ ] Real-time meeting analysis
- [ ] Integration with Zoom/Teams
- [ ] Multi-language support
- [ ] Custom framework builder
- [ ] Export reports to PDF
- [ ] Email digest summaries

---

⭐ **Star this repo** if you find it helpful!

🎯 **[Try the Live Demo](https://your-app-name.streamlit.app)**
