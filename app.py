# BONUS QUEST: Complete Dashboard with Team Meeting Analysis
# This fully integrates the 1:1 dashboard to also handle team conversations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from docx import Document
import re
from datetime import datetime
from collections import Counter, defaultdict
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try transformer imports
try:
    import torch
except Exception as _e:
    st.error("PyTorch required. Install with: pip install torch")
    raise

from transformers import pipeline

st.set_page_config(
    page_title="Meeting Intelligence Dashboard (1:1 + Team)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def get_pipelines():
    device = 0 if torch.cuda.is_available() else -1
    sa = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)
    emo = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True, device=device)
    return sa, emo

sentiment_analyzer, emotion_classifier = get_pipelines()

def transformer_sentiment_scalar(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    chunk_size = 512
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    vals = []
    try:
        context = torch.inference_mode()
    except AttributeError:
        class _Noop:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        context = _Noop()
    with context:
        for ch in chunks:
            res = sentiment_analyzer(ch)[0]
            label = res["label"].upper()
            score = float(res["score"])
            if "POS" in label:
                vals.append(score)
            elif "NEG" in label:
                vals.append(-score)
            else:
                vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0

# Enhanced CSS with better design
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.12);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Info boxes */
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-left: 5px solid #667eea;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%);
        padding: 1.2rem;
        border-left: 5px solid #ffc107;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #d4edda 100%);
        padding: 1.2rem;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .danger-box {
        background: linear-gradient(135deg, #ffe6e6 0%, #f8d7da 100%);
        padding: 1.2rem;
        border-left: 5px solid #dc3545;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .team-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem;
        border-left: 5px solid #2196f3;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Progress bars */
    .progress-bar-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 24px;
        margin: 10px 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: white;
        transition: width 0.3s ease;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-success {
        background-color: #28a745;
        color: white;
    }
    .badge-warning {
        background-color: #ffc107;
        color: #000;
    }
    .badge-danger {
        background-color: #dc3545;
        color: white;
    }
    .badge-info {
        background-color: #17a2b8;
        color: white;
    }
    .badge-team {
        background-color: #2196f3;
        color: white;
    }
    .badge-one-on-one {
        background-color: #9c27b0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class ConversationMetrics:
    """Data class for conversation metrics"""
    total_words: int
    manager_words: int
    report_words: int
    talk_ratio: float
    listen_ratio: float
    turn_count: int
    avg_turn_length: float
    manager_sentiment: float
    report_sentiment: float
    sentiment_arc: List[float]
    questions_total: int
    manager_questions: int
    question_density: float


@dataclass
class TeamMetrics:
    """Metrics specific to team meetings"""
    participant_count: int
    participation_distribution: Dict[str, float]
    turn_distribution: Dict[str, int]
    dominance_score: float
    inclusion_score: float
    collaboration_patterns: Dict[str, Any]
    decision_quality: float
    topic_coverage: List[str]


class AdvancedNLPAnalyzer:
    """Enhanced NLP analysis with more sophisticated metrics"""

    @staticmethod
    def calculate_linguistic_diversity(text: str) -> Dict[str, float]:
        words = text.lower().split()
        unique_words = set(words)
        return {
            'type_token_ratio': len(unique_words) / len(words) if words else 0,
            'unique_words': len(unique_words),
            'total_words': len(words),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
        }

    @staticmethod
    def detect_power_dynamics(manager_text: str, report_text: str) -> Dict[str, Any]:
        hedges = ['maybe', 'perhaps', 'probably', 'might', 'could be', 'i think', 'kind of', 'sort of']
        manager_hedges = sum(manager_text.lower().count(h) for h in hedges)
        report_hedges = sum(report_text.lower().count(h) for h in hedges)

        certainty = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously']
        manager_certainty = sum(manager_text.lower().count(c) for c in certainty)
        report_certainty = sum(report_text.lower().count(c) for c in certainty)

        return {
            'manager_hedges': manager_hedges,
            'report_hedges': report_hedges,
            'manager_certainty': manager_certainty,
            'report_certainty': report_certainty,
            'psychological_safety_score': (report_certainty - report_hedges) / max(len(report_text.split()), 1) * 100,
            'power_balance': 'balanced' if abs(manager_hedges - report_hedges) < 3 else 'imbalanced'
        }

    @staticmethod
    def analyze_sentiment_arc(dialogue: List[Dict]) -> Tuple[List[float], List[List[Tuple[str, float]]]]:
        """
        Returns:
          arc: List[float] of sentiment (-1..1-ish)
          emotions: List per segment of top-2 (label, score)
        """
        arc: List[float] = []
        emotions: List[List[Tuple[str, float]]] = []

        if not dialogue:
            return [0.0] * 5, []

        segments = np.array_split(dialogue, 5)
        for segment in segments:
            text = ' '.join([d['text'] for d in segment])[:512]

            # Sentiment (use module-level pipelines already created)
            result = sentiment_analyzer(text)[0]
            label = result['label'].upper()
            # Map 3-class labels → signed score
            if 'POS' in label:
                score = float(result['score'])
            elif 'NEG' in label:
                score = -float(result['score'])
            else:  # NEU
                score = 0.0
            arc.append(score)

            # Emotions: top-2
            e_scores = emotion_classifier(text)[0]
            e_top2 = sorted(e_scores, key=lambda x: x['score'], reverse=True)[:2]
            emotions.append([(e['label'], round(float(e['score']), 2)) for e in e_top2])

        return arc, emotions

    @staticmethod
    def extract_action_items(text: str) -> List[str]:
        action_patterns = [
            r'i will ([\w\s]+)',
            r"i'll ([\w\s]+)",
            r'going to ([\w\s]+)',
            r'plan to ([\w\s]+)',
            r'need to ([\w\s]+)',
            r'should ([\w\s]+)',
        ]
        actions = []
        for pattern in action_patterns:
            for match in re.finditer(pattern, text.lower()):
                action = match.group(1).strip()
                if 1 <= len(action.split()) <= 10:
                    actions.append(action)
        return list(dict.fromkeys(actions))[:5]


def analyze_chat_framework(dialogue: List[Dict], sarah_utterances: List[str],
                          report_utterances: List[str]) -> Dict[str, float]:
    """Enhanced CHAT Framework analysis with weighted scoring"""
    
    sarah_text = ' '.join(sarah_utterances).lower()
    report_text = ' '.join(report_utterances).lower()
    
    scores = {
        'connect': 0.0,
        'hear': 0.0,
        'ask': 0.0,
        'transform': 0.0,
        'overall': 0.0,
        'mastery_level': 1
    }
    
    # CONNECT: Trust, empathy, goal clarity (0-1 score)
    connect_signals = {
        'empathy': [r'\bhow\s+(are|is)\s+you', r'\bfeeling\b', r'\bcare\b', r'\bconcerned\b'],
        'goal_clarity': [r'\bgoal\b', r'\bobjective\b', r'\baim\b', r'\bpurpose\b'],
        'personal_check_in': [r'weekend', r'family', r'wellbeing', r'doing okay'],
        'confidentiality': [r'between us', r'confidential', r'private', r'safe space']
    }
    
    connect_count = 0
    for category, patterns in connect_signals.items():
        for pattern in patterns:
            if re.search(pattern, sarah_text):
                connect_count += 1
    scores['connect'] = min(connect_count / 8, 1.0)  # Normalize to 0-1
    
    # HEAR: Active listening, understanding (0-1 score)
    hear_signals = {
        'paraphrasing': [r"so what (you're|you are) saying", r"what i('m| am) hearing",
                        r"if i understand", r"let me (check|see) if"],
        'clarifying': [r"can you (tell me more|elaborate|explain)", r"what do you mean",
                      r"help me understand", r"clarify"],
        'validation': [r"that makes sense", r"i (see|understand)", r"(right|exactly|yes),"],
        'emotion_naming': [r"sounds? (like you|frustrating|exciting|concerning)"]
    }
    
    hear_count = 0
    for category, patterns in hear_signals.items():
        for pattern in patterns:
            if re.search(pattern, sarah_text):
                hear_count += 1
    
    # Penalize interruptions (check for very short back-and-forths)
    short_exchanges = sum(1 for d in dialogue if d['speaker'] == 'Sarah' and d['word_count'] < 3)
    interruption_penalty = min(short_exchanges / len(dialogue), 0.3)
    scores['hear'] = max(min(hear_count / 10, 1.0) - interruption_penalty, 0.0)
    
    # ASK: Question quality and depth (0-1 score)
    question_count = sarah_text.count('?')
    
    # Open-ended questions (higher value)
    open_ended = len(re.findall(r'\b(what|how|why|tell me|describe|explain)\b.*\?', sarah_text))
    
    # Closed questions (lower value)
    closed = len(re.findall(r'\b(is|are|do|does|did|will|would|can|could|should)\b.*\?', sarah_text))
    
    # Future-focused questions (highest value)
    future_focused = len(re.findall(r'\b(what would|how could|what if|imagine)\b.*\?', sarah_text))
    
    # Weighted scoring
    question_score = (open_ended * 1.0 + future_focused * 1.5 - closed * 0.3) / max(question_count, 1)
    scores['ask'] = min(max(question_score, 0), 1.0)
    
    # TRANSFORM: Action, empowerment, accountability (0-1 score)
    transform_signals = {
        'action_items': [r"what will you", r"what's your (next|first) step", r"by when",
                        r"who will", r"action item"],
        'empowerment': [r"what do you think", r"your decision", r"you choose",
                       r"what would you do", r"trust you"],
        'celebration': [r"great (job|work)", r"proud of", r"impressive", r"well done",
                       r"progress", r"improvement"],
        'follow_up': [r"last time we", r"you mentioned (last|previously)", r"following up"]
    }
    
    transform_count = 0
    for category, patterns in transform_signals.items():
        for pattern in patterns:
            if re.search(pattern, sarah_text):
                transform_count += 1
    scores['transform'] = min(transform_count / 10, 1.0)
    
    # Overall CHAT score
    scores['overall'] = (scores['connect'] + scores['hear'] +
                        scores['ask'] + scores['transform']) / 4
    
    # Determine mastery level (1-4)
    if scores['overall'] >= 0.85:
        scores['mastery_level'] = 4  # Master
    elif scores['overall'] >= 0.70:
        scores['mastery_level'] = 3  # Experienced
    elif scores['overall'] >= 0.50:
        scores['mastery_level'] = 2  # Emerging
    else:
        scores['mastery_level'] = 1  # Beginning
    
    return scores


def predict_act_outcomes(dialogue: List[Dict], report_utterances: List[str],
                         report_sentiment: float) -> Dict[str, float]:
    """Enhanced ACT prediction with ML-inspired features"""
    
    report_text = ' '.join(report_utterances).lower()
    
    predictions = {
        'engagement': 0.5,
        'readiness': 0.5,
        'ownership': 0.5,
        'overall': 0.5,
        'confidence': 0.5
    }
    
    # ENGAGEMENT: Connection and emotional investment
    engagement_signals = {
        'positive_emotion': len(re.findall(r'\b(excited|happy|glad|love|enjoy|appreciate)\b', report_text)),
        'energy_words': len(re.findall(r'\b(definitely|absolutely|really|very|great|awesome)\b', report_text)),
        'future_talk': len(re.findall(r'\b(will|going to|planning|looking forward)\b', report_text)),
        'negative_emotion': len(re.findall(r'\b(frustrated|concerned|worried|difficult|struggling)\b', report_text)),
    }
    
    # Sentiment-weighted engagement
    base_engagement = 0.5 + (report_sentiment * 0.3)
    signal_boost = (engagement_signals['positive_emotion'] + engagement_signals['energy_words'] -
                   engagement_signals['negative_emotion']) / max(len(report_utterances), 1)
    
    predictions['engagement'] = min(max(base_engagement + signal_boost * 0.2, 0), 1)
    
    # READINESS: Preparedness to take action
    readiness_signals = {
        'commitment': len(re.findall(r"\b(i will|i'll|i'm going to|i plan to)\b", report_text)),
        'timeline': len(re.findall(r'\b(today|tomorrow|this week|by \w+day|next \w+)\b', report_text)),
        'resources': len(re.findall(r'\b(need|require|have|got|ready|prepared)\b', report_text)),
        'uncertainty': len(re.findall(r'\b(maybe|might|possibly|not sure|don\'t know)\b', report_text)),
    }
    
    readiness_score = (
        readiness_signals['commitment'] * 0.4 +
        readiness_signals['timeline'] * 0.3 +
        readiness_signals['resources'] * 0.2 -
        readiness_signals['uncertainty'] * 0.3
    ) / max(len(report_utterances), 1)
    
    predictions['readiness'] = min(max(0.5 + readiness_score, 0), 1)
    
    # OWNERSHIP: Personal accountability
    ownership_signals = {
        'personal_pronouns': len(re.findall(r'\bi (will|can|should|need)\b', report_text)),
        'problem_solving': len(re.findall(r'\b(solution|idea|approach|strategy|plan)\b', report_text)),
        'initiative': len(re.findall(r'\b(i propose|i suggest|what if i|i could try)\b', report_text)),
        'deflection': len(re.findall(r'\b(they should|someone should|we need to|hopefully)\b', report_text)),
    }
    
    ownership_score = (
        ownership_signals['personal_pronouns'] * 0.35 +
        ownership_signals['problem_solving'] * 0.25 +
        ownership_signals['initiative'] * 0.4 -
        ownership_signals['deflection'] * 0.3
    ) / max(len(report_utterances), 1)
    
    predictions['ownership'] = min(max(0.5 + ownership_score, 0), 1)
    
    # Overall ACT score
    predictions['overall'] = (predictions['engagement'] + predictions['readiness'] +
                             predictions['ownership']) / 3
    
    # Confidence in prediction (based on dialogue length and signals detected)
    signal_count = sum([sum(s.values()) for s in [engagement_signals, readiness_signals, ownership_signals]])
    predictions['confidence'] = min(signal_count / 20, 1.0)
    
    return predictions


def analyze_psychological_safety(report_utterances: List[str], dialogue: List[Dict]) -> Dict[str, Any]:
    """Enhanced psychological safety analysis"""
    
    report_text = ' '.join(report_utterances).lower()
    
    # Risk-taking indicators
    vulnerability = len(re.findall(r'\b(struggling|confused|don\'t know|need help|unsure|mistake|failed)\b', report_text))
    
    # Idea sharing
    idea_sharing = len(re.findall(r'\b(i think|my idea|what if|suggestion|propose|consider)\b', report_text))
    
    # Question asking by report
    report_questions = sum(u.count('?') for u in report_utterances)
    
    # Disagreement/challenge (positive sign of safety)
    challenge = len(re.findall(r'\b(but|however|although|actually|what about|have you considered)\b', report_text))
    
    # Calculate safety score (0-1)
    safety_indicators = vulnerability + idea_sharing + report_questions + challenge
    safety_score = min(safety_indicators / 15, 1.0)
    
    # Detailed breakdown
    return {
        'overall_score': safety_score,
        'vulnerability_count': vulnerability,
        'ideas_shared': idea_sharing,
        'questions_asked': report_questions,
        'challenges_made': challenge,
        'level': 'High' if safety_score >= 0.7 else 'Medium' if safety_score >= 0.4 else 'Low'
    }


def analyze_question_quality(sarah_utterances: List[str]) -> Dict[str, Any]:
    """Enhanced question quality analysis"""
    
    sarah_text = ' '.join(sarah_utterances)
    
    # Count all questions
    total_questions = sarah_text.count('?')
    
    if total_questions == 0:
        return {
            'total': 0,
            'open_ended': 0,
            'closed': 0,
            'future_focused': 0,
            'open_ended_pct': 0,
            'quality_score': 0
        }
    
    # Categorize questions
    open_ended = len(re.findall(r'\b(what|how|why|tell me|describe|explain|walk me through)\b[^?]*\?', sarah_text, re.IGNORECASE))
    closed = len(re.findall(r'\b(is|are|do|does|did|will|would|can|could|should|have|has|had)\b[^?]*\?', sarah_text, re.IGNORECASE))
    future_focused = len(re.findall(r'\b(what would|how could|what if|imagine|envision|picture)\b[^?]*\?', sarah_text, re.IGNORECASE))
    
    # Leading questions (negative indicator)
    leading = len(re.findall(r'\b(don\'t you think|wouldn\'t you say|isn\'t it true)\b[^?]*\?', sarah_text, re.IGNORECASE))
    
    # Quality score (0-1)
    quality_score = (
        (open_ended / max(total_questions, 1)) * 0.4 +
        (future_focused / max(total_questions, 1)) * 0.4 +
        (1 - (closed / max(total_questions, 1))) * 0.2 -
        (leading / max(total_questions, 1)) * 0.2
    )
    quality_score = max(min(quality_score, 1), 0)
    
    return {
        'total': total_questions,
        'open_ended': open_ended,
        'closed': closed,
        'future_focused': future_focused,
        'leading': leading,
        'open_ended_pct': (open_ended / total_questions * 100) if total_questions > 0 else 0,
        'quality_score': quality_score
    }


def calculate_interactivity_score(dialogue: List[Dict]) -> float:
    """Calculate conversation interactivity (0-10 scale)"""
    
    if len(dialogue) < 2:
        return 0
    
    # Conversation switches per 5-minute interval (assuming ~150 words/min)
    total_words = sum(d['word_count'] for d in dialogue)
    estimated_minutes = total_words / 150
    intervals = max(estimated_minutes / 5, 1)
    
    # Count speaker switches
    switches = sum(1 for i in range(len(dialogue)-1)
                   if dialogue[i]['speaker'] != dialogue[i+1]['speaker'])
    
    switches_per_interval = switches / intervals
    
    # Score: 5+ switches per interval is healthy
    score = min(switches_per_interval, 10)
    
    return score


def find_longest_monologue(dialogue: List[Dict]) -> Tuple[str, int, int]:
    """Find the longest uninterrupted speaking turn"""
    
    max_words = 0
    max_speaker = ""
    max_index = 0
    
    for i, d in enumerate(dialogue):
        if d['word_count'] > max_words:
            max_words = d['word_count']
            max_speaker = d['speaker']
            max_index = i
    
    # Convert words to approximate seconds (150 words/min = 2.5 words/sec)
    max_seconds = max_words / 2.5
    
    return max_speaker, max_words, max_seconds


def is_team_meeting(dialogue: List[Dict]) -> bool:
    """Determine if this is a team meeting (3+ participants) vs 1:1"""
    speakers = set(d['speaker'] for d in dialogue)
    return len(speakers) >= 3


def analyze_team_dynamics(dialogue: List[Dict]) -> TeamMetrics:
    """Analyze team meeting dynamics"""
    
    speaker_words = defaultdict(int)
    speaker_turns = defaultdict(int)
    
    for d in dialogue:
        speaker_words[d['speaker']] += d['word_count']
        speaker_turns[d['speaker']] += 1
    
    total_words = sum(speaker_words.values())
    participant_count = len(speaker_words)
    
    participation_dist = {
        speaker: (words / total_words * 100) if total_words > 0 else 0
        for speaker, words in speaker_words.items()
    }
    
    participation_values = list(participation_dist.values())
    dominance_score = np.std(participation_values) / 100 if participation_values else 0
    
    max_possible_std = np.sqrt(((100 - 100/participant_count)**2 * (participant_count - 1) +
                                (100/participant_count)**2) / participant_count)
    inclusion_score = 1 - (np.std(participation_values) / max_possible_std) if max_possible_std > 0 else 1.0
    
    speaker_interactions = defaultdict(int)
    for i in range(len(dialogue) - 1):
        from_speaker = dialogue[i]['speaker']
        to_speaker = dialogue[i+1]['speaker']
        if from_speaker != to_speaker:
            speaker_interactions[f"{from_speaker}→{to_speaker}"] += 1
    
    decision_words = ['decide', 'agreed', 'action item', 'will do', 'assigned', 'owner']
    decision_count = sum(
        1 for d in dialogue
        for word in decision_words
        if word in d['text'].lower()
    )
    decision_quality = min(decision_count / max(len(dialogue), 1), 1.0)
    
    all_text = ' '.join([d['text'] for d in dialogue])
    topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)
    topic_coverage = list(set(topics))[:10]
    
    return TeamMetrics(
        participant_count=participant_count,
        participation_distribution=participation_dist,
        turn_distribution=dict(speaker_turns),
        dominance_score=dominance_score,
        inclusion_score=inclusion_score,
        collaboration_patterns=dict(speaker_interactions),
        decision_quality=decision_quality,
        topic_coverage=topic_coverage
    )


@st.cache_data
def load_all_meetings(folder_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load both 1:1 meetings and team meetings from folder.
    Returns: (one_on_one_meetings, team_meetings)
    """
    
    one_on_one = []
    team_meetings = []
    
    try:
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.docx')])
    except FileNotFoundError:
        st.error(f"Folder not found: {folder_path}")
        return [], []
    
    if not files:
        st.warning(f"No .docx files found in {folder_path}")
        return [], []
    
    for file in files:
        try:
            # Read document
            doc = Document(os.path.join(folder_path, file))
            full_text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            
            if not full_text:
                continue
            
            # Parse dialogue - try different speaker patterns
            speaker_pattern_1on1 = r'(Sarah|Alex|Javier):\s'
            speaker_pattern_team = r'\[Speaker (\d+)\]'
            
            # Try 1:1 pattern first
            parts = re.split(speaker_pattern_1on1, full_text)
            
            dialogue = []
            speakers_found = set()
            
            if len(parts) > 2:  # Found 1:1 format
                for i in range(1, len(parts), 2):
                    if i+1 < len(parts):
                        speaker = parts[i]
                        content = parts[i+1].strip()
                        speakers_found.add(speaker)
                        
                        if content:
                            dialogue.append({
                                'speaker': speaker,
                                'text': content,
                                'word_count': len(content.split())
                            })
            
            # If 1:1 pattern didn't work, try team pattern
            if not dialogue:
                pattern = re.compile(r'\[Speaker (\d+)\][^\[]*')
                matches = pattern.findall(full_text)
                
                if matches:
                    splits = re.split(r'\[Speaker \d+\]', full_text)
                    
                    for i, match in enumerate(matches):
                        if i+1 < len(splits):
                            speaker_num = match
                            content = splits[i+1].strip()
                            content = re.sub(r'\(\d+:\d+\s*-\s*\d+:\d+\)', '', content).strip()
                            speakers_found.add(f"Speaker {speaker_num}")
                            
                            if content:
                                dialogue.append({
                                    'speaker': f"Speaker {speaker_num}",
                                    'text': content,
                                    'word_count': len(content.split())
                                })
            
            if not dialogue:
                continue
            
            # Determine meeting type
            is_team = len(speakers_found) >= 3
            
            # Parse date from filename
            date_obj = None
            
            # Try 1:1 format: FirstName_LastName_DDMmmYY.docx
            match = re.search(r'(\w+)_(\w+)_(\d+)([A-Za-z]+)(\d+)', file)
            if match:
                try:
                    day = int(match.group(3))
                    month_str = match.group(4)
                    year = 2000 + int(match.group(5)) if int(match.group(5)) < 100 else int(match.group(5))
                    
                    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                    month = month_map.get(month_str, 1)
                    date_obj = datetime(year, month, day)
                except:
                    pass
            
            # Try team format: Something_..._Mmm_DD__YYYY.docx
            if not date_obj:
                match = re.search(r'([A-Za-z]+)_(\d+)__(\d+)', file)
                if match:
                    try:
                        month_str = match.group(1)
                        day = int(match.group(2))
                        year = int(match.group(3))
                        
                        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                        month = month_map.get(month_str, 1)
                        date_obj = datetime(year, month, day)
                    except:
                        pass
            
            if not date_obj:
                date_obj = datetime.now()  # Fallback
            
            # Basic metrics
            total_words = sum(d['word_count'] for d in dialogue)
            
            # Build meeting object
            meeting = {
                'file': file,
                'date': date_obj,
                'date_str': date_obj.strftime('%b %d, %Y'),
                'dialogue': dialogue,
                'total_words': total_words,
                'turn_count': len(dialogue),
                'is_team': is_team,
                'participants': list(speakers_found)
            }
            
            if is_team:
                # Team meeting specific analysis
                team_metrics = analyze_team_dynamics(dialogue)
                meeting['team_metrics'] = team_metrics
                
                # Overall sentiment
                all_text = ' '.join([d['text'] for d in dialogue])
                meeting['overall_sentiment'] = transformer_sentiment_scalar(all_text)
                
                team_meetings.append(meeting)
            else:
                # 1:1 meeting - FULL ANALYSIS
                sarah_utterances = []
                report_utterances = []
                
                for d in dialogue:
                    if d['speaker'] == 'Sarah':
                        sarah_utterances.append(d['text'])
                    else:
                        report_utterances.append(d['text'])
                
                # Parse person name from filename or dialogue
                if 'Sarah' in speakers_found:
                    person_name = list(speakers_found - {'Sarah'})[0]
                else:
                    # Try to parse from filename
                    match = re.match(r'(\w+)_(\w+)_', file)
                    if match:
                        person_name = f"{match.group(1)} {match.group(2)}"
                    else:
                        person_name = 'Unknown'
                
                sarah_words = sum(d['word_count'] for d in dialogue if d['speaker'] == 'Sarah')
                report_words = total_words - sarah_words
                
                talk_ratio = (sarah_words / total_words * 100) if total_words > 0 else 0
                avg_turn_length = total_words / len(dialogue) if dialogue else 0
                
                # Sentiment analysis
                report_text = ' '.join(report_utterances)
                sarah_text = ' '.join(sarah_utterances)
                
                report_sentiment = transformer_sentiment_scalar(report_text)
                sarah_sentiment = transformer_sentiment_scalar(sarah_text)
                
                # Advanced analyses
                arc, emotions = AdvancedNLPAnalyzer.analyze_sentiment_arc(dialogue)
                linguistic_diversity = AdvancedNLPAnalyzer.calculate_linguistic_diversity(report_text)
                power_dynamics = AdvancedNLPAnalyzer.detect_power_dynamics(sarah_text, report_text)
                action_items = AdvancedNLPAnalyzer.extract_action_items(report_text)
                
                # Framework analyses
                chat_scores = analyze_chat_framework(dialogue, sarah_utterances, report_utterances)
                act_predictions = predict_act_outcomes(dialogue, report_utterances, report_sentiment)
                psych_safety = analyze_psychological_safety(report_utterances, dialogue)
                question_analysis = analyze_question_quality(sarah_utterances)
                
                # Conversation quality metrics
                interactivity = calculate_interactivity_score(dialogue)
                longest_speaker, longest_words, longest_seconds = find_longest_monologue(dialogue)
                
                # Question metrics
                questions_total = sarah_text.count('?') + report_text.count('?')
                sarah_questions = sarah_text.count('?')
                question_density = (questions_total / total_words * 100) if total_words > 0 else 0
                
                meeting.update({
                    'person': person_name,
                    'sarah_words': sarah_words,
                    'report_words': report_words,
                    'talk_ratio': talk_ratio,
                    'listen_ratio': 100 - talk_ratio,
                    'avg_turn_length': avg_turn_length,
                    'report_sentiment': report_sentiment,
                    'sarah_sentiment': sarah_sentiment,
                    'sentiment_arc': arc,
                    'emotion_arc': emotions,
                    'questions_total': questions_total,
                    'sarah_questions': sarah_questions,
                    'question_density': question_density,
                    'chat_scores': chat_scores,
                    'act_predictions': act_predictions,
                    'psych_safety': psych_safety,
                    'question_analysis': question_analysis,
                    'interactivity': interactivity,
                    'longest_monologue': {
                        'speaker': longest_speaker,
                        'words': longest_words,
                        'seconds': longest_seconds
                    },
                    'linguistic_diversity': linguistic_diversity,
                    'power_dynamics': power_dynamics,
                    'action_items': action_items,
                    'sarah_utterances': sarah_utterances,
                    'report_utterances': report_utterances
                })
                
                one_on_one.append(meeting)
            
        except Exception as e:
            st.warning(f"Error processing {file}: {str(e)}")
            continue
    
    return sorted(one_on_one, key=lambda x: x['date']), sorted(team_meetings, key=lambda x: x['date'])


def create_radar_chart(scores: Dict[str, float], title: str, color: str = '#667eea') -> go.Figure:
    """Create an enhanced radar chart for multi-dimensional scores"""
    
    categories = list(scores.keys())
    values = [scores[k] * 100 for k in categories]
    
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor=f'rgba(102, 126, 234, 0.3)',
        line=dict(color=color, width=3),
        name=title,
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%'],
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray'
            )
        ),
        showlegend=False,
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16, weight='bold')),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_timeline_chart(meetings: List[Dict], metric_name: str, person_filter: Optional[str] = None) -> go.Figure:
    """Create enhanced timeline visualization"""
    
    filtered = meetings if not person_filter else [m for m in meetings if m['person'] == person_filter]
    
    if not filtered:
        return go.Figure()
    
    dates = [m['date'] for m in filtered]
    
    # Extract metric values
    if metric_name == 'CHAT Overall':
        values = [m['chat_scores']['overall'] * 100 for m in filtered]
        yaxis_title = 'CHAT Score (%)'
        color = '#667eea'
    elif metric_name == 'ACT Overall':
        values = [m['act_predictions']['overall'] * 100 for m in filtered]
        yaxis_title = 'ACT Prediction (%)'
        color = '#764ba2'
    elif metric_name == 'Talk Ratio':
        values = [m['talk_ratio'] for m in filtered]
        yaxis_title = 'Sarah Talk Ratio (%)'
        color = '#f093fb'
    elif metric_name == 'Psychological Safety':
        values = [m['psych_safety']['overall_score'] * 100 for m in filtered]
        yaxis_title = 'Safety Score (%)'
        color = '#28a745'
    elif metric_name == 'Engagement':
        values = [m['act_predictions']['engagement'] * 100 for m in filtered]
        yaxis_title = 'Engagement Prediction (%)'
        color = '#17a2b8'
    else:
        values = [0] * len(filtered)
        yaxis_title = 'Value'
        color = '#667eea'
    
    fig = go.Figure()
    
    # Line trace
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name=metric_name,
        line=dict(color=color, width=3),
        marker=dict(size=10, color=color, line=dict(width=2, color='white')),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>' + yaxis_title + ': %{y:.1f}<extra></extra>'
    ))
    
    # Add trend line
    if len(dates) > 1:
        z = np.polyfit(range(len(dates)), values, 1)
        p = np.poly1d(z)
        trend_values = [p(i) for i in range(len(dates))]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=trend_values,
            mode='lines',
            name='Trend',
            line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dash'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text=f'{metric_name} Over Time', x=0.5, xanchor='center'),
        xaxis_title='Date',
        yaxis_title=yaxis_title,
        hovermode='x unified',
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_comparison_bar(alex_meetings: List[Dict], javier_meetings: List[Dict],
                          metric_name: str) -> go.Figure:
    """Create comparative bar chart"""
    
    # Calculate averages
    if metric_name == 'CHAT Score':
        alex_val = sum(m['chat_scores']['overall'] for m in alex_meetings) / len(alex_meetings) * 100
        javier_val = sum(m['chat_scores']['overall'] for m in javier_meetings) / len(javier_meetings) * 100
        title = 'Average CHAT Score'
    elif metric_name == 'ACT Score':
        alex_val = sum(m['act_predictions']['overall'] for m in alex_meetings) / len(alex_meetings) * 100
        javier_val = sum(m['act_predictions']['overall'] for m in javier_meetings) / len(javier_meetings) * 100
        title = 'Average ACT Prediction'
    elif metric_name == 'Talk Ratio':
        alex_val = sum(m['talk_ratio'] for m in alex_meetings) / len(alex_meetings)
        javier_val = sum(m['talk_ratio'] for m in javier_meetings) / len(javier_meetings)
        title = 'Average Talk Ratio (Sarah)'
    elif metric_name == 'Question Quality':
        alex_val = sum(m['question_analysis']['quality_score'] for m in alex_meetings) / len(alex_meetings) * 100
        javier_val = sum(m['question_analysis']['quality_score'] for m in javier_meetings) / len(javier_meetings) * 100
        title = 'Average Question Quality'
    else:
        alex_val = javier_val = 0
        title = metric_name
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Alex Rodriguez', 'Javier Morales'],
        y=[alex_val, javier_val],
        marker=dict(
            color=['#667eea', '#764ba2'],
            line=dict(color='white', width=2)
        ),
        text=[f'{alex_val:.1f}%', f'{javier_val:.1f}%'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>'
    ))
    
    # Add benchmark line if applicable
    if 'Talk Ratio' in title:
        fig.add_hline(y=50, line_dash="dash", line_color="gray",
                     annotation_text="Ideal: 50%", annotation_position="right")
        fig.add_hrect(y0=40, y1=60, fillcolor="green", opacity=0.1, line_width=0)
    elif 'Score' in title or 'Quality' in title:
        fig.add_hline(y=70, line_dash="dash", line_color="gray",
                     annotation_text="Target: 70%", annotation_position="right")
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        yaxis_title='Percentage (%)',
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_sentiment_arc_chart(meeting: Dict) -> go.Figure:
    """Visualize sentiment progression through conversation"""
    
    arc = meeting['sentiment_arc']
    segments = ['Beginning', 'Early', 'Middle', 'Late', 'End'][:len(arc)]
    
    # Color based on sentiment
    colors = ['#dc3545' if s < 0 else '#ffc107' if s < 0.2 else '#28a745' for s in arc]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=segments,
        y=arc,
        marker=dict(color=colors, line=dict(color='white', width=2)),
        hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    
    fig.update_layout(
        title='Sentiment Arc Throughout Conversation',
        yaxis_title='Sentiment Polarity',
        yaxis_range=[-1, 1],
        showlegend=False,
        height=300,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def get_mastery_level_badge(level: int) -> str:
    """Generate HTML badge for mastery level"""
    
    levels = {
        1: ('Beginning', 'danger', '🌱'),
        2: ('Emerging', 'warning', '🌿'),
        3: ('Experienced', 'info', '🌳'),
        4: ('Master', 'success', '🏆')
    }
    
    name, badge_type, emoji = levels.get(level, ('Unknown', 'secondary', '❓'))
    
    return f'<span class="badge badge-{badge_type}">{emoji} Level {level}: {name}</span>'


def create_participation_pie_chart(team_meeting: Dict) -> go.Figure:
    """Create pie chart showing participation distribution"""
    
    metrics = team_meeting['team_metrics']
    
    speakers = list(metrics.participation_distribution.keys())
    percentages = list(metrics.participation_distribution.values())
    
    colors = px.colors.qualitative.Set3[:len(speakers)]
    
    fig = go.Figure(data=[go.Pie(
        labels=speakers,
        values=percentages,
        hole=0.3,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>%{value:.1f}% of words<extra></extra>'
    )])
    
    fig.update_layout(
        title='Participation Distribution',
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_interaction_network(team_meeting: Dict) -> go.Figure:
    """Create network diagram showing speaker interactions"""
    
    metrics = team_meeting['team_metrics']
    interactions = metrics.collaboration_patterns
    
    if not interactions:
        return go.Figure()
    
    top_interactions = sorted(interactions.items(), key=lambda x: x[1], reverse=True)[:10]
    
    labels = [interaction.replace('→', ' → ') for interaction, _ in top_interactions]
    values = [count for _, count in top_interactions]
    
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color='#2196f3')
    )])
    
    fig.update_layout(
        title='Top Speaker Interactions',
        xaxis_title='Number of Exchanges',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_participation_balance_gauge(inclusion_score: float) -> go.Figure:
    """Create gauge showing participation balance"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=inclusion_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Inclusion Score", 'font': {'size': 20}},
        delta={'reference': 70, 'increasing': {'color': "#28a745"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#2196f3"},
            'steps': [
                {'range': [0, 40], 'color': '#ffcdd2'},
                {'range': [40, 70], 'color': '#fff9c4'},
                {'range': [70, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def get_chat_recommendation(dimension: str) -> str:
    """Get specific tactical recommendation for CHAT dimension"""
    
    recommendations = {
        'connect': 'Start every 1:1 with a genuine personal check-in. Ask "How are you really doing?" and pause 3-5 seconds for authentic answers. Explicitly state the meeting purpose. Build a confidential, judgment-free zone.',
        
        'hear': 'Practice active listening: Paraphrase what you hear ("So what I\'m hearing is..."), ask clarifying questions ("Tell me more about that"), and pause 1-2 seconds before responding. Avoid interrupting or problem-solving too quickly.',
        
        'ask': 'Transform closed questions into open-ended ones. Replace "Did you..." with "What..." and "How..." questions. Ask future-focused questions like "What would success look like?" or "How could we approach this differently?" Avoid leading questions.',
        
        'transform': 'Always end with clear, specific action items. Ask "What will you do? By when? What support do you need?" Celebrate progress from previous meetings. Empower their decisions rather than prescribing solutions. Follow up consistently.'
    }
    
    return recommendations.get(dimension, 'Focus on building this dimension through deliberate practice and reflection.')


def show_overview(meetings, alex_meetings, javier_meetings):
    """Display overview dashboard"""
    
    st.markdown("## 🎯 Overall Performance")
    
    # Overall mastery level
    avg_mastery = sum(m['chat_scores']['mastery_level'] for m in meetings) / len(meetings)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### CHAT Framework Mastery")
        st.markdown(f"**Current Level:** {get_mastery_level_badge(round(avg_mastery))}", unsafe_allow_html=True)
        
        # Progress to next level
        if avg_mastery < 4:
            progress = (avg_mastery % 1) * 100
            st.markdown(f"""
            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {progress}%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    {progress:.0f}% to Level {int(avg_mastery) + 1}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("🏆 Master level achieved!")
    
    with col2:
        latest_meeting = meetings[-1]
        st.markdown(f"""
        <div class="insight-box">
        <strong>Last Meeting:</strong> {latest_meeting['date_str']}<br>
        <strong>With:</strong> {latest_meeting['person']}<br>
        <strong>Total Meetings:</strong> {len(meetings)}<br>
        <strong>Time Span:</strong> {(meetings[-1]['date'] - meetings[0]['date']).days} days
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CHAT dimensions breakdown
    st.markdown("### 🎨 CHAT Dimensions Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average CHAT scores
        avg_chat_dims = {
            'Connect': sum(m['chat_scores']['connect'] for m in meetings) / len(meetings),
            'Hear': sum(m['chat_scores']['hear'] for m in meetings) / len(meetings),
            'Ask': sum(m['chat_scores']['ask'] for m in meetings) / len(meetings),
            'Transform': sum(m['chat_scores']['transform'] for m in meetings) / len(meetings)
        }
        
        fig = create_radar_chart(avg_chat_dims, "Average CHAT Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ACT predictions
        avg_act_dims = {
            'Engagement': sum(m['act_predictions']['engagement'] for m in meetings) / len(meetings),
            'Readiness': sum(m['act_predictions']['readiness'] for m in meetings) / len(meetings),
            'Ownership': sum(m['act_predictions']['ownership'] for m in meetings) / len(meetings)
        }
        
        fig = create_radar_chart(avg_act_dims, "Average ACT Predictions", color='#764ba2')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Quick insights
    st.markdown("### 🔍 Quick Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Strongest dimension
        strongest_dim = max(avg_chat_dims, key=avg_chat_dims.get)
        st.markdown(f"""
        <div class="success-box">
        <strong>💪 Strongest Dimension</strong><br>
        <span style="font-size: 1.5rem;">{strongest_dim}</span><br>
        Score: {avg_chat_dims[strongest_dim] * 100:.0f}%
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Growth opportunity
        weakest_dim = min(avg_chat_dims, key=avg_chat_dims.get)
        st.markdown(f"""
        <div class="warning-box">
        <strong>📈 Growth Opportunity</strong><br>
        <span style="font-size: 1.5rem;">{weakest_dim}</span><br>
        Score: {avg_chat_dims[weakest_dim] * 100:.0f}%
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Most improved
        if len(meetings) >= 2:
            first_half = meetings[:len(meetings)//2]
            second_half = meetings[len(meetings)//2:]
            
            improvements = {}
            for dim in ['connect', 'hear', 'ask', 'transform']:
                first_avg = sum(m['chat_scores'][dim] for m in first_half) / len(first_half)
                second_avg = sum(m['chat_scores'][dim] for m in second_half) / len(second_half)
                improvements[dim.title()] = second_avg - first_avg
            
            most_improved = max(improvements, key=improvements.get)
            improvement_val = improvements[most_improved]
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>🚀 Most Improved</strong><br>
            <span style="font-size: 1.5rem;">{most_improved}</span><br>
            +{improvement_val * 100:.0f}% improvement
            </div>
            """, unsafe_allow_html=True)


def show_meeting_analysis(meetings, show_detailed, show_actions):
    """Display detailed meeting analysis"""
    
    st.markdown("## 💬 Individual Meeting Deep Dive")
    
    # Meeting selector
    meeting_options = [f"{m['date_str']} - {m['person']}" for m in meetings]
    selected_idx = st.selectbox("Select Meeting", range(len(meetings)),
                                format_func=lambda i: meeting_options[i])
    
    meeting = meetings[selected_idx]
    
    # Meeting header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### {meeting['person']} - {meeting['date_str']}")
        mastery_badge = get_mastery_level_badge(meeting['chat_scores']['mastery_level'])
        st.markdown(mastery_badge, unsafe_allow_html=True)
    
    with col2:
        st.metric("CHAT Score", f"{meeting['chat_scores']['overall'] * 100:.0f}%")
    
    with col3:
        st.metric("ACT Score", f"{meeting['act_predictions']['overall'] * 100:.0f}%")
    
    st.markdown("---")
    
    # Core metrics
    st.markdown("### 📊 Conversation Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Words", f"{meeting['total_words']:,}")
    
    with col2:
        st.metric("Talk Ratio", f"{meeting['talk_ratio']:.0f}% / {meeting['listen_ratio']:.0f}%")
    
    with col3:
        st.metric("Turn Count", meeting['turn_count'])
    
    with col4:
        st.metric("Interactivity", f"{meeting['interactivity']:.1f}/10")
    
    with col5:
        st.metric("Questions", meeting['questions_total'])
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # CHAT radar
        chat_dims = {
            'Connect': meeting['chat_scores']['connect'],
            'Hear': meeting['chat_scores']['hear'],
            'Ask': meeting['chat_scores']['ask'],
            'Transform': meeting['chat_scores']['transform']
        }
        fig = create_radar_chart(chat_dims, "CHAT Framework")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ACT radar
        act_dims = {
            'Engagement': meeting['act_predictions']['engagement'],
            'Readiness': meeting['act_predictions']['readiness'],
            'Ownership': meeting['act_predictions']['ownership']
        }
        fig = create_radar_chart(act_dims, "ACT Predictions", color='#764ba2')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment arc
    if meeting['sentiment_arc']:
        fig = create_sentiment_arc_chart(meeting)
        st.plotly_chart(fig, use_container_width=True)
    
    if show_detailed:
        st.markdown("---")
        st.markdown("### 🔬 Advanced Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Psychological safety
            safety = meeting['psych_safety']
            safety_color = '#28a745' if safety['level'] == 'High' else '#ffc107' if safety['level'] == 'Medium' else '#dc3545'
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>🛡️ Psychological Safety: <span style="color: {safety_color};">{safety['level']}</span></strong><br>
            <br>
            • Vulnerabilities shared: {safety['vulnerability_count']}<br>
            • Ideas proposed: {safety['ideas_shared']}<br>
            • Questions asked: {safety['questions_asked']}<br>
            • Challenges made: {safety['challenges_made']}<br>
            <br>
            <strong>Overall Score:</strong> {safety['overall_score'] * 100:.0f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Question quality
            qa = meeting['question_analysis']
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>❓ Question Quality Analysis</strong><br>
            <br>
            • Total questions: {qa['total']}<br>
            • Open-ended: {qa['open_ended']} ({qa['open_ended_pct']:.0f}%)<br>
            • Closed: {qa['closed']}<br>
            • Future-focused: {qa['future_focused']}<br>
            • Leading: {qa['leading']}<br>
            <br>
            <strong>Quality Score:</strong> {qa['quality_score'] * 100:.0f}%
            </div>
            """, unsafe_allow_html=True)
        
        # Power dynamics
        col1, col2 = st.columns(2)
        
        with col1:
            pd = meeting['power_dynamics']
            st.markdown(f"""
            <div class="insight-box">
            <strong>⚖️ Power Dynamics</strong><br>
            <br>
            <strong>Sarah:</strong><br>
            • Hedging: {pd['manager_hedges']}<br>
            • Certainty: {pd['manager_certainty']}<br>
            <br>
            <strong>{meeting['person'].split()[0]}:</strong><br>
            • Hedging: {pd['report_hedges']}<br>
            • Certainty: {pd['report_certainty']}<br>
            <br>
            <strong>Balance:</strong> {pd['power_balance'].title()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Linguistic diversity
            ld = meeting['linguistic_diversity']
            st.markdown(f"""
            <div class="insight-box">
            <strong>📚 Linguistic Diversity</strong><br>
            <br>
            • Unique words: {ld['unique_words']:,}<br>
            • Total words: {ld['total_words']:,}<br>
            • Type-Token Ratio: {ld['type_token_ratio']:.2f}<br>
            • Avg word length: {ld['avg_word_length']:.1f} chars<br>
            <br>
            <em>Higher TTR indicates more diverse vocabulary</em>
            </div>
            """, unsafe_allow_html=True)
    
    if show_actions and meeting['action_items']:
        st.markdown("---")
        st.markdown("### ✅ Extracted Action Items")
        
        for i, action in enumerate(meeting['action_items'], 1):
            st.markdown(f"{i}. {action.title()}")
    
    # Longest monologue warning
    mono = meeting['longest_monologue']
    if mono['seconds'] > 150:  # 2.5 minutes
        st.markdown(f"""
        <div class="warning-box">
        ⚠️ <strong>Long Monologue Detected:</strong> {mono['speaker']} spoke for {mono['seconds']:.0f} seconds 
        ({mono['words']} words). Consider breaking this up with questions for better engagement.
        </div>
        """, unsafe_allow_html=True)


def show_trends(meetings, alex_meetings, javier_meetings):
    """Display trend analysis"""
    
    st.markdown("## 📈 Trends & Patterns Over Time")
    
    # Metric selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        metric = st.selectbox(
            "Select Metric to Track",
            ['CHAT Overall', 'ACT Overall', 'Talk Ratio', 'Psychological Safety', 'Engagement']
        )
    
    with col2:
        person_filter = st.selectbox(
            "Filter by Person",
            ['All', 'Alex Rodriguez', 'Javier Morales']
        )
    
    # Create timeline chart
    filter_val = None if person_filter == 'All' else person_filter
    fig = create_timeline_chart(meetings, metric, filter_val)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # CHAT dimension trends
    st.markdown("### 🎨 CHAT Dimensions Over Time")
    
    dates = [m['date'] for m in meetings]
    
    fig = go.Figure()
    
    colors = {'connect': '#667eea', 'hear': '#764ba2', 'ask': '#f093fb', 'transform': '#4facfe'}
    
    for dim, color in colors.items():
        values = [m['chat_scores'][dim] * 100 for m in meetings]
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=dim.title(),
            line=dict(color=color, width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='CHAT Dimensions Progression',
        xaxis_title='Date',
        yaxis_title='Score (%)',
        hovermode='x unified',
        height=450,
        showlegend=True,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ACT predictions trends
    st.markdown("### 🔮 ACT Predictions Over Time")
    
    fig = go.Figure()
    
    act_colors = {'engagement': '#28a745', 'readiness': '#17a2b8', 'ownership': '#ffc107'}
    
    for dim, color in act_colors.items():
        values = [m['act_predictions'][dim] * 100 for m in meetings]
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=dim.title(),
            line=dict(color=color, width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='ACT Predictions Progression',
        xaxis_title='Date',
        yaxis_title='Prediction (%)',
        hovermode='x unified',
        height=450,
        showlegend=True,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chat_scores = [m['chat_scores']['overall'] * 100 for m in meetings]
        st.markdown(f"""
        <div class="insight-box">
        <strong>CHAT Score Statistics</strong><br>
        <br>
        • Mean: {np.mean(chat_scores):.1f}%<br>
        • Median: {np.median(chat_scores):.1f}%<br>
        • Std Dev: {np.std(chat_scores):.1f}%<br>
        • Min: {np.min(chat_scores):.1f}%<br>
        • Max: {np.max(chat_scores):.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        act_scores = [m['act_predictions']['overall'] * 100 for m in meetings]
        st.markdown(f"""
        <div class="insight-box">
        <strong>ACT Score Statistics</strong><br>
        <br>
        • Mean: {np.mean(act_scores):.1f}%<br>
        • Median: {np.median(act_scores):.1f}%<br>
        • Std Dev: {np.std(act_scores):.1f}%<br>
        • Min: {np.min(act_scores):.1f}%<br>
        • Max: {np.max(act_scores):.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        talk_ratios = [m['talk_ratio'] for m in meetings]
        st.markdown(f"""
        <div class="insight-box">
        <strong>Talk Ratio Statistics</strong><br>
        <br>
        • Mean: {np.mean(talk_ratios):.1f}%<br>
        • Median: {np.median(talk_ratios):.1f}%<br>
        • Std Dev: {np.std(talk_ratios):.1f}%<br>
        • Min: {np.min(talk_ratios):.1f}%<br>
        • Max: {np.max(talk_ratios):.1f}%
        </div>
        """, unsafe_allow_html=True)


def show_comparison(alex_meetings, javier_meetings):
    """Display person-to-person comparison"""
    
    st.markdown("## 👥 Alex vs. Javier: Communication Patterns")
    
    if not alex_meetings or not javier_meetings:
        st.warning("Need meetings from both Alex and Javier for comparison")
        return
    
    # Comparison metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Alex Rodriguez")
        alex_latest = alex_meetings[-1]
        
        st.metric("Latest CHAT Score", f"{alex_latest['chat_scores']['overall'] * 100:.0f}%")
        st.metric("Latest ACT Score", f"{alex_latest['act_predictions']['overall'] * 100:.0f}%")
        st.metric("Avg Talk Ratio", f"{sum(m['talk_ratio'] for m in alex_meetings) / len(alex_meetings):.0f}%")
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>Communication Style:</strong><br>
        • Total meetings: {len(alex_meetings)}<br>
        • Avg words per meeting: {sum(m['total_words'] for m in alex_meetings) / len(alex_meetings):,.0f}<br>
        • Avg sentiment: {sum(m['report_sentiment'] for m in alex_meetings) / len(alex_meetings):.2f}<br>
        • Psychological safety: {sum(m['psych_safety']['overall_score'] for m in alex_meetings) / len(alex_meetings) * 100:.0f}%
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Javier Morales")
        javier_latest = javier_meetings[-1]
        
        st.metric("Latest CHAT Score", f"{javier_latest['chat_scores']['overall'] * 100:.0f}%")
        st.metric("Latest ACT Score", f"{javier_latest['act_predictions']['overall'] * 100:.0f}%")
        st.metric("Avg Talk Ratio", f"{sum(m['talk_ratio'] for m in javier_meetings) / len(javier_meetings):.0f}%")
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>Communication Style:</strong><br>
        • Total meetings: {len(javier_meetings)}<br>
        • Avg words per meeting: {sum(m['total_words'] for m in javier_meetings) / len(javier_meetings):,.0f}<br>
        • Avg sentiment: {sum(m['report_sentiment'] for m in javier_meetings) / len(javier_meetings):.2f}<br>
        • Psychological safety: {sum(m['psych_safety']['overall_score'] for m in javier_meetings) / len(javier_meetings) * 100:.0f}%
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison charts
    st.markdown("### 📊 Comparative Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_comparison_bar(alex_meetings, javier_meetings, 'CHAT Score')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_comparison_bar(alex_meetings, javier_meetings, 'ACT Score')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_comparison_bar(alex_meetings, javier_meetings, 'Talk Ratio')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_comparison_bar(alex_meetings, javier_meetings, 'Question Quality')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Communication insights
    st.markdown("### 💡 Communication Style Differences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alex_avg_talk = sum(m['talk_ratio'] for m in alex_meetings) / len(alex_meetings)
        alex_avg_questions = sum(m['question_analysis']['open_ended_pct'] for m in alex_meetings) / len(alex_meetings)
        alex_avg_interactivity = sum(m['interactivity'] for m in alex_meetings) / len(alex_meetings)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🎯 Alex's Pattern</strong><br>
        <br>
        <strong>Talk Balance:</strong> {alex_avg_talk:.0f}% Sarah<br>
        <strong>Open Questions:</strong> {alex_avg_questions:.0f}%<br>
        <strong>Interactivity:</strong> {alex_avg_interactivity:.1f}/10<br>
        <br>
        <strong>Best Approach:</strong><br>
        • Strategic discussions<br>
        • Big-picture thinking<br>
        • Problem-solving focus<br>
        • Career development emphasis
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        javier_avg_talk = sum(m['talk_ratio'] for m in javier_meetings) / len(javier_meetings)
        javier_avg_questions = sum(m['question_analysis']['open_ended_pct'] for m in javier_meetings) / len(javier_meetings)
        javier_avg_interactivity = sum(m['interactivity'] for m in javier_meetings) / len(javier_meetings)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🎯 Javier's Pattern</strong><br>
        <br>
        <strong>Talk Balance:</strong> {javier_avg_talk:.0f}% Sarah<br>
        <strong>Open Questions:</strong> {javier_avg_questions:.0f}%<br>
        <strong>Interactivity:</strong> {javier_avg_interactivity:.1f}/10<br>
        <br>
        <strong>Best Approach:</strong><br>
        • Technical deep-dives<br>
        • Process discussions<br>
        • Quality metrics focus<br>
        • Explicit growth planning
        </div>
        """, unsafe_allow_html=True)


def show_coaching_insights(meetings, alex_meetings, javier_meetings):
    """Display AI-powered coaching recommendations"""
    
    st.markdown("## 💡 AI-Powered Coaching Insights")
    st.markdown("**Personalized recommendations for Sarah's leadership development**")
    
    st.markdown("---")
    
    # Overall assessment
    st.markdown("### 🎯 Overall Assessment")
    
    avg_chat = sum(m['chat_scores']['overall'] for m in meetings) / len(meetings)
    avg_act = sum(m['act_predictions']['overall'] for m in meetings) / len(meetings)
    
    if avg_chat >= 0.75 and avg_act >= 0.65:
        st.markdown("""
        <div class="success-box">
        🎉 <strong>Excellent Performance</strong><br>
        Sarah demonstrates strong conversational leadership across both direct reports. 
        CHAT and ACT scores consistently exceed targets, indicating effective coaching and 
        high predicted employee outcomes.
        </div>
        """, unsafe_allow_html=True)
    elif avg_chat >= 0.55 and avg_act >= 0.50:
        st.markdown("""
        <div class="warning-box">
        👍 <strong>Good Progress</strong><br>
        Sarah is developing her coaching skills effectively with clear areas for continued growth. 
        Focus on consistency and strengthening specific CHAT dimensions.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="danger-box">
        ⚠️ <strong>Development Needed</strong><br>
        Sarah should focus on foundational conversation skills. Targeted coaching and practice 
        in specific CHAT dimensions will drive improvement.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strengths
    st.markdown("### 🌟 Key Strengths")
    
    # Find strongest dimensions
    all_chat_scores = defaultdict(list)
    for m in meetings:
        for dim in ['connect', 'hear', 'ask', 'transform']:
            all_chat_scores[dim].append(m['chat_scores'][dim])
    
    avg_by_dim = {dim: sum(scores)/len(scores) for dim, scores in all_chat_scores.items()}
    strongest = max(avg_by_dim, key=avg_by_dim.get)
    second_strongest = sorted(avg_by_dim.items(), key=lambda x: x[1], reverse=True)[1][0]
    
    st.markdown(f"""
    <div class="success-box">
    ✨ <strong>Top CHAT Dimensions:</strong><br>
    <br>
    1. <strong>{strongest.title()}</strong>: {avg_by_dim[strongest]*100:.0f}% - Excellent foundation<br>
    2. <strong>{second_strongest.title()}</strong>: {avg_by_dim[second_strongest]*100:.0f}% - Strong performance<br>
    <br>
    • <strong>Consistent Cadence:</strong> {len(meetings)} meetings completed<br>
    • <strong>Positive Sentiment:</strong> Both reports show {sum(m['report_sentiment'] for m in meetings)/len(meetings):.2f} avg sentiment<br>
    • <strong>Active Engagement:</strong> {sum(m['questions_total'] for m in meetings)} total questions asked<br>
    • <strong>Psychological Safety:</strong> {sum(m['psych_safety']['overall_score'] for m in meetings)/len(meetings)*100:.0f}% average
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Growth opportunities
    st.markdown("### 📈 Priority Growth Opportunities")
    
    weakest = min(avg_by_dim, key=avg_by_dim.get)
    alex_latest = alex_meetings[-1] if alex_meetings else None
    javier_latest = javier_meetings[-1] if javier_meetings else None
    
    recommendations = []
    
    # CHAT dimension recommendations
    if avg_by_dim[weakest] < 0.6:
        recommendations.append({
            'priority': 'High',
            'area': f'{weakest.title()} Dimension',
            'issue': f'Currently at {avg_by_dim[weakest]*100:.0f}%, below target of 60%',
            'action': get_chat_recommendation(weakest),
            'impact': 'Foundational for coaching effectiveness'
        })
    
    # Talk ratio balance
    if alex_latest:
        alex_talk = alex_latest['talk_ratio']
        if alex_talk > 70:
            recommendations.append({
                'priority': 'Medium',
                'area': 'Talk-Listen Balance (Alex)',
                'issue': f'Sarah talking {alex_talk:.0f}% of the time (target: 40-60%)',
                'action': 'Ask more open-ended questions and pause 1-2 seconds after Alex responds. Let him drive more of the conversation.',
                'impact': 'Improves Alex\'s engagement and ownership'
            })
        elif alex_talk < 30:
            recommendations.append({
                'priority': 'Medium',
                'area': 'Talk-Listen Balance (Alex)',
                'issue': f'Sarah talking only {alex_talk:.0f}% (may be too passive)',
                'action': 'Provide more strategic guidance and share perspective. Balance listening with active coaching.',
                'impact': 'Ensures Alex gets needed direction'
            })
    
    if javier_latest:
        javier_talk = javier_latest['talk_ratio']
        if javier_talk > 70:
            recommendations.append({
                'priority': 'Medium',
                'area': 'Talk-Listen Balance (Javier)',
                'issue': f'Sarah talking {javier_talk:.0f}% of the time (target: 40-60%)',
                'action': 'Create more space for Javier to share technical expertise. Ask "What do you think?" and "Tell me more."',
                'impact': 'Builds Javier\'s confidence and voice'
            })
    
    # Question quality
    avg_question_quality = sum(m['question_analysis']['quality_score'] for m in meetings) / len(meetings)
    if avg_question_quality < 0.7:
        recommendations.append({
            'priority': 'High',
            'area': 'Question Quality',
            'issue': f'Quality score at {avg_question_quality*100:.0f}% (target: 70%+)',
            'action': 'Shift from closed (Yes/No) to open-ended questions. Use "What...", "How...", and "Tell me..." starters. Practice future-focused questions like "What would success look like?"',
            'impact': 'Drives deeper thinking and ownership'
        })
    
    # ACT predictions
    if alex_latest and alex_latest['act_predictions']['engagement'] < 0.6:
        recommendations.append({
            'priority': 'High',
            'area': 'Alex Engagement',
            'issue': f'Predicted engagement at {alex_latest["act_predictions"]["engagement"]*100:.0f}%',
            'action': 'Schedule more frequent check-ins. Ask "How are you really doing?" and allow silence. Explore career aspirations and provide growth opportunities.',
            'impact': 'Critical for retention and performance'
        })
    
    if javier_latest and javier_latest['act_predictions']['ownership'] < 0.6:
        recommendations.append({
            'priority': 'Medium',
            'area': 'Javier Ownership',
            'issue': f'Predicted ownership at {javier_latest["act_predictions"]["ownership"]*100:.0f}%',
            'action': 'Connect QA work to business outcomes. Ask "How does this impact customers?" Encourage solution proposals before offering answers.',
            'impact': 'Increases initiative and accountability'
        })
    
    # Interactivity score
    avg_interactivity = sum(m['interactivity'] for m in meetings) / len(meetings)
    if avg_interactivity < 5:
        recommendations.append({
            'priority': 'Low',
            'area': 'Conversation Flow',
            'issue': f'Interactivity score at {avg_interactivity:.1f}/10 (target: 5+)',
            'action': 'Break up long monologues. Check in every 1-2 minutes with questions like "What are you thinking?" or "Does this resonate?"',
            'impact': 'Creates more dynamic dialogue'
        })
    
    # Display top recommendations
    for i, rec in enumerate(recommendations[:6], 1):  # Top 6
        priority_colors = {
            'High': '#dc3545',
            'Medium': '#ffc107',
            'Low': '#17a2b8'
        }
        
        color = priority_colors.get(rec['priority'], '#6c757d')
        
        st.markdown(f"""
        <div class="warning-box">
        <strong>{i}. {rec['area']}</strong> 
        <span class="badge" style="background-color: {color}; color: white; float: right;">{rec['priority']} Priority</span><br>
        <br>
        <strong>📊 Current State:</strong> {rec['issue']}<br>
        <strong>🎯 Action:</strong> {rec['action']}<br>
        <strong>💫 Impact:</strong> {rec['impact']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tactical recommendations
    st.markdown("### 🎓 Specific Coaching Tactics by Person")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 For Alex Rodriguez (Product Manager)")
        
        if alex_meetings:
            alex_avg_chat = sum(m['chat_scores']['overall'] for m in alex_meetings) / len(alex_meetings)
            alex_avg_engagement = sum(m['act_predictions']['engagement'] for m in alex_meetings) / len(alex_meetings)
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>Current Scores:</strong><br>
            • CHAT: {alex_avg_chat*100:.0f}%<br>
            • Engagement: {alex_avg_engagement*100:.0f}%<br>
            <br>
            <strong>Recommended Tactics:</strong><br>
            <br>
            1. <strong>Strategic Time:</strong> Dedicate 20% of each 1:1 to big-picture product strategy<br>
            2. <strong>Conflict Coaching:</strong> Role-play difficult stakeholder conversations<br>
            3. <strong>Empowerment Questions:</strong> "What would you do if I wasn't here?"<br>
            4. <strong>Connect to Impact:</strong> "How does this advance our vision?"<br>
            5. <strong>Celebrate Wins:</strong> Recognize strategic thinking and problem-solving<br>
            6. <strong>Career Development:</strong> Quarterly discussions about growth path
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔬 For Javier Morales (QA Lead)")
        
        if javier_meetings:
            javier_avg_chat = sum(m['chat_scores']['overall'] for m in javier_meetings) / len(javier_meetings)
            javier_avg_ownership = sum(m['act_predictions']['ownership'] for m in javier_meetings) / len(javier_meetings)
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>Current Scores:</strong><br>
            • CHAT: {javier_avg_chat*100:.0f}%<br>
            • Ownership: {javier_avg_ownership*100:.0f}%<br>
            <br>
            <strong>Recommended Tactics:</strong><br>
            <br>
            1. <strong>Business Translation:</strong> Help articulate QA value to stakeholders<br>
            2. <strong>Technical Deep-Dives:</strong> Honor expertise while connecting to outcomes<br>
            3. <strong>Stakeholder Mapping:</strong> Identify and strengthen key relationships<br>
            4. <strong>Visibility:</strong> Create opportunities to showcase QA impact<br>
            5. <strong>Growth Path:</strong> Explicit career conversations quarterly<br>
            6. <strong>Process Improvement:</strong> Empower him to lead QA initiatives
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Next steps
    st.markdown("### ✅ Your Action Plan (Next 30 Days)")
    
    st.markdown(f"""
    <div class="success-box">
    <strong>Week 1: Immediate Focus</strong><br>
    • 📚 Review and practice {weakest.title()} dimension techniques<br>
    • 🎯 Set specific goal: Improve {weakest.title()} from {avg_by_dim[weakest]*100:.0f}% to {min(avg_by_dim[weakest]*100 + 15, 100):.0f}%<br>
    • 🔄 Adjust talk ratio toward 40-60% range<br>
    <br>
    <strong>Week 2-3: Practice & Refinement</strong><br>
    • ❓ Increase open-ended question ratio to 70%+<br>
    • 🎤 Record one 1:1 and self-assess using CHAT framework<br>
    • 🤝 Get peer feedback from another manager<br>
    <br>
    <strong>Week 4: Measure & Iterate</strong><br>
    • 📊 Compare metrics to previous month<br>
    • 🎯 Identify most improved dimension<br>
    • 📈 Set next month's development goals<br>
    • 🤝 Share learnings with team
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Additional resources
    st.markdown("### 📚 Recommended Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
        <strong>For {strongest.title()}:</strong><br>
        Keep strengthening your best skill<br>
        <br>
        • Study: Expert {strongest} techniques<br>
        • Practice: Teach others this skill<br>
        • Measure: Track consistency
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box">
        <strong>For {weakest.title()}:</strong><br>
        Priority development area<br>
        <br>
        • Study: {weakest.title()} best practices<br>
        • Practice: Daily micro-improvements<br>
        • Measure: Weekly progress check
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
        <strong>General Growth:</strong><br>
        Continuous improvement<br>
        <br>
        • Book: "The Coaching Habit"<br>
        • Practice: Role-play difficult conversations<br>
        • Reflect: Weekly self-assessment
        </div>
        """, unsafe_allow_html=True)


def show_team_meeting_analysis(team_meetings: List[Dict]):
    """Display analysis for team meetings"""
    
    st.markdown("## 👥 Team Meeting Analysis")
    
    if not team_meetings:
        st.warning("No team meetings found. Team meetings should have 3+ participants.")
        return
    
    # Meeting selector
    meeting_options = [f"{m['date_str']} - {m['file']}" for m in team_meetings]
    selected_idx = st.selectbox(
        "Select Team Meeting",
        range(len(team_meetings)),
        format_func=lambda i: meeting_options[i],
        key="team_meeting_selector"
    )
    
    meeting = team_meetings[selected_idx]
    metrics = meeting['team_metrics']
    
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {meeting['file']}")
        st.markdown(f"**Date:** {meeting['date_str']}")
        st.markdown(f"<span class='badge badge-team'>Team Meeting</span>", unsafe_allow_html=True)
    
    with col2:
        st.metric("Participants", metrics.participant_count)
    
    st.markdown("---")
    
    # Key metrics
    st.markdown("### 📊 Meeting Dynamics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Words", f"{meeting['total_words']:,}")
    
    with col2:
        st.metric("Turn Count", meeting['turn_count'])
    
    with col3:
        inclusion_pct = metrics.inclusion_score * 100
        st.metric(
            "Inclusion Score",
            f"{inclusion_pct:.0f}%",
            delta="Balanced" if inclusion_pct >= 70 else "Imbalanced",
            delta_color="normal" if inclusion_pct >= 70 else "inverse"
        )
    
    with col4:
        decision_pct = metrics.decision_quality * 100
        st.metric("Decision Quality", f"{decision_pct:.0f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_participation_pie_chart(meeting)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_participation_balance_gauge(metrics.inclusion_score)
        st.plotly_chart(fig, use_container_width=True)
    
    fig = create_interaction_network(meeting)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed breakdown
    st.markdown("### 🔍 Participation Breakdown")
    
    participation_data = []
    for speaker, pct in metrics.participation_distribution.items():
        participation_data.append({
            'Speaker': speaker,
            'Word %': f"{pct:.1f}%",
            'Turns': metrics.turn_distribution[speaker],
            'Avg Words/Turn': f"{(metrics.participation_distribution[speaker] / 100 * meeting['total_words']) / metrics.turn_distribution[speaker]:.0f}"
        })
    
    df = pd.DataFrame(participation_data)
    df = df.sort_values('Word %', ascending=False)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Insights
    st.markdown("### 💡 Team Dynamics Insights")
    
    participation_dict = metrics.participation_distribution
    max_speaker = max(participation_dict, key=participation_dict.get)
    min_speaker = min(participation_dict, key=participation_dict.get)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if participation_dict[max_speaker] > 40:
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ Dominant Speaker Detected</strong><br>
            {max_speaker} spoke {participation_dict[max_speaker]:.0f}% of the time. 
            Consider techniques to encourage more balanced participation.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
            <strong>✅ Balanced Participation</strong><br>
            No single participant dominated the conversation. 
            Good facilitation!
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if participation_dict[min_speaker] < 10:
            st.markdown(f"""
            <div class="warning-box">
            <strong>👤 Quiet Participant</strong><br>
            {min_speaker} only spoke {participation_dict[min_speaker]:.0f}% of the time. 
            Consider directly inviting their input: "What do you think, {min_speaker}?"
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
            <strong>✅ Everyone Contributed</strong><br>
            All participants spoke at least 10% of the time. 
            Strong inclusion!
            </div>
            """, unsafe_allow_html=True)
    
    if metrics.topic_coverage:
        st.markdown("### 📚 Topics Discussed")
        topics_str = ", ".join(metrics.topic_coverage[:10])
        st.markdown(f"""
        <div class="team-box">
        <strong>Detected Topics:</strong> {topics_str}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🎯 Meeting Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">1:1 Conversations + Team Meetings - Complete Integration</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Source")
        
        folder_path = st.text_input(
            "Meeting Transcripts Folder",
            value="meetings",
            help="Folder containing both 1:1 and team meeting transcripts"
        )
        
        if st.button("🔄 Load/Refresh Data", use_container_width=True):
            st.cache_data.clear()
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
        show_action_items = st.checkbox("Show Action Items", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Meeting Types")
        st.markdown("""
        **1:1 Meetings** (2 participants):
        - Manager-employee conversations
        - CHAT + ACT frameworks
        - Full conversational analytics
        
        **Team Meetings** (3+ participants):
        - Group discussions
        - Participation balance
        - Collaboration patterns
        """)
    
    # Load data
    one_on_one_meetings, team_meetings = load_all_meetings(folder_path)
    
    if not one_on_one_meetings and not team_meetings:
        st.error("No meeting data found. Please check the folder path.")
        st.info("""
        **Expected formats:**
        - 1:1: `FirstName_LastName_DDMmmYY.docx` with "Name:" speaker format
        - Team: Any .docx with [Speaker N] format
        """)
        return
    
    # Summary
    st.markdown("## 📊 Meeting Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Meetings", len(one_on_one_meetings) + len(team_meetings))
    
    with col2:
        st.metric("1:1 Meetings", len(one_on_one_meetings))
        st.markdown('<span class="badge badge-one-on-one">Individual</span>', unsafe_allow_html=True)
    
    with col3:
        st.metric("Team Meetings", len(team_meetings))
        st.markdown('<span class="badge badge-team">Collaborative</span>', unsafe_allow_html=True)
    
    with col4:
        if team_meetings:
            avg_participants = sum(m['team_metrics'].participant_count for m in team_meetings) / len(team_meetings)
            st.metric("Avg Team Size", f"{avg_participants:.1f}")
    
    st.markdown("---")
    
    # Filter by person for 1:1s
    alex_meetings = [m for m in one_on_one_meetings if 'Alex' in m['person']]
    javier_meetings = [m for m in one_on_one_meetings if 'Javier' in m['person']]
    
    # Main tabs
    if one_on_one_meetings and team_meetings:
        # Both types available - show all tabs
        tabs = st.tabs([
            "📊 1:1 Overview",
            "💬 1:1 Meeting Analysis",
            "📈 1:1 Trends",
            "👥 Person Comparison",
            "💡 1:1 Coaching",
            "🤝 Team Meetings",
            "🎯 All Recommendations"
        ])
        
        with tabs[0]:
            if one_on_one_meetings:
                show_overview(one_on_one_meetings, alex_meetings, javier_meetings)
        
        with tabs[1]:
            if one_on_one_meetings:
                show_meeting_analysis(one_on_one_meetings, show_detailed_analysis, show_action_items)
        
        with tabs[2]:
            if one_on_one_meetings:
                show_trends(one_on_one_meetings, alex_meetings, javier_meetings)
        
        with tabs[3]:
            if alex_meetings and javier_meetings:
                show_comparison(alex_meetings, javier_meetings)
            else:
                st.info("Need meetings from both Alex and Javier for comparison")
        
        with tabs[4]:
            if one_on_one_meetings:
                show_coaching_insights(one_on_one_meetings, alex_meetings, javier_meetings)
        
        with tabs[5]:
            if team_meetings:
                show_team_meeting_analysis(team_meetings)
        
        with tabs[6]:
            st.markdown("## 🎯 Combined Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### For 1:1 Meetings")
                if one_on_one_meetings:
                    avg_chat = sum(m['chat_scores']['overall'] for m in one_on_one_meetings) / len(one_on_one_meetings)
                    avg_talk_ratio = sum(m['talk_ratio'] for m in one_on_one_meetings) / len(one_on_one_meetings)
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>Performance Summary:</strong><br>
                    • Average CHAT Score: {avg_chat * 100:.0f}%<br>
                    • Average Talk Ratio: {avg_talk_ratio:.0f}%<br>
                    • Total 1:1s: {len(one_on_one_meetings)}<br>
                    <br>
                    See "1:1 Coaching" tab for detailed recommendations.
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### For Team Meetings")
                if team_meetings:
                    avg_inclusion = sum(m['team_metrics'].inclusion_score for m in team_meetings) / len(team_meetings)
                    
                    st.markdown(f"""
                    <div class="team-box">
                    <strong>Performance Summary:</strong><br>
                    • Average Inclusion: {avg_inclusion * 100:.0f}%<br>
                    • Total Team Meetings: {len(team_meetings)}<br>
                    <br>
                    Focus on balanced participation and clear action items.
                    </div>
                    """, unsafe_allow_html=True)
    
    elif one_on_one_meetings:
        # Only 1:1 meetings
        tabs = st.tabs([
            "📊 Overview",
            "💬 Meeting Analysis",
            "📈 Trends & Patterns",
            "👥 Person Comparison",
            "💡 Coaching Insights"
        ])
        
        with tabs[0]:
            show_overview(one_on_one_meetings, alex_meetings, javier_meetings)
        
        with tabs[1]:
            show_meeting_analysis(one_on_one_meetings, show_detailed_analysis, show_action_items)
        
        with tabs[2]:
            show_trends(one_on_one_meetings, alex_meetings, javier_meetings)
        
        with tabs[3]:
            if alex_meetings and javier_meetings:
                show_comparison(alex_meetings, javier_meetings)
            else:
                st.info("Need meetings from both Alex and Javier for comparison")
        
        with tabs[4]:
            show_coaching_insights(one_on_one_meetings, alex_meetings, javier_meetings)
    
    elif team_meetings:
        # Only team meetings
        tabs = st.tabs([
            "👥 Team Meetings",
            "🎯 Recommendations"
        ])
        
        with tabs[0]:
            show_team_meeting_analysis(team_meetings)
        
        with tabs[1]:
            st.markdown("## 🎯 Team Meeting Recommendations")
            
            latest_team = team_meetings[-1]
            metrics = latest_team['team_metrics']
            
            recommendations = []
            
            if metrics.inclusion_score < 0.7:
                recommendations.append({
                    'priority': 'High',
                    'area': 'Participation Balance',
                    'issue': f'Inclusion score at {metrics.inclusion_score*100:.0f}%',
                    'action': 'Use round-robin to ensure everyone speaks.'
                })
            
            participation = metrics.participation_distribution
            max_speaker = max(participation, key=participation.get)
            if participation[max_speaker] > 40:
                recommendations.append({
                    'priority': 'Medium',
                    'area': 'Speaker Balance',
                    'issue': f'{max_speaker} dominated with {participation[max_speaker]:.0f}%',
                    'action': f'Politely redirect conversations to other participants.'
                })
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    color = '#dc3545' if rec['priority'] == 'High' else '#ffc107'
                    st.markdown(f"""
                    <div class="warning-box">
                    <strong>{i}. {rec['area']}</strong>
                    <span class="badge" style="background-color: {color}; color: white; float: right;">{rec['priority']}</span><br>
                    <strong>Issue:</strong> {rec['issue']}<br>
                    <strong>Action:</strong> {rec['action']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ Team meetings showing excellent facilitation!")


if __name__ == "__main__":
    main()
