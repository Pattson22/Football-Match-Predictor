from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import joblib
import numpy as np
import requests
import os
import json
import time
import threading
import smtplib
import math
import csv
import io
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone, timedelta

load_dotenv()

# --- App Setup ---
app = Flask(__name__)
base_dir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback_dev_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(base_dir, 'project.db')

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)


class Bet(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    match_date = db.Column(db.String(10))
    home_team  = db.Column(db.String(80))
    away_team  = db.Column(db.String(80))
    league     = db.Column(db.String(80))
    bet_on     = db.Column(db.String(1))    # H, D, A
    odds       = db.Column(db.Float)
    stake      = db.Column(db.Float)
    status     = db.Column(db.String(1), default='P')  # P=pending, W=win, L=loss, V=void
    pnl        = db.Column(db.Float, default=0.0)


class PredictionLog(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    logged_at  = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    match_date = db.Column(db.String(10))
    home_team  = db.Column(db.String(80))
    away_team  = db.Column(db.String(80))
    league     = db.Column(db.String(80))
    pred_label = db.Column(db.String(1))    # H, D, A
    pred_prob  = db.Column(db.Integer)      # 0-100
    actual     = db.Column(db.String(1), nullable=True)


class OddsSnapshot(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    match_date = db.Column(db.String(10))
    home_team  = db.Column(db.String(80))
    away_team  = db.Column(db.String(80))
    odds_h     = db.Column(db.Float)
    odds_d     = db.Column(db.Float)
    odds_a     = db.Column(db.Float)
    fetched_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class UserSettings(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    bankroll       = db.Column(db.Float, default=1000.0)
    default_stake  = db.Column(db.Float, default=10.0)
    kelly_fraction = db.Column(db.Float, default=0.25)
    notify_email   = db.Column(db.Boolean, default=True)
    notify_browser = db.Column(db.Boolean, default=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- Configuration ---
ODDS_API_KEY          = os.environ.get('ODDS_API_KEY', '')
FOOTBALL_DATA_API_KEY = os.environ.get('FOOTBALL_DATA_API_KEY', '')
REGIONS               = 'eu'
ODDS_FORMAT           = 'decimal'

LEAGUES = [
    ('soccer_epl',                'PL'),
    ('soccer_efl_champ',          'ELC'),
    ('soccer_italy_serie_a',      'SA'),
    ('soccer_spain_la_liga',      'PD'),
    ('soccer_germany_bundesliga', 'BL1'),
    ('soccer_france_ligue_1',     'FL1'),
]

FALLBACK = {
    'form_pts':      1.2,
    'gspg':          1.35,
    'gcpg':          1.35,
    'elo':           1500.0,
    'season_ppg':    1.2,
    'season_gd':     0.0,
    'home_form_pts': 1.4,
    'home_gspg':     1.5,
    'home_gcpg':     1.2,
    'away_form_pts': 1.0,
    'away_gspg':     1.2,
    'away_gcpg':     1.5,
    'momentum':      0.0,
    'rest_days':     7.0,
}

API_TO_DATA_NAME = {
    'Manchester City':         'Man City',
    'Manchester United':       'Man United',
    'Wolverhampton Wanderers': 'Wolves',
    'Nottingham Forest':       "Nott'm Forest",
    'Tottenham Hotspur':       'Tottenham',
    'Newcastle United':        'Newcastle',
    'West Ham United':         'West Ham',
    'Brighton & Hove Albion':  'Brighton',
    'Leicester City':          'Leicester',
    'Luton Town':              'Luton',
    'Sheffield Utd':           'Sheffield United',
    'Ipswich Town':            'Ipswich',
}

CREST_ALIASES = {
    'Inter Milan':         'Inter',
    'Como':                'Como 1907',
    'Brighton and Hove Albion': 'Brighton & Hove Albion',
    'Bayern Munich':       'Bayern München',
    'Bayer Leverkusen':    'Bayer 04 Leverkusen',
    'RB Leipzig':          'RasenBallsport Leipzig',
    'Atletico Madrid':     'Atlético de Madrid',
    'Real Betis':          'Real Betis Balompié',
    'Celta Vigo':          'Celta de Vigo',
    'Paris Saint Germain': 'PSG',
    'Marseille':           'Olympique de Marseille',
    'Lyon':                'Olympique Lyonnais',
    'Monaco':              'AS Monaco',
    'Nantes':              'FC Nantes',
    'Nice':                'OGC Nice',
    'Lille':               'LOSC Lille',
    'Rennes':              'Stade Rennais FC',
    'Lens':                'RC Lens',
    'Strasbourg':          'RC Strasbourg Alsace',
    'Reims':               'Stade de Reims',
    'Montpellier':         'Montpellier HSC',
    'Brest':               'Stade Brestois 29',
    'Saint-Etienne':       'AS Saint-Étienne',
    'Toulouse':            'Toulouse FC',
    'Auxerre':             'AJ Auxerre',
    'Le Havre':            'HAC Le Havre',
}

LOGO_MAP = {
    'Arsenal': 'arsenal.com', 'Aston Villa': 'avfc.co.uk', 'Bournemouth': 'afcb.co.uk',
    'Brentford': 'brentfordfc.com', 'Brighton & Hove Albion': 'brightonandhovealbion.com',
    'Brighton': 'brightonandhovealbion.com', 'Chelsea': 'chelseafc.com', 'Crystal Palace': 'cpfc.co.uk',
    'Everton': 'evertonfc.com', 'Fulham': 'fulhamfc.com', 'Liverpool': 'liverpoolfc.com',
    'Luton Town': 'lutontown.co.uk', 'Manchester City': 'mancity.com', 'Man City': 'mancity.com',
    'Manchester United': 'manutd.com', 'Man Utd': 'manutd.com', 'Newcastle United': 'nufc.co.uk',
    'Newcastle': 'nufc.co.uk', 'Nottingham Forest': 'nottinghamforest.co.uk', 'Sheffield United': 'sufc.co.uk',
    'Tottenham Hotspur': 'tottenhamhotspur.com', 'Spurs': 'tottenhamhotspur.com', 'West Ham United': 'whufc.com',
    'West Ham': 'whufc.com', 'Wolverhampton Wanderers': 'wolves.co.uk', 'Wolves': 'wolves.co.uk',
    'Burnley': 'burnleyfootballclub.com', 'Ipswich Town': 'itfc.co.uk',
}


# --- Alert Config ---
ALERT_ENABLED      = os.environ.get('ALERT_ENABLED', 'false').lower() == 'true'
ALERT_EMAIL        = os.environ.get('ALERT_EMAIL', '')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', '')
_alerted_bets: set = set()  # tracks bet keys already emailed this session


# --- Poisson / Kelly helpers ---
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0 or k < 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_predict(home_team: str, away_team: str) -> dict:
    home_dn = normalise(home_team)
    away_dn = normalise(away_team)
    hf = team_form.get(home_dn) or team_form.get(home_team) or {}
    af = team_form.get(away_dn) or team_form.get(away_team) or {}
    avg = 1.35
    lam_h = max(0.3, hf.get('gspg', avg) / avg * af.get('gcpg', avg) / avg * avg * 1.10)
    lam_a = max(0.3, af.get('gspg', avg) / avg * hf.get('gcpg', avg) / avg * avg)
    ph = pd_ = pa = 0.0
    scores: dict = {}
    for h in range(7):
        for a in range(7):
            p = poisson_pmf(h, lam_h) * poisson_pmf(a, lam_a)
            scores[f"{h}-{a}"] = round(p * 100, 1)
            if h > a:   ph  += p
            elif h == a: pd_ += p
            else:        pa  += p
    top = sorted(scores.items(), key=lambda x: -x[1])[:8]
    return {'ph': round(ph*100,1), 'pd': round(pd_*100,1), 'pa': round(pa*100,1),
            'lam_h': round(lam_h,2), 'lam_a': round(lam_a,2), 'top_scores': top}


def kelly_stake(model_prob_pct: float, odds: float, bankroll: float, fraction: float = 0.25) -> float:
    p = model_prob_pct / 100
    b = odds - 1
    if b <= 0:
        return 0.0
    k = (b * p - (1 - p)) / b
    return round(max(0.0, bankroll * k * fraction), 2)


def get_user_settings() -> UserSettings:
    s = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not s:
        s = UserSettings(user_id=current_user.id)
        db.session.add(s)
        db.session.commit()
    return s


def _send_value_bet_email(value_bets: list) -> None:
    if not ALERT_EMAIL or not GMAIL_APP_PASSWORD or GMAIL_APP_PASSWORD.startswith('your_'):
        print("Email alerts not configured — skipping send.")
        return
    rows = ''.join(
        f"""<tr>
          <td style="padding:10px 12px;border-bottom:1px solid #30363d">
            <strong>{vb['match']['home_team']} vs {vb['match']['away_team']}</strong><br>
            <span style="color:#8b949e;font-size:12px">{vb['match']['league']} · {vb['match']['kickoff']}</span>
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #30363d;color:#2ea043;font-weight:700">
            {vb['label_text']}
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #30363d;text-align:center">{vb['odds']}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #30363d;text-align:center">{vb['model_prob']}%</td>
          <td style="padding:10px 12px;border-bottom:1px solid #30363d;text-align:center;color:#2ea043;font-weight:700">
            +{vb['ev']}%
          </td>
        </tr>"""
        for vb in value_bets
    )
    html = f"""<html><body style="background:#0d1117;color:#e6edf3;font-family:Inter,sans-serif;padding:24px">
      <h2 style="color:#e6edf3;margin-bottom:4px">⚽ {len(value_bets)} Value Bet{'s' if len(value_bets)!=1 else ''} Found</h2>
      <p style="color:#8b949e;margin-bottom:20px;font-size:14px">Football Predictor · {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}</p>
      <table style="width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;overflow:hidden">
        <thead>
          <tr style="background:#21262d">
            <th style="padding:10px 12px;text-align:left;font-size:12px;color:#8b949e">Match</th>
            <th style="padding:10px 12px;text-align:left;font-size:12px;color:#8b949e">Bet</th>
            <th style="padding:10px 12px;text-align:center;font-size:12px;color:#8b949e">Odds</th>
            <th style="padding:10px 12px;text-align:center;font-size:12px;color:#8b949e">Model %</th>
            <th style="padding:10px 12px;text-align:center;font-size:12px;color:#8b949e">EV</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="color:#8b949e;font-size:12px;margin-top:16px">Visit <a href="http://127.0.0.1:5000/best-bets" style="color:#2ea043">Football Predictor</a> to log bets.</p>
    </body></html>"""

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"⚽ {len(value_bets)} Value Bet{'s' if len(value_bets)!=1 else ''} — Football Predictor"
    msg['From']    = ALERT_EMAIL
    msg['To']      = ALERT_EMAIL
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10) as smtp:
            smtp.login(ALERT_EMAIL, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"Value bet alert sent → {ALERT_EMAIL} ({len(value_bets)} bets)")
    except Exception as e:
        print(f"Email alert failed: {e}")


def _alert_worker() -> None:
    time.sleep(60)  # wait for first dashboard load before starting checks
    while True:
        time.sleep(30 * 60)
        if not _matches_cache:
            continue
        new_bets = []
        for m in _matches_cache.values():
            p = m.get('pred')
            if not p:
                continue
            for outcome, label in [('val_h', 'H'), ('val_d', 'D'), ('val_a', 'A')]:
                if not p.get(outcome):
                    continue
                key = f"{m['home_team']}||{m['away_team']}||{label}"
                if key in _alerted_bets:
                    continue
                _alerted_bets.add(key)
                odds_map = {'H': 'b365h', 'D': 'b365d', 'A': 'b365a'}
                prob_map = {'H': 'h',     'D': 'd',     'A': 'a'}
                ev_map   = {'H': 'ev_h',  'D': 'ev_d',  'A': 'ev_a'}
                new_bets.append({
                    'match':      m,
                    'outcome':    label,
                    'label_text': {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[label],
                    'odds':       m[odds_map[label]],
                    'model_prob': p[prob_map[label]],
                    'ev':         p[ev_map[label]],
                })
        if new_bets:
            _send_value_bet_email(new_bets)


if ALERT_ENABLED:
    threading.Thread(target=_alert_worker, daemon=True).start()
    print("Email alert worker started.")


# --- Auto-settle ---
def _result_from_winner(winner: str) -> str | None:
    if winner == 'HOME_TEAM': return 'H'
    if winner == 'AWAY_TEAM': return 'A'
    if winner == 'DRAW':      return 'D'
    return None


def _name_set(fd_team: dict) -> set:
    short = fd_team.get('shortName', '')
    full  = fd_team.get('name', '')
    stripped = full[:-3] if full.endswith(' FC') else full
    return {n for n in (short, full, stripped, normalise(short), normalise(full)) if n}


def auto_settle() -> None:
    from datetime import date
    date_from = (date.today() - timedelta(days=7)).isoformat()
    date_to   = date.today().isoformat()
    with app.app_context():
        for _, comp_code in LEAGUES:
            try:
                resp = requests.get(
                    f'https://api.football-data.org/v4/competitions/{comp_code}/matches',
                    params={'status': 'FINISHED', 'dateFrom': date_from, 'dateTo': date_to},
                    headers={'X-Auth-Token': FOOTBALL_DATA_API_KEY},
                    timeout=8,
                )
                resp.raise_for_status()
                for m in resp.json().get('matches', []):
                    result = _result_from_winner(m.get('score', {}).get('winner'))
                    if not result:
                        continue
                    match_date  = m.get('utcDate', '')[:10]
                    home_names  = _name_set(m.get('homeTeam', {}))
                    away_names  = _name_set(m.get('awayTeam', {}))
                    # Settle prediction logs
                    for pred in PredictionLog.query.filter_by(actual=None, match_date=match_date).all():
                        if normalise(pred.home_team) in home_names and normalise(pred.away_team) in away_names:
                            pred.actual = result
                    # Settle bets
                    for bet in Bet.query.filter_by(status='P', match_date=match_date).all():
                        if normalise(bet.home_team) in home_names and normalise(bet.away_team) in away_names:
                            bet.status = 'W' if bet.bet_on == result else 'L'
                            bet.pnl    = round(bet.stake * bet.odds - bet.stake, 2) if bet.status == 'W' else -bet.stake
                db.session.commit()
                print(f"Auto-settle: processed {comp_code}")
            except Exception as e:
                print(f"Auto-settle error ({comp_code}): {e}")
            time.sleep(7)


def _auto_settle_worker() -> None:
    time.sleep(120)
    while True:
        auto_settle()
        time.sleep(3600)

threading.Thread(target=_auto_settle_worker, daemon=True).start()


# --- Load Model & Static Artifacts ---
try:
    _md            = joblib.load('football_model.joblib')
    model          = _md['model']
    model_features = _md['features']
    print(f"Model loaded — {len(model_features)} features: {model_features}")
except FileNotFoundError:
    print("Error: football_model.joblib not found. Run train_model.py first.")
    model, model_features = None, []
except Exception as e:
    print(f"Error loading model: {e}")
    model, model_features = None, []

model_accuracy = "N/A"
try:
    with open("accuracy.txt") as f:
        model_accuracy = f.read().strip()
    print(f"Model accuracy: {model_accuracy}%")
except FileNotFoundError:
    pass

team_form = {}
try:
    with open("team_form.json") as f:
        team_form = json.load(f)
    print(f"Loaded form data for {len(team_form)} teams.")
except FileNotFoundError:
    print("Warning: team_form.json not found. Run prepare_data.py.")

team_elo = {}
try:
    with open("team_elo.json") as f:
        team_elo = json.load(f)
    print(f"Loaded Elo ratings for {len(team_elo)} teams.")
except FileNotFoundError:
    print("Warning: team_elo.json not found. Run prepare_data.py.")

h2h_cache = {}
try:
    with open("h2h_cache.json") as f:
        h2h_cache = json.load(f)
    print(f"Loaded H2H data for {len(h2h_cache)//2} matchups.")
except FileNotFoundError:
    print("Warning: h2h_cache.json not found. Run prepare_data.py.")

# In-memory match cache for detail page (refreshed each dashboard load)
_matches_cache: dict = {}  # key: "home||away" → match dict


# --- Crests Cache ---
_crests_cache: dict = {}

def _register_crest(name: str, url: str) -> None:
    if not name or not url:
        return
    _crests_cache[name] = url
    stripped = name[:-3] if name.endswith(' FC') else name
    if stripped != name:
        _crests_cache[stripped] = url
    if '&' in name:
        _crests_cache[name.replace('&', 'and')] = url
        if stripped != name:
            _crests_cache[stripped.replace('&', 'and')] = url


_CRESTS_FILE = os.path.join(base_dir, 'crests_cache.json')

def fetch_crests() -> dict:
    global _crests_cache
    if _crests_cache:
        return _crests_cache
    try:
        with open(_CRESTS_FILE) as f:
            _crests_cache = json.load(f)
        print(f"Crests loaded from file: {len(_crests_cache)} entries.")
        return _crests_cache
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    comp_codes = [fd_code for _, fd_code in LEAGUES]
    for i, code in enumerate(comp_codes):
        if i > 0:
            time.sleep(7)
        try:
            resp = requests.get(
                f'https://api.football-data.org/v4/competitions/{code}/teams',
                headers={'X-Auth-Token': FOOTBALL_DATA_API_KEY},
                timeout=10,
            )
            resp.raise_for_status()
            for team in resp.json().get('teams', []):
                crest = team.get('crest', '')
                for key in (team.get('shortName'), team.get('name'), team.get('tla')):
                    _register_crest(key, crest)
            print(f"  Crests fetched: {code}")
        except Exception as e:
            print(f"  Could not fetch crests for {code}: {e}")
    print(f"Crests fetched: {len(_crests_cache)} entries.")
    try:
        with open(_CRESTS_FILE, 'w') as f:
            json.dump(_crests_cache, f)
    except Exception as e:
        print(f"Could not save crests file: {e}")
    return _crests_cache

fetch_crests()


# --- Known-team check ---
def _is_known_team(api_name: str) -> bool:
    dn = normalise(api_name)
    stripped = api_name[:-3] if api_name.endswith(' FC') else api_name
    for name in {api_name, dn, stripped}:
        if name in team_elo or name in team_form or name in _form_cache:
            return True
    return False


# --- Form Cache ---
_form_cache: dict = {}
_FORM_FILE = os.path.join(base_dir, 'form_cache.json')

def fetch_form() -> dict:
    global _form_cache
    if _form_cache:
        return _form_cache
    try:
        if os.path.exists(_FORM_FILE) and (time.time() - os.path.getmtime(_FORM_FILE)) < 86400:
            with open(_FORM_FILE) as f:
                _form_cache = json.load(f)
            print(f"Form loaded from file: {len(_form_cache)} teams.")
            return _form_cache
    except (json.JSONDecodeError, OSError):
        pass
    from datetime import date, timedelta
    today     = date.today()
    date_from = (today - timedelta(days=60)).isoformat()
    date_to   = today.isoformat()
    for _, comp_code in LEAGUES:
        try:
            resp = requests.get(
                f'https://api.football-data.org/v4/competitions/{comp_code}/matches',
                params={'status': 'FINISHED', 'dateFrom': date_from, 'dateTo': date_to},
                headers={'X-Auth-Token': FOOTBALL_DATA_API_KEY},
                timeout=8,
            )
            resp.raise_for_status()
            recent = sorted(resp.json().get('matches', []),
                            key=lambda m: m.get('utcDate', ''), reverse=True)
            for match in recent:
                winner  = match.get('score', {}).get('winner')
                h_score = match.get('score', {}).get('fullTime', {}).get('home', '?')
                a_score = match.get('score', {}).get('fullTime', {}).get('away', '?')
                home_t  = match.get('homeTeam', {})
                away_t  = match.get('awayTeam', {})
                for team, is_home, opp_t in [(home_t, True, away_t), (away_t, False, home_t)]:
                    short     = team.get('shortName', '')
                    full      = team.get('name', '')
                    canonical = short or full
                    if not canonical:
                        continue
                    entries = _form_cache.setdefault(canonical, [])
                    stripped = full[:-3] if full.endswith(' FC') else full
                    for alias in {full, stripped} - {canonical}:
                        if alias:
                            _form_cache.setdefault(alias, entries)
                    if len(entries) >= 5:
                        continue
                    if winner == 'HOME_TEAM':
                        r = 'W' if is_home else 'L'
                    elif winner == 'AWAY_TEAM':
                        r = 'L' if is_home else 'W'
                    elif winner == 'DRAW':
                        r = 'D'
                    else:
                        continue
                    opp_name = opp_t.get('shortName') or opp_t.get('name', '?')
                    entries.append({'r': r, 'score': f"{h_score}-{a_score}",
                                    'opp': opp_name, 'venue': 'H' if is_home else 'A'})
        except Exception as e:
            print(f"Could not fetch form for {comp_code}: {e}")
        time.sleep(7)
    print(f"Form fetched for {len(_form_cache)} teams.")
    try:
        with open(_FORM_FILE, 'w') as f:
            json.dump(_form_cache, f)
    except Exception as e:
        print(f"Could not save form file: {e}")
    return _form_cache

fetch_form()


def get_team_form(name: str) -> list:
    form = _form_cache.get(name)
    if form is not None:
        return form
    stripped = name[:-3] if name.endswith(' FC') else name
    return _form_cache.get(stripped, [])


# --- Standings Cache ---
_standings_cache = {'data': {}, 'ts': 0}
CACHE_TTL = 3600
_STANDINGS_FILE       = os.path.join(base_dir, 'standings_cache.json')
_LEAGUES_DISPLAY_FILE = os.path.join(base_dir, 'leagues_display.json')
_leagues_display      = []
_odds_refreshed_at    = None
_FD_CODE_TO_NAME = {
    'PL':  'Premier League', 'ELC': 'Championship',
    'SA':  'Serie A',        'PD':  'La Liga',
    'BL1': 'Bundesliga',     'FL1': 'Ligue 1',
}


def fetch_standings():
    global _leagues_display
    if FOOTBALL_DATA_API_KEY.startswith('YOUR_'):
        return {}
    now = time.time()
    if now - _standings_cache['ts'] < CACHE_TTL and _standings_cache['data']:
        return _standings_cache['data']
    try:
        if os.path.exists(_STANDINGS_FILE) and (now - os.path.getmtime(_STANDINGS_FILE)) < CACHE_TTL:
            with open(_STANDINGS_FILE) as f:
                _standings_cache['data'] = json.load(f)
            _standings_cache['ts'] = now
            print(f"Standings loaded from file: {len(_standings_cache['data'])} teams.")
            try:
                with open(_LEAGUES_DISPLAY_FILE) as f:
                    _leagues_display = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
            return _standings_cache['data']
    except (json.JSONDecodeError, OSError):
        pass
    _leagues_display = []
    standings = {}
    comp_codes = [fd_code for _, fd_code in LEAGUES]
    for i, code in enumerate(comp_codes):
        if i > 0:
            time.sleep(7)
        try:
            resp = requests.get(
                f'https://api.football-data.org/v4/competitions/{code}/standings',
                headers={'X-Auth-Token': FOOTBALL_DATA_API_KEY},
                timeout=8,
            )
            resp.raise_for_status()
            table = resp.json()['standings'][0]['table']
            for entry in table:
                short  = entry['team']['shortName']
                full   = entry['team']['name']
                played = entry['playedGames']
                data = {
                    'position': entry['position'],
                    'ppg':      round(entry['points'] / max(1, played), 3),
                    'gd':       float(entry['goalDifference']),
                }
                standings[short] = data
                stripped = full[:-3] if full.endswith(' FC') else full
                for alias in {full, stripped}:
                    if alias and alias != short:
                        standings[alias] = data
            _leagues_display.append({
                'name':  _FD_CODE_TO_NAME.get(code, code),
                'code':  code,
                'table': [{'pos': e['position'], 'team': e['team']['shortName'],
                            'crest': e['team'].get('crest', ''), 'played': e['playedGames'],
                            'pts': e['points'], 'gd': e['goalDifference']} for e in table],
            })
            print(f"Standings fetched: {code} ({len(table)} teams)")
        except Exception as e:
            print(f"Could not fetch standings for {code}: {e}")
    _standings_cache['data'] = standings
    _standings_cache['ts']   = now
    try:
        with open(_STANDINGS_FILE, 'w') as f:
            json.dump(standings, f)
    except Exception as e:
        print(f"Could not save standings file: {e}")
    try:
        with open(_LEAGUES_DISPLAY_FILE, 'w') as f:
            json.dump(_leagues_display, f)
    except Exception as e:
        print(f"Could not save leagues display file: {e}")
    print(f"Standings refreshed — {len(standings)} teams across {len(comp_codes)} leagues.")
    return standings

fetch_standings()


# --- Helpers ---
def format_kickoff(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%a %d %b · %H:%M UTC')
    except Exception:
        return ''


def get_logo_url(team_name):
    crests = _crests_cache
    crest = crests.get(team_name)
    if crest:
        return crest
    aliased = CREST_ALIASES.get(team_name)
    if aliased:
        crest = crests.get(aliased)
        if crest:
            return crest
    data_name = normalise(team_name)
    crest = crests.get(data_name)
    if crest:
        return crest
    domain = LOGO_MAP.get(team_name) or LOGO_MAP.get(data_name)
    return f'https://logo.clearbit.com/{domain}' if domain else \
           'https://upload.wikimedia.org/wikipedia/commons/d/d3/Soccerball.svg'


def normalise(api_name):
    return API_TO_DATA_NAME.get(api_name, api_name)


def _build_features(avg_H, avg_D, avg_A, hf, af, home_api_name='', away_api_name=''):
    raw_h = 1.0 / avg_H
    raw_d = 1.0 / avg_D
    raw_a = 1.0 / avg_A
    total = raw_h + raw_d + raw_a
    imp_H = round(raw_h / total, 4)
    imp_D = round(raw_d / total, 4)
    imp_A = round(raw_a / total, 4)

    home_dn = normalise(API_TO_DATA_NAME.get(home_api_name, home_api_name))
    away_dn = normalise(API_TO_DATA_NAME.get(away_api_name, away_api_name))
    h2h = h2h_cache.get(f"{home_dn}||{away_dn}", {})

    return {
        "avg_H":                  avg_H,
        "avg_D":                  avg_D,
        "avg_A":                  avg_A,
        "imp_H":                  imp_H,
        "imp_D":                  imp_D,
        "imp_A":                  imp_A,
        "home_elo":               hf['elo'],
        "away_elo":               af['elo'],
        "elo_diff":               round(hf['elo'] - af['elo'], 1),
        "home_form_pts":          hf['form_pts'],
        "home_gspg":              hf['gspg'],
        "home_gcpg":              hf['gcpg'],
        "away_form_pts":          af['form_pts'],
        "away_gspg":              af['gspg'],
        "away_gcpg":              af['gcpg'],
        "home_venue_form_pts":    hf['venue_form_pts'],
        "home_venue_gspg":        hf['venue_gspg'],
        "home_venue_gcpg":        hf['venue_gcpg'],
        "away_venue_form_pts":    af['venue_form_pts'],
        "away_venue_gspg":        af['venue_gspg'],
        "away_venue_gcpg":        af['venue_gcpg'],
        "home_season_ppg":        hf['season_ppg'],
        "home_season_gd":         hf['season_gd'],
        "away_season_ppg":        af['season_ppg'],
        "away_season_gd":         af['season_gd'],
        "form_pts_diff":          round(hf['form_pts']   - af['form_pts'],   3),
        "season_ppg_diff":        round(hf['season_ppg'] - af['season_ppg'], 3),
        "home_attack_vs_away_def": round(hf['gspg'] - af['gcpg'], 3),
        "away_attack_vs_home_def": round(af['gspg'] - hf['gcpg'], 3),
        "home_momentum":          hf['momentum'],
        "away_momentum":          af['momentum'],
        "home_rest_days":         hf['rest_days'],
        "away_rest_days":         af['rest_days'],
        "h2h_home_win_rate":      h2h.get('hw', 0.46),
        "h2h_draw_rate":          h2h.get('dr', 0.26),
    }


def get_team_features(api_name, standings, is_home=True):
    data_name = normalise(api_name)

    form = team_form.get(data_name) or team_form.get(api_name) or {}
    elo  = team_elo.get(data_name)  or team_elo.get(api_name)  or FALLBACK['elo']

    standing = standings.get(api_name) or standings.get(data_name) or {}
    ppg = standing.get('ppg', FALLBACK['season_ppg'])
    gd  = standing.get('gd',  FALLBACK['season_gd'])

    if is_home:
        venue_form_pts = form.get('home_form_pts', FALLBACK['home_form_pts'])
        venue_gspg     = form.get('home_gspg',     FALLBACK['home_gspg'])
        venue_gcpg     = form.get('home_gcpg',     FALLBACK['home_gcpg'])
    else:
        venue_form_pts = form.get('away_form_pts', FALLBACK['away_form_pts'])
        venue_gspg     = form.get('away_gspg',     FALLBACK['away_gspg'])
        venue_gcpg     = form.get('away_gcpg',     FALLBACK['away_gcpg'])

    form_pts  = form.get('form_pts', FALLBACK['form_pts'])
    form3_pts = form.get('form3_pts', form_pts)

    return {
        'elo':            float(elo),
        'form_pts':       form_pts,
        'gspg':           form.get('gspg', FALLBACK['gspg']),
        'gcpg':           form.get('gcpg', FALLBACK['gcpg']),
        'season_ppg':     ppg,
        'season_gd':      gd,
        'venue_form_pts': float(venue_form_pts),
        'venue_gspg':     float(venue_gspg),
        'venue_gcpg':     float(venue_gcpg),
        'momentum':       round(float(form3_pts) - float(form_pts), 3),
        'rest_days':      FALLBACK['rest_days'],
    }


def compute_avg_odds(bookmakers, home_team, away_team):
    h, d, a = [], [], []
    for bookie in bookmakers:
        for market in bookie.get('markets', []):
            if market.get('key') != 'h2h':
                continue
            by_name = {o['name']: o['price'] for o in market['outcomes']}
            hv = by_name.get(home_team)
            dv = next((v for k, v in by_name.items() if k.lower() == 'draw'), None)
            av = by_name.get(away_team)
            if hv: h.append(hv)
            if dv: d.append(dv)
            if av: a.append(av)
    if not h or not d or not a:
        return None, None, None
    return round(float(np.mean(h)), 3), round(float(np.mean(d)), 3), round(float(np.mean(a)), 3)


# --- Auth Routes ---
@app.route('/')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    email, password = request.form.get('email'), request.form.get('password')
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        flash('Invalid email or password.', 'error')
        return redirect(url_for('login'))
    login_user(user)
    return redirect(url_for('dashboard'))

@app.route('/register')
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_post():
    email, password = request.form.get('email'), request.form.get('password')
    if User.query.filter_by(email=email).first():
        flash('Email already registered.', 'error')
        return redirect(url_for('register'))
    db.session.add(User(email=email,
                        password_hash=generate_password_hash(password, method='pbkdf2:sha256')))
    db.session.commit()
    flash('Account created! Please log in.', 'info')
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# --- Dashboard ---
@app.route('/dashboard')
@login_required
def dashboard():
    global _odds_refreshed_at, _matches_cache
    upcoming_matches, api_error = [], None
    standings = fetch_standings()

    # Load previous odds snapshots for movement detection (most recent per match, >5 min old)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
    prev_snaps = OddsSnapshot.query.filter(OddsSnapshot.fetched_at < cutoff).all()
    prev_odds_map: dict = {}
    for s in prev_snaps:
        key = f"{s.home_team}||{s.away_team}||{s.match_date}"
        if key not in prev_odds_map or s.fetched_at > prev_odds_map[key]['ts']:
            prev_odds_map[key] = {'h': s.odds_h, 'd': s.odds_d, 'a': s.odds_a, 'ts': s.fetched_at}

    try:
        api_data = []
        for sport_key, _ in LEAGUES:
            try:
                resp = requests.get(
                    f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds',
                    params={'api_key': ODDS_API_KEY, 'regions': REGIONS,
                            'markets': 'h2h', 'oddsFormat': ODDS_FORMAT},
                    timeout=10,
                )
                resp.raise_for_status()
                api_data.extend(resp.json())
            except Exception as e:
                print(f"Could not fetch odds for {sport_key}: {e}")

        api_data.sort(key=lambda g: g.get('commence_time', ''))
        _odds_refreshed_at = datetime.now(timezone.utc)
        print(f"Fetched {len(api_data)} games across {len(LEAGUES)} leagues.")

        for game in api_data:
            home_team  = game.get('home_team')
            away_team  = game.get('away_team')
            bookmakers = game.get('bookmakers', [])
            if not home_team or not away_team or not bookmakers:
                continue

            avg_H, avg_D, avg_A = compute_avg_odds(bookmakers, home_team, away_team)
            if not all([avg_H, avg_D, avg_A]):
                continue

            if not _is_known_team(home_team) or not _is_known_team(away_team):
                continue

            hf = get_team_features(home_team, standings, is_home=True)
            af = get_team_features(away_team, standings, is_home=False)

            match_date_str = game.get('commence_time', '')[:10]
            snap_key = f"{home_team}||{away_team}||{match_date_str}"
            prev = prev_odds_map.get(snap_key)
            def _mv(curr, old): return 'up' if curr > old + 0.04 else ('down' if curr < old - 0.04 else 'flat')
            odds_move = {'h': _mv(avg_H, prev['h']), 'd': _mv(avg_D, prev['d']), 'a': _mv(avg_A, prev['a'])} \
                        if prev else {'h': 'flat', 'd': 'flat', 'a': 'flat'}
            db.session.add(OddsSnapshot(match_date=match_date_str, home_team=home_team, away_team=away_team,
                                        odds_h=avg_H, odds_d=avg_D, odds_a=avg_A))

            upcoming_matches.append({
                "league":        game.get('sport_title'),
                "date":          game.get('commence_time', '')[:10],
                "kickoff":       format_kickoff(game.get('commence_time', '')),
                "home_team":     home_team,
                "home_logo_url": get_logo_url(home_team),
                "away_team":     away_team,
                "away_logo_url": get_logo_url(away_team),
                "home_form": get_team_form(home_team),
                "away_form": get_team_form(away_team),
                "b365h": avg_H, "b365d": avg_D, "b365a": avg_A,
                "features": _build_features(avg_H, avg_D, avg_A, hf, af, home_team, away_team),
                "home_features": hf,
                "away_features": af,
                "odds_move": odds_move,
            })

        # Commit snapshots and purge those older than 7 days
        db.session.commit()
        OddsSnapshot.query.filter(
            OddsSnapshot.fetched_at < datetime.now(timezone.utc) - timedelta(days=7)
        ).delete()
        db.session.commit()

    except requests.exceptions.RequestException as e:
        api_error = f"Error fetching odds: {e}"
        print(api_error)

    # Pre-compute predictions + cache matches for detail page
    _matches_cache = {}
    if model and model_features:
        for m in upcoming_matches:
            try:
                feat_vec = [float(m['features'][f]) for f in model_features]
                probs    = model.predict_proba(np.array([feat_vec]))[0]
                ph, pd_, pa = float(probs[0]), float(probs[1]), float(probs[2])
                label        = ['H', 'D', 'A'][int(np.argmax(probs))]
                raw_mkt      = (1/m['b365h']) + (1/m['b365d']) + (1/m['b365a'])
                mkt_h        = round(100 / (m['b365h'] * raw_mkt))
                mkt_d        = round(100 / (m['b365d'] * raw_mkt))
                mkt_a        = round(100 / (m['b365a'] * raw_mkt))
                best         = max(ph, pd_, pa) * 100
                m['pred'] = {
                    'h': round(ph * 100), 'd': round(pd_ * 100), 'a': round(pa * 100),
                    'label': label, 'best_prob': round(best),
                    'mkt_h': mkt_h, 'mkt_d': mkt_d, 'mkt_a': mkt_a,
                    'ev_h':  round((ph  * m['b365h'] - 1) * 100),
                    'ev_d':  round((pd_ * m['b365d'] - 1) * 100),
                    'ev_a':  round((pa  * m['b365a'] - 1) * 100),
                    'val_h': (round(ph  * 100) - mkt_h) >= 5,
                    'val_d': (round(pd_ * 100) - mkt_d) >= 5,
                    'val_a': (round(pa  * 100) - mkt_a) >= 5,
                    'confidence': 'HIGH' if best >= 60 else ('MED' if best >= 45 else 'LOW'),
                }
            except Exception:
                m['pred'] = None

            _matches_cache[f"{m['home_team']}||{m['away_team']}"] = m

    # Auto-log predictions for current user (skip duplicates for same match_date+teams)
    if model and model_features:
        with app.app_context():
            for m in upcoming_matches:
                if not m.get('pred'):
                    continue
                exists = PredictionLog.query.filter_by(
                    user_id=current_user.id,
                    match_date=m['date'],
                    home_team=m['home_team'],
                    away_team=m['away_team'],
                ).first()
                if not exists:
                    db.session.add(PredictionLog(
                        user_id    = current_user.id,
                        match_date = m['date'],
                        home_team  = m['home_team'],
                        away_team  = m['away_team'],
                        league     = m['league'],
                        pred_label = m['pred']['label'],
                        pred_prob  = m['pred']['best_prob'],
                    ))
            db.session.commit()

    dates = sorted(set(m['date'] for m in upcoming_matches if m.get('date')))

    refreshed_str = _odds_refreshed_at.strftime('%H:%M UTC') if _odds_refreshed_at else None
    return render_template('index.html',
                           matches=upcoming_matches,
                           dates=dates,
                           error=api_error,
                           user_email=current_user.email,
                           model_accuracy=model_accuracy,
                           leagues_display=_leagues_display,
                           odds_refreshed_at=refreshed_str)


# --- Best Bets ---
@app.route('/best-bets')
@login_required
def best_bets():
    matches = list(_matches_cache.values())
    value_bets = []
    for m in matches:
        if not m.get('pred'):
            continue
        p = m['pred']
        for outcome, label in [('val_h', 'H'), ('val_d', 'D'), ('val_a', 'A')]:
            if p.get(outcome):
                ev_key = f"ev_{label.lower()}"
                odds_key = {'H': 'b365h', 'D': 'b365d', 'A': 'b365a'}[label]
                prob_key = {'H': 'h', 'D': 'd', 'A': 'a'}[label]
                value_bets.append({
                    'match':      m,
                    'outcome':    label,
                    'label_text': {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[label],
                    'odds':       m[odds_key],
                    'model_prob': p[prob_key],
                    'ev':         p[ev_key],
                })
    value_bets.sort(key=lambda x: x['ev'], reverse=True)
    settings = get_user_settings()
    for vb in value_bets:
        vb['kelly'] = kelly_stake(vb['model_prob'], vb['odds'], settings.bankroll, settings.kelly_fraction)
    return render_template('best_bets.html',
                           value_bets=value_bets,
                           user_email=current_user.email,
                           model_accuracy=model_accuracy,
                           settings=settings)


# --- Match Detail ---
@app.route('/match/<path:home>/<path:away>')
@login_required
def match_detail(home, away):
    m = _matches_cache.get(f"{home}||{away}")
    if not m:
        flash('Match not found. Return to dashboard to refresh.', 'info')
        return redirect(url_for('dashboard'))

    # Build H2H history from cache
    home_dn = normalise(home)
    away_dn = normalise(away)
    h2h_key = f"{home_dn}||{away_dn}"
    h2h_data = h2h_cache.get(h2h_key, {})

    return render_template('match_detail.html',
                           match=m,
                           h2h=h2h_data,
                           user_email=current_user.email,
                           model_accuracy=model_accuracy)


# --- Betting Tracker ---
@app.route('/bets')
@login_required
def bets():
    user_bets = Bet.query.filter_by(user_id=current_user.id).order_by(Bet.created_at.desc()).all()
    settled   = [b for b in user_bets if b.status != 'P']
    pending   = [b for b in user_bets if b.status == 'P']

    total_staked = sum(b.stake for b in settled)
    total_pnl    = sum(b.pnl for b in settled)
    wins         = sum(1 for b in settled if b.status == 'W')
    hit_rate     = round(wins / len(settled) * 100) if settled else 0
    roi          = round(total_pnl / total_staked * 100, 1) if total_staked else 0

    # Pass available matches from cache for the add-bet form
    available_matches = list(_matches_cache.values())

    return render_template('bets.html',
                           bets=user_bets,
                           pending=pending,
                           settled=settled,
                           total_staked=round(total_staked, 2),
                           total_pnl=round(total_pnl, 2),
                           wins=wins,
                           hit_rate=hit_rate,
                           roi=roi,
                           available_matches=available_matches,
                           user_email=current_user.email,
                           model_accuracy=model_accuracy)


@app.route('/bets/add', methods=['POST'])
@login_required
def bets_add():
    try:
        bet = Bet(
            user_id    = current_user.id,
            match_date = request.form.get('match_date', ''),
            home_team  = request.form.get('home_team', ''),
            away_team  = request.form.get('away_team', ''),
            league     = request.form.get('league', ''),
            bet_on     = request.form.get('bet_on', 'H'),
            odds       = float(request.form.get('odds', 2.0)),
            stake      = float(request.form.get('stake', 10.0)),
        )
        db.session.add(bet)
        db.session.commit()
        flash('Bet logged.', 'info')
    except Exception as e:
        flash(f'Error logging bet: {e}', 'error')
    return redirect(url_for('bets'))


@app.route('/bets/<int:bet_id>/settle', methods=['POST'])
@login_required
def bets_settle(bet_id):
    bet = Bet.query.filter_by(id=bet_id, user_id=current_user.id).first_or_404()
    result = request.form.get('result', 'V')
    bet.status = result
    if result == 'W':
        bet.pnl = round(bet.stake * bet.odds - bet.stake, 2)
    elif result == 'L':
        bet.pnl = -bet.stake
    else:
        bet.pnl = 0.0
    db.session.commit()
    flash(f'Bet settled as {"Win" if result == "W" else "Loss" if result == "L" else "Void"}.', 'info')
    return redirect(url_for('bets'))


@app.route('/bets/<int:bet_id>/delete', methods=['POST'])
@login_required
def bets_delete(bet_id):
    bet = Bet.query.filter_by(id=bet_id, user_id=current_user.id).first_or_404()
    db.session.delete(bet)
    db.session.commit()
    flash('Bet deleted.', 'info')
    return redirect(url_for('bets'))


# --- Predictions Tracker ---
@app.route('/predictions')
@login_required
def predictions():
    logs = PredictionLog.query.filter_by(user_id=current_user.id)\
                              .order_by(PredictionLog.logged_at.desc()).all()
    settled = [p for p in logs if p.actual is not None]
    correct = sum(1 for p in settled if p.pred_label == p.actual)
    accuracy = round(correct / len(settled) * 100, 1) if settled else None

    return render_template('predictions.html',
                           logs=logs,
                           settled=settled,
                           correct=correct,
                           accuracy=accuracy,
                           user_email=current_user.email,
                           model_accuracy=model_accuracy)


@app.route('/predictions/<int:log_id>/settle', methods=['POST'])
@login_required
def predictions_settle(log_id):
    log = PredictionLog.query.filter_by(id=log_id, user_id=current_user.id).first_or_404()
    log.actual = request.form.get('actual', '')
    db.session.commit()
    flash('Result recorded.', 'info')
    return redirect(url_for('predictions'))


# --- API: value bets (polled by browser notification JS) ---
@app.route('/api/value-bets')
@login_required
def api_value_bets():
    bets = []
    for m in _matches_cache.values():
        p = m.get('pred')
        if not p:
            continue
        for outcome, label in [('val_h', 'H'), ('val_d', 'D'), ('val_a', 'A')]:
            if p.get(outcome):
                odds_map = {'H': 'b365h', 'D': 'b365d', 'A': 'b365a'}
                prob_map = {'H': 'h',     'D': 'd',     'A': 'a'}
                ev_map   = {'H': 'ev_h',  'D': 'ev_d',  'A': 'ev_a'}
                bets.append({
                    'match':      f"{m['home_team']} vs {m['away_team']}",
                    'league':     m['league'],
                    'outcome':    {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[label],
                    'odds':       m[odds_map[label]],
                    'model_prob': p[prob_map[label]],
                    'ev':         p[ev_map[label]],
                })
    bets.sort(key=lambda x: x['ev'], reverse=True)
    return jsonify(bets)


# --- Reload model from disk (call after overnight training completes) ---
@app.route('/reload-model', methods=['POST'])
@login_required
def reload_model():
    global model, model_features, model_accuracy
    try:
        _md = joblib.load('football_model.joblib')
        model          = _md['model']
        model_features = _md['features']
        with open('accuracy.txt') as f:
            model_accuracy = f.read().strip()
        print(f"Model hot-reloaded — accuracy: {model_accuracy}%")
        return jsonify({'status': 'ok', 'accuracy': model_accuracy})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# --- Predict (AJAX fallback) ---
@app.route('/predict', methods=['POST'])
@login_required
def predict_match():
    if not model:
        return jsonify({'error': 'Model not loaded.'}), 500
    try:
        features = [float(request.form[f]) for f in model_features]
        probs    = model.predict_proba(np.array([features]))[0]
        return jsonify({
            'home_win_prob': round(probs[0] * 100),
            'draw_prob':     round(probs[1] * 100),
            'away_win_prob': round(probs[2] * 100),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- Analytics ---
@app.route('/analytics')
@login_required
def analytics():
    from collections import defaultdict
    logs  = PredictionLog.query.filter_by(user_id=current_user.id).order_by(PredictionLog.logged_at).all()
    bets  = Bet.query.filter_by(user_id=current_user.id).filter(Bet.status != 'P').order_by(Bet.created_at).all()
    settled = [l for l in logs if l.actual is not None]

    # Accuracy over time (weekly)
    weekly: dict = defaultdict(lambda: {'c': 0, 't': 0})
    for l in settled:
        w = l.logged_at.strftime('%Y-W%V')
        weekly[w]['t'] += 1
        if l.pred_label == l.actual: weekly[w]['c'] += 1
    weeks     = sorted(weekly)
    weekly_acc = [round(weekly[w]['c']/weekly[w]['t']*100,1) if weekly[w]['t'] else 0 for w in weeks]

    # By league
    by_league: dict = defaultdict(lambda: {'c': 0, 't': 0})
    for l in settled:
        lg = (l.league or 'Unknown').split(' –')[0]
        by_league[lg]['t'] += 1
        if l.pred_label == l.actual: by_league[lg]['c'] += 1
    lg_names  = sorted(by_league)
    lg_acc    = [round(by_league[lg]['c']/by_league[lg]['t']*100,1) if by_league[lg]['t'] else 0 for lg in lg_names]
    lg_counts = [by_league[lg]['t'] for lg in lg_names]

    # By confidence level
    conf: dict = {'HIGH': {'c':0,'t':0}, 'MED': {'c':0,'t':0}, 'LOW': {'c':0,'t':0}}
    for l in settled:
        lv = 'HIGH' if l.pred_prob >= 60 else ('MED' if l.pred_prob >= 45 else 'LOW')
        conf[lv]['t'] += 1
        if l.pred_label == l.actual: conf[lv]['c'] += 1
    conf_labels = ['HIGH (≥60%)', 'MED (45–60%)', 'LOW (<45%)']
    conf_acc    = [round(conf[k]['c']/conf[k]['t']*100,1) if conf[k]['t'] else 0 for k in ('HIGH','MED','LOW')]
    conf_counts = [conf[k]['t'] for k in ('HIGH','MED','LOW')]

    # Calibration (10-pp buckets)
    cal: dict = defaultdict(lambda: {'c': 0, 't': 0})
    for l in settled:
        b = (l.pred_prob // 10) * 10
        cal[b]['t'] += 1
        if l.pred_label == l.actual: cal[b]['c'] += 1
    cal_keys    = sorted(cal)
    cal_labels  = [f"{b}–{b+10}%" for b in cal_keys]
    cal_pred    = [b + 5 for b in cal_keys]
    cal_actual  = [round(cal[b]['c']/cal[b]['t']*100,1) if cal[b]['t'] else 0 for b in cal_keys]

    # ROI curve
    roi_labels, roi_vals = [], []
    cumulative = 0.0
    for bet in bets:
        cumulative += bet.pnl
        roi_labels.append(bet.match_date or '')
        roi_vals.append(round(cumulative, 2))

    total_acc = round(sum(1 for l in settled if l.pred_label==l.actual)/len(settled)*100,1) if settled else None
    total_pnl = round(sum(b.pnl for b in bets), 2)
    settings  = get_user_settings()

    return render_template('analytics.html',
        user_email=current_user.email, model_accuracy=model_accuracy,
        total_predictions=len(logs), settled_count=len(settled), total_acc=total_acc,
        total_bets=len(bets), total_pnl=total_pnl,
        weeks=json.dumps(weeks), weekly_acc=json.dumps(weekly_acc),
        lg_names=json.dumps(lg_names), lg_acc=json.dumps(lg_acc), lg_counts=json.dumps(lg_counts),
        conf_labels=json.dumps(conf_labels), conf_acc=json.dumps(conf_acc), conf_counts=json.dumps(conf_counts),
        cal_labels=json.dumps(cal_labels), cal_pred=json.dumps(cal_pred), cal_actual=json.dumps(cal_actual),
        roi_labels=json.dumps(roi_labels), roi_vals=json.dumps(roi_vals),
    )


# --- Settings ---
@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    s = get_user_settings()
    if request.method == 'POST':
        try:
            s.bankroll       = float(request.form.get('bankroll', 1000))
            s.default_stake  = float(request.form.get('default_stake', 10))
            s.kelly_fraction = float(request.form.get('kelly_fraction', 0.25))
            s.notify_email   = 'notify_email'   in request.form
            s.notify_browser = 'notify_browser' in request.form
            db.session.commit()
            flash('Settings saved.', 'info')
        except Exception as e:
            flash(f'Error saving settings: {e}', 'error')
        return redirect(url_for('settings'))
    return render_template('settings.html',
        s=s, user_email=current_user.email, model_accuracy=model_accuracy)


# --- Export ---
@app.route('/bets/export')
@login_required
def bets_export():
    bets = Bet.query.filter_by(user_id=current_user.id).order_by(Bet.created_at.desc()).all()
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(['Date','Home','Away','League','Bet On','Odds','Stake','Status','P&L','Logged At'])
    for b in bets:
        w.writerow([b.match_date, b.home_team, b.away_team, b.league,
                    {'H':'Home Win','D':'Draw','A':'Away Win'}.get(b.bet_on,''),
                    b.odds, b.stake, {'P':'Pending','W':'Win','L':'Loss','V':'Void'}.get(b.status,''),
                    b.pnl, b.created_at.strftime('%Y-%m-%d %H:%M')])
    from flask import Response
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=bets.csv'})


@app.route('/predictions/export')
@login_required
def predictions_export():
    logs = PredictionLog.query.filter_by(user_id=current_user.id).order_by(PredictionLog.logged_at.desc()).all()
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(['Date','Home','Away','League','Prediction','Confidence %','Actual','Correct','Logged At'])
    for l in logs:
        correct = '' if l.actual is None else ('Yes' if l.pred_label == l.actual else 'No')
        w.writerow([l.match_date, l.home_team, l.away_team, l.league,
                    {'H':'Home Win','D':'Draw','A':'Away Win'}.get(l.pred_label,''),
                    l.pred_prob, {'H':'Home Win','D':'Draw','A':'Away Win'}.get(l.actual,'') if l.actual else '',
                    correct, l.logged_at.strftime('%Y-%m-%d %H:%M')])
    from flask import Response
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=predictions.csv'})


# --- Poisson API ---
@app.route('/api/poisson/<path:home>/<path:away>')
@login_required
def api_poisson(home, away):
    return jsonify(poisson_predict(home, away))


# --- Auto-settle trigger (manual) ---
@app.route('/settle-now', methods=['POST'])
@login_required
def settle_now():
    threading.Thread(target=auto_settle, daemon=True).start()
    return jsonify({'status': 'started'})


# --- Main ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
