#!/usr/bin/env python3
"""
BlobMaster Training Dashboard - Web UI

Single-file web dashboard for monitoring and controlling training.
No extra dependencies beyond Python stdlib + Chart.js (CDN).

Usage:
    python ml/dashboard.py                  # Start on port 8000
    python ml/dashboard.py --port 9000      # Custom port

Then open http://localhost:8000 in your browser.

Features:
    - Live training metrics (iteration, ELO, loss, ETA)
    - ELO progression chart
    - Loss progression chart (policy + value)
    - Start / Pause / Resume training
    - Training log viewer
    - Works independently of browser (close & reopen anytime)
"""

import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
STATUS_FILE = PROJECT_ROOT / "models" / "checkpoints" / "status.json"
CONTROL_FILE = PROJECT_ROOT / "models" / "checkpoints" / "control.signal"
HISTORY_FILE = PROJECT_ROOT / "models" / "checkpoints" / "dashboard_history.json"
PID_FILE = PROJECT_ROOT / "models" / "checkpoints" / "training.pid"
METRICS_FILE = PROJECT_ROOT / "models" / "checkpoints" / "metrics_history.json"
OUTPUT_LOG = PROJECT_ROOT / "runs" / "training_output.log"

# --- Curriculum stage definitions (must match ml/config.py mcts_schedule) ---
CURRICULUM_STAGES = [
    {"stage": 1, "start": 1,   "end": 50,  "mcts": "1x15",  "label": "Stage 1"},
    {"stage": 2, "start": 51,  "end": 150, "mcts": "2x25",  "label": "Stage 2"},
    {"stage": 3, "start": 151, "end": 300, "mcts": "3x35",  "label": "Stage 3"},
    {"stage": 4, "start": 301, "end": 450, "mcts": "4x45",  "label": "Stage 4"},
    {"stage": 5, "start": 451, "end": 500, "mcts": "5x50",  "label": "Stage 5"},
]

def _get_stage_info(iteration):
    """Return current stage dict and progress-within-stage for a given iteration."""
    for s in CURRICULUM_STAGES:
        if iteration <= s["end"]:
            within = iteration - s["start"] + 1
            total = s["end"] - s["start"] + 1
            return s, within, total
    # Past last stage
    last = CURRICULUM_STAGES[-1]
    return last, last["end"] - last["start"] + 1, last["end"] - last["start"] + 1

# --- HTML Template ---
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BlobMaster Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{background:#1e293b;padding:14px 24px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #334155}
.header h1{font-size:18px;font-weight:600}
.header-right{display:flex;align-items:center;gap:16px}
.conn{display:flex;align-items:center;gap:6px;font-size:13px;color:#94a3b8}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot.on{background:#22c55e}.dot.off{background:#ef4444}
.container{max-width:1200px;margin:0 auto;padding:20px}

/* Progress section with stage pipeline */
.progress-section{background:#1e293b;border-radius:10px;padding:16px 20px;margin-bottom:16px}
.progress-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.progress-left{display:flex;align-items:center;gap:12px}
.badge{padding:3px 10px;border-radius:5px;font-weight:600;font-size:12px;text-transform:uppercase}
.elapsed-text{color:#94a3b8;font-size:13px}

/* Overall progress bar */
.bar-bg{background:#334155;border-radius:6px;height:24px;overflow:hidden;position:relative;margin-bottom:12px}
.bar-fill{background:linear-gradient(90deg,#3b82f6,#8b5cf6);height:100%;border-radius:6px;transition:width .5s ease}
.bar-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-weight:600;font-size:12px;text-shadow:0 1px 2px rgba(0,0,0,.5)}

/* Stage pipeline */
.stage-pipeline{display:flex;gap:6px;margin-bottom:4px}
.stage-block{flex:1;background:#1e293b;border:1px solid #334155;border-radius:6px;padding:8px 6px;text-align:center;position:relative;transition:all .3s}
.stage-block.completed{background:#1e3a2f;border-color:#22c55e55}
.stage-block.active{background:#1e293b;border-color:#3b82f6;box-shadow:0 0 12px rgba(59,130,246,.25)}
.stage-block.future{opacity:.5}
.stage-name{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.stage-mcts{font-size:13px;font-weight:700;color:#e2e8f0}
.stage-iters{font-size:10px;color:#64748b;margin-top:2px}
.stage-bar-bg{background:#334155;border-radius:3px;height:4px;margin-top:5px;overflow:hidden}
.stage-bar-fill{height:100%;border-radius:3px;transition:width .5s ease}
.stage-block.completed .stage-bar-fill{background:#22c55e;width:100%}
.stage-block.active .stage-bar-fill{background:#3b82f6}
.stage-block.future .stage-bar-fill{width:0}

/* Metric cards */
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.card{background:#1e293b;border-radius:10px;padding:14px;text-align:center}
.card-label{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.card-value{font-size:22px;font-weight:700}
.card-sub{font-size:11px;color:#64748b;margin-top:1px}
.pos{color:#22c55e}.neg{color:#ef4444}

/* Controls */
.controls{display:flex;gap:10px;margin-bottom:16px}
button{padding:9px 22px;border:none;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s}
button:disabled{opacity:.35;cursor:not-allowed}
.btn-start{background:#22c55e;color:#0f172a}.btn-start:hover:not(:disabled){background:#16a34a}
.btn-pause{background:#f59e0b;color:#0f172a}.btn-pause:hover:not(:disabled){background:#d97706}
.btn-folder{background:#6366f1;color:#fff}.btn-folder:hover:not(:disabled){background:#4f46e5}

/* Charts */
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.chart-card{background:#1e293b;border-radius:10px;padding:16px}
.chart-card h3{font-size:13px;color:#94a3b8;margin-bottom:10px}

/* Log */
.log-section{background:#1e293b;border-radius:10px;padding:16px}
.log-section h3{font-size:13px;color:#94a3b8;margin-bottom:10px}
.log-box{background:#0f172a;border-radius:7px;padding:10px;font-family:'JetBrains Mono','Fira Code',monospace;font-size:11px;line-height:1.6;max-height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:#94a3b8}

@media(max-width:768px){.charts{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,1fr)}.stage-pipeline{flex-wrap:wrap}}
</style>
</head>
<body>
<div class="header">
  <h1>BlobMaster Training Dashboard</h1>
  <div class="header-right">
    <div class="conn"><span class="dot on" id="dot"></span><span id="conn-text">Connected</span></div>
  </div>
</div>
<div class="container">
  <!-- Progress & Stage Pipeline -->
  <div class="progress-section">
    <div class="progress-top">
      <div class="progress-left">
        <span class="badge" id="badge" style="background:#64748b;color:#0f172a">IDLE</span>
        <span style="font-size:14px;font-weight:600" id="iter-text">-</span>
      </div>
      <span class="elapsed-text" id="elapsed">-</span>
    </div>
    <div class="bar-bg"><div class="bar-fill" id="bar" style="width:0"></div><span class="bar-text" id="bar-text">-</span></div>
    <div class="stage-pipeline" id="stage-pipeline">
      <div class="stage-block future"><div class="stage-name">Stage 1</div><div class="stage-mcts">1x15</div><div class="stage-iters">iter 1-50</div><div class="stage-bar-bg"><div class="stage-bar-fill"></div></div></div>
      <div class="stage-block future"><div class="stage-name">Stage 2</div><div class="stage-mcts">2x25</div><div class="stage-iters">iter 51-150</div><div class="stage-bar-bg"><div class="stage-bar-fill"></div></div></div>
      <div class="stage-block future"><div class="stage-name">Stage 3</div><div class="stage-mcts">3x35</div><div class="stage-iters">iter 151-300</div><div class="stage-bar-bg"><div class="stage-bar-fill"></div></div></div>
      <div class="stage-block future"><div class="stage-name">Stage 4</div><div class="stage-mcts">4x45</div><div class="stage-iters">iter 301-450</div><div class="stage-bar-bg"><div class="stage-bar-fill"></div></div></div>
      <div class="stage-block future"><div class="stage-name">Stage 5</div><div class="stage-mcts">5x50</div><div class="stage-iters">iter 451-500</div><div class="stage-bar-bg"><div class="stage-bar-fill"></div></div></div>
    </div>
  </div>

  <!-- Metric Cards -->
  <div class="grid">
    <div class="card"><div class="card-label">ELO Rating</div><div class="card-value" id="v-elo">-</div><div class="card-sub" id="v-elo-d">-</div></div>
    <div class="card"><div class="card-label">ETA</div><div class="card-value" id="v-eta">-</div><div class="card-sub" id="v-eta-sub">-</div></div>
    <div class="card"><div class="card-label">Learning Rate</div><div class="card-value" id="v-lr">-</div></div>
    <div class="card"><div class="card-label">Rounds / Iteration</div><div class="card-value" id="v-units">-</div></div>
  </div>
  <div class="grid" style="grid-template-columns:repeat(3,1fr)">
    <div class="card"><div class="card-label">Total Loss</div><div class="card-value" id="v-loss">-</div></div>
    <div class="card"><div class="card-label">Policy Loss</div><div class="card-value" id="v-ploss">-</div></div>
    <div class="card"><div class="card-label">Value Loss</div><div class="card-value" id="v-vloss">-</div></div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <button class="btn-start" id="btn-start" onclick="doStart()">Start Training</button>
    <button class="btn-pause" id="btn-pause" onclick="doPause()" disabled>Pause</button>
    <button class="btn-folder" onclick="doOpenCheckpoints()">Open Checkpoints</button>
  </div>

  <!-- Charts -->
  <div class="charts">
    <div class="chart-card"><h3>ELO Progression</h3><canvas id="eloC"></canvas></div>
    <div class="chart-card"><h3>Training Loss</h3><canvas id="lossC"></canvas></div>
  </div>

  <!-- Log -->
  <div class="log-section">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <h3 style="margin:0">Training Log (last 100 lines)</h3>
      <button style="padding:4px 12px;background:#334155;color:#94a3b8;border-radius:5px;font-size:11px" onclick="doClearLog()">Clear</button>
    </div>
    <div class="log-box" id="log">Waiting for training output...</div>
  </div>
</div>
<script>
// Curriculum stages definition (mirrors ml/config.py mcts_schedule)
const STAGES=[
  {stage:1,start:1,end:50,mcts:'1\u00d715'},
  {stage:2,start:51,end:150,mcts:'2\u00d725'},
  {stage:3,start:151,end:300,mcts:'3\u00d735'},
  {stage:4,start:301,end:450,mcts:'4\u00d745'},
  {stage:5,start:451,end:500,mcts:'5\u00d750'},
];

function getStageForIter(it){
  for(const s of STAGES){if(it<=s.end)return s;}
  return STAGES[STAGES.length-1];
}

function fmtHM(hours){
  if(hours==null||isNaN(hours))return '-';
  const h=Math.floor(hours);
  const m=Math.round((hours-h)*60);
  if(h>=24){const d=Math.floor(h/24);const rh=h%24;return d+'d '+rh+'h '+m+'m';}
  if(h>0)return h+'h '+m+'m';
  return m+'m';
}

let eloChart,lossChart;

function init(){
  // ELO chart
  eloChart=new Chart(document.getElementById('eloC'),{type:'line',
    data:{labels:[],datasets:[{label:'ELO',data:[],borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,.1)',fill:true,tension:.3,pointRadius:2}]},
    options:{responsive:true,animation:false,
      scales:{x:{title:{display:true,text:'Iteration',color:'#64748b'},grid:{color:'#1e293b22'},ticks:{color:'#64748b'}},
              y:{title:{display:true,text:'ELO',color:'#64748b'},grid:{color:'#1e293b'},ticks:{color:'#64748b'}}},
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}}}});

  // Loss chart with dual y-axes
  lossChart=new Chart(document.getElementById('lossC'),{type:'line',
    data:{labels:[],datasets:[
      {label:'Total Loss',data:[],borderColor:'#f59e0b',tension:.3,pointRadius:1,yAxisID:'yLeft'},
      {label:'Policy Loss',data:[],borderColor:'#ef4444',tension:.3,pointRadius:1,yAxisID:'yLeft'},
      {label:'Value Loss',data:[],borderColor:'#22c55e',tension:.3,pointRadius:1,yAxisID:'yRight'}]},
    options:{responsive:true,animation:false,
      scales:{
        x:{title:{display:true,text:'Iteration',color:'#64748b'},grid:{color:'#1e293b22'},ticks:{color:'#64748b'}},
        yLeft:{type:'linear',position:'left',title:{display:true,text:'Total / Policy Loss',color:'#64748b'},grid:{color:'#1e293b'},ticks:{color:'#f59e0b'}},
        yRight:{type:'linear',position:'right',title:{display:true,text:'Value Loss',color:'#64748b'},grid:{drawOnChartArea:false},ticks:{color:'#22c55e'}}
      },
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}}}});
}

async function poll(){
  try{
    const r=await fetch('/api/status');const d=await r.json();
    upStatus(d);
    document.getElementById('dot').className='dot on';
    document.getElementById('conn-text').textContent='Connected';
  }catch{
    document.getElementById('dot').className='dot off';
    document.getElementById('conn-text').textContent='Disconnected';
  }
}

async function pollHistory(){
  try{const r=await fetch('/api/history');const h=await r.json();upCharts(h);}catch{}
}

async function pollLog(){
  try{
    const r=await fetch('/api/log');const d=await r.json();
    const el=document.getElementById('log');
    el.textContent=d.lines.join('');
    el.scrollTop=el.scrollHeight;
  }catch{}
}

function upStatus(d){
  const s=d.status,st=d.state;
  const colors={idle:'#64748b',running:'#22c55e',paused:'#f59e0b',completed:'#3b82f6'};
  const b=document.getElementById('badge');
  b.textContent=st.toUpperCase();b.style.background=colors[st]||'#64748b';
  document.getElementById('btn-start').disabled=(st==='running');
  document.getElementById('btn-start').textContent=st==='paused'?'Resume Training':st==='completed'?'Restart Training':'Start Training';
  document.getElementById('btn-pause').disabled=(st!=='running');
  if(!s)return;

  const it=s.iteration||0;
  const total=s.total_iterations||500;
  const pct=(s.progress*100).toFixed(1);

  // Overall progress bar
  document.getElementById('bar').style.width=pct+'%';
  document.getElementById('bar-text').textContent=it+'/'+total+' ('+pct+'%)';

  // Iteration text
  document.getElementById('iter-text').textContent='Iteration '+it+' / '+total;

  // Elapsed time in hours and minutes
  const h=s.elapsed_hours||0;
  document.getElementById('elapsed').textContent='Elapsed: '+fmtHM(h);

  // Stage pipeline update
  const curStage=getStageForIter(it);
  const blocks=document.getElementById('stage-pipeline').children;
  for(let i=0;i<STAGES.length;i++){
    const sg=STAGES[i];const bl=blocks[i];
    bl.className='stage-block';
    if(it>sg.end){
      bl.classList.add('completed');
    }else if(sg.stage===curStage.stage){
      bl.classList.add('active');
      const within=it-sg.start+1;
      const stTotal=sg.end-sg.start+1;
      const fill=bl.querySelector('.stage-bar-fill');
      fill.style.width=Math.min(100,Math.round(within/stTotal*100))+'%';
      // Update stage iter text to show progress
      bl.querySelector('.stage-iters').textContent=within+'/'+stTotal+' (iter '+sg.start+'-'+sg.end+')';
    }else{
      bl.classList.add('future');
    }
  }

  // ELO
  if(s.elo!=null){
    document.getElementById('v-elo').textContent=Math.round(s.elo);
    if(s.elo_change!=null){
      const sign=s.elo_change>=0?'+':'';
      const cls=s.elo_change>=0?'pos':'neg';
      document.getElementById('v-elo-d').innerHTML='<span class="'+cls+'">'+sign+Math.round(s.elo_change)+'</span>';
    }
  }

  // ETA
  const etaH=s.eta_hours||0;
  document.getElementById('v-eta').textContent=fmtHM(etaH);
  if(s.eta_days!=null&&s.eta_days>=1){
    document.getElementById('v-eta-sub').textContent='~'+s.eta_days.toFixed(1)+' days';
  }else{
    document.getElementById('v-eta-sub').textContent='';
  }

  // LR, losses, units
  document.getElementById('v-lr').textContent=s.learning_rate?s.learning_rate.toFixed(6):'-';
  document.getElementById('v-loss').textContent=s.loss?s.loss.toFixed(4):'-';
  document.getElementById('v-ploss').textContent=s.policy_loss?s.policy_loss.toFixed(4):'-';
  document.getElementById('v-vloss').textContent=s.value_loss?s.value_loss.toFixed(4):'-';
  document.getElementById('v-units').textContent=s.training_units_generated||'-';
}

function upCharts(hist){
  if(!hist.length)return;
  const eD=hist.filter(h=>h.elo!=null);
  eloChart.data.labels=eD.map(h=>h.iteration);
  eloChart.data.datasets[0].data=eD.map(h=>h.elo);
  eloChart.update('none');
  const lD=hist.filter(h=>h.loss!=null);
  lossChart.data.labels=lD.map(h=>h.iteration);
  lossChart.data.datasets[0].data=lD.map(h=>h.loss);
  lossChart.data.datasets[1].data=lD.map(h=>h.policy_loss);
  lossChart.data.datasets[2].data=lD.map(h=>h.value_loss);
  lossChart.update('none');
}

async function doStart(){
  document.getElementById('btn-start').disabled=true;
  document.getElementById('btn-start').textContent='Starting...';
  try{
    const r=await fetch('/api/start',{method:'POST'});const d=await r.json();
    if(!d.ok)alert('Failed: '+d.error);
  }catch(e){alert('Error: '+e.message)}
  setTimeout(poll,2000);
}

async function doPause(){
  document.getElementById('btn-pause').disabled=true;
  try{await fetch('/api/pause',{method:'POST'});}catch{}
  setTimeout(poll,2000);
}

async function doOpenCheckpoints(){
  try{await fetch('/api/open-checkpoints',{method:'POST'});}catch{}
}

async function doClearLog(){
  try{await fetch('/api/clear-log',{method:'POST'});}catch{}
  document.getElementById('log').textContent='Log cleared.';
}

document.addEventListener('DOMContentLoaded',()=>{
  init();poll();pollHistory();pollLog();
  setInterval(poll,5000);
  setInterval(pollHistory,15000);
  setInterval(pollLog,10000);
});
</script>
</body>
</html>"""


# --- Training Process Manager ---

class TrainingManager:
    """Manages the training subprocess lifecycle."""

    def __init__(self):
        self.process = None
        self._tracked_pid = None
        self._check_existing()

    def _check_existing(self):
        """Detect training already running from a previous session."""
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text().strip())
                if self._pid_alive(pid):
                    self._tracked_pid = pid
                    return
            except (ValueError, OSError):
                pass
        self._tracked_pid = None

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def is_running(self):
        if self.process and self.process.poll() is None:
            return True
        if self._tracked_pid and self._pid_alive(self._tracked_pid):
            return True
        return False

    def get_state(self):
        if self.is_running():
            return "running"
        if STATUS_FILE.exists():
            try:
                s = json.loads(STATUS_FILE.read_text())
                if s.get("progress", 0) >= 1.0:
                    return "completed"
                return "paused"
            except (json.JSONDecodeError, OSError):
                pass
        return "idle"

    def start(self):
        if self.is_running():
            return {"ok": False, "error": "Training already running"}

        checkpoint = self._find_latest_checkpoint()

        cmd = [
            sys.executable, "ml/train.py",
            "--training-on", "rounds",
            "--enable-curriculum",
            "--iterations", "500",
            "--device", "cuda",
            "--workers", "32",
        ]
        if checkpoint:
            cmd.extend(["--resume", str(checkpoint)])

        # Ensure output directory exists
        OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)

        log_file = open(OUTPUT_LOG, "a")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()  # child inherited the fd

        self._tracked_pid = self.process.pid
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(self.process.pid))

        return {"ok": True, "pid": self.process.pid, "resumed": checkpoint is not None}

    def pause(self):
        CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONTROL_FILE.write_text("PAUSE\n")
        return {"ok": True}

    @staticmethod
    def _find_latest_checkpoint():
        search_dirs = [
            PROJECT_ROOT / "models" / "checkpoints" / "cache",
            PROJECT_ROOT / "models" / "checkpoints" / "permanent",
            PROJECT_ROOT / "models" / "checkpoints",
        ]
        candidates = []
        for d in search_dirs:
            if d.exists():
                candidates.extend(d.glob("*.pth"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)


# --- History Tracker (background thread) ---

class HistoryTracker(threading.Thread):
    """Accumulates training history from status.json snapshots."""

    def __init__(self):
        super().__init__(daemon=True)
        self.history = []
        self.seen = set()
        self.lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        """Load history from dashboard file and/or trainer's metrics file."""
        # 1) Load dashboard's own history
        if HISTORY_FILE.exists():
            try:
                self.history = json.loads(HISTORY_FILE.read_text())
                self.seen = {h.get("iteration") for h in self.history}
            except (json.JSONDecodeError, OSError):
                pass

        # 2) Merge from trainer's metrics_history.json (fills gaps)
        if METRICS_FILE.exists():
            try:
                metrics = json.loads(METRICS_FILE.read_text())
                for m in metrics:
                    it = m.get("iteration")
                    if it is None:
                        continue
                    display_it = it + 1  # metrics_history is 0-indexed
                    if display_it in self.seen:
                        continue
                    entry = {
                        "iteration": display_it,
                        "elo": m.get("current_elo"),
                        "elo_change": m.get("elo_change"),
                        "loss": m.get("avg_total_loss"),
                        "policy_loss": m.get("avg_policy_loss"),
                        "value_loss": m.get("avg_value_loss"),
                        "learning_rate": m.get("learning_rate"),
                        "mcts_config": m.get("mcts_config"),
                        "progress": m.get("progress"),
                        "elapsed_hours": m.get("iteration_time_minutes", 0) / 60,
                    }
                    self.history.append(entry)
                    self.seen.add(display_it)
                self.history.sort(key=lambda h: h.get("iteration", 0))
            except (json.JSONDecodeError, OSError):
                pass

    def run(self):
        while True:
            try:
                if STATUS_FILE.exists():
                    status = json.loads(STATUS_FILE.read_text())
                    it = status.get("iteration")
                    if it is not None and it not in self.seen:
                        with self.lock:
                            self.history.append(status)
                            self.seen.add(it)
                            self._save()
            except (json.JSONDecodeError, OSError):
                pass
            time.sleep(5)

    def _save(self):
        try:
            tmp = HISTORY_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.history, indent=1))
            tmp.replace(HISTORY_FILE)
        except OSError:
            pass

    def get_history(self):
        with self.lock:
            return list(self.history)


# --- HTTP Handler ---

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    manager: TrainingManager
    tracker: HistoryTracker

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html", HTML_TEMPLATE.encode())
        elif self.path == "/api/status":
            self._api_status()
        elif self.path == "/api/history":
            self._api_history()
        elif self.path == "/api/log":
            self._api_log()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/start":
            result = self.manager.start()
            self._json(result)
        elif self.path == "/api/pause":
            result = self.manager.pause()
            self._json(result)
        elif self.path == "/api/open-checkpoints":
            self._api_open_checkpoints()
        elif self.path == "/api/clear-log":
            self._api_clear_log()
        else:
            self.send_error(404)

    def _api_status(self):
        status = None
        if STATUS_FILE.exists():
            try:
                status = json.loads(STATUS_FILE.read_text())
                # Enrich with stage info
                it = status.get("iteration", 0)
                stage, within, total = _get_stage_info(it)
                status["stage_num"] = stage["stage"]
                status["stage_iter"] = within
                status["stage_total"] = total
                status["stage_label"] = stage["label"]
            except (json.JSONDecodeError, OSError):
                pass
        self._json({"status": status, "state": self.manager.get_state()})

    def _api_history(self):
        self._json(self.tracker.get_history())

    def _api_open_checkpoints(self):
        ckpt_dir = str(PROJECT_ROOT / "models" / "checkpoints")
        try:
            subprocess.Popen(["xdg-open", ckpt_dir])
            self._json({"ok": True})
        except OSError as e:
            self._json({"ok": False, "error": str(e)})

    def _api_clear_log(self):
        try:
            if OUTPUT_LOG.exists():
                OUTPUT_LOG.write_text("")
            self._json({"ok": True})
        except OSError as e:
            self._json({"ok": False, "error": str(e)})

    def _api_log(self):
        lines = []
        if OUTPUT_LOG.exists():
            try:
                with open(OUTPUT_LOG) as f:
                    lines = f.readlines()[-100:]
            except OSError:
                pass
        self._json({"lines": lines})

    def _json(self, data, status=200):
        self._send(status, "application/json", json.dumps(data).encode())

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress per-request logging


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="BlobMaster Training Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    # Initialize components
    manager = TrainingManager()
    tracker = HistoryTracker()
    tracker.start()

    # Wire into handler
    DashboardHandler.manager = manager
    DashboardHandler.tracker = tracker

    server = http.server.HTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print(f"Press Ctrl+C to stop the dashboard (training continues independently)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped. Training continues in background.")
        server.server_close()


if __name__ == "__main__":
    main()
