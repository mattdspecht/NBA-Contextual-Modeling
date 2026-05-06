let currentMu = 0;
let currentSigma = 0;
let distChart = null;
let sparkChart = null;
let distributionPoints = [];
let cachedMinX = 0;
let cachedMaxX = 40;
let cachedMaxY = 0;
let currentSparkOpponents = [];
let selectedVenue = 1; // 1 = home, 0 = away
let _refreshPollInterval = null;

/** x where right-tail mass P(X>x) is negligible (~0.01% → reads ~0% past this point). */
function tailXNegligibleRightMass(mu, sigma) {
    return jStat.normal.inv(0.9999, mu, sigma);
}

/**
 * Tail x → next multiple of 5 → +5 → next multiple of 10 = chart right edge.
 * Slider max: graph edge minus 0.5 (half-integer lines). Capped at 80.5 / graph 81.
 */
function computeGraphAndSliderMax(mu, sigma) {
    const tailX = tailXNegligibleRightMass(mu, sigma);
    const nextFive = Math.ceil(tailX / 5) * 5;
    const plusFive = nextFive + 5;
    let graphMax = Math.ceil(plusFive / 10) * 10;
    graphMax = Math.max(10, graphMax);
    const SLIDER_CAP = 80;
    let sliderMax = Math.min(graphMax, SLIDER_CAP);
    return { graphMax, sliderMax };
}

const TEAM_ACRONYMS = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
};

document.addEventListener('DOMContentLoaded', async () => {
    // Fetch Data
    try {
        const [cfgRes, playersRes, teamsRes, refreshStateRes] = await Promise.all([
            fetch('/api/config'),
            fetch('/api/players'),
            fetch('/api/teams'),
            fetch('/api/refresh-state'),
        ]);
        const cfg = await cfgRes.json();
        const players = await playersRes.json();
        const teams = await teamsRes.json();

        const refreshState = await refreshStateRes.json();
        applyRefreshState(refreshState);
        if (refreshState.status === 'running') startRefreshPolling();
        
        const playersList = document.getElementById('players-list');
        players.forEach(p => {
            const option = document.createElement('option');
            const teamCode = resolveTeamCode(p);
            option.value = `${teamCode} ${p.player_name}`;
            playersList.appendChild(option);
        });
        
        const teamsList = document.getElementById('teams-list');
        teams.teams.forEach(t => {
            const option = document.createElement('option');
            option.value = t;
            teamsList.appendChild(option);
        });

        // Optional override from server (e.g. NBA_PROP_BG_VIDEO). Do not strip the default
        // <video src> from HTML when config omits a URL — that caused a visible flash.
        if (cfg.backgroundVideoUrl) {
            const overlay = document.getElementById('background-overlay');
            const video = document.getElementById('bg-video');
            video.src = cfg.backgroundVideoUrl;
            overlay.classList.add('has-background-video');
        }

    } catch (e) {
        console.error("Failed to load initial data", e);
    }

    initScrollVideo();

    // UI Elements
    const predictBtn = document.getElementById('predict-btn');
    const slider = document.getElementById('line-slider');
    const refreshBtn = document.getElementById('refresh-btn');

    predictBtn.addEventListener('click', handlePredict);
    slider.addEventListener('input', handleSliderMove);
    refreshBtn.addEventListener('click', handleRefresh);

    const venueHome = document.getElementById('venue-home');
    const venueAway = document.getElementById('venue-away');
    venueHome.addEventListener('click', () => {
        selectedVenue = 1;
        venueHome.classList.add('active');
        venueAway.classList.remove('active');
    });
    venueAway.addEventListener('click', () => {
        selectedVenue = 0;
        venueAway.classList.add('active');
        venueHome.classList.remove('active');
    });
});

// ── Scroll-driven background video ───────────────────────────────────────────

function initScrollVideo() {
    const video = document.getElementById('bg-video');
    if (!video) return;

    // Keep video paused at all times — scrubbing sets currentTime directly
    video.pause();
    video.addEventListener('play', () => video.pause());

    let rafPending = false;

    const scrub = () => {
        rafPending = false;
        if (!video.duration) return;
        const scrollMax = document.documentElement.scrollHeight - window.innerHeight;
        const progress = scrollMax > 0
            ? Math.min(1, Math.max(0, window.scrollY / scrollMax))
            : 0;
        video.currentTime = progress * video.duration;
    };

    window.addEventListener('scroll', () => {
        if (!rafPending) {
            rafPending = true;
            requestAnimationFrame(scrub);
        }
    }, { passive: true });

    // Re-scrub any time the page layout changes (e.g. visualization section appears)
    new ResizeObserver(scrub).observe(document.documentElement);

    if (video.readyState >= 1) {
        scrub();
    } else {
        video.addEventListener('loadedmetadata', scrub, { once: true });
    }
}

// ── Refresh helpers ──────────────────────────────────────────────────────────

function formatLastUpdatedDate(dateStr) {
    try {
        const d = new Date(dateStr + 'T12:00:00');
        if (Number.isNaN(d.getTime())) return dateStr;
        return d.toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' });
    } catch (_) {
        return dateStr;
    }
}

async function fetchRefreshState() {
    try {
        const res = await fetch('/api/refresh-state');
        if (!res.ok) return null;
        return await res.json();
    } catch (_) {
        return null;
    }
}

function applyRefreshState(state) {
    if (!state) return;
    const lastUpdatedEl = document.getElementById('refresh-last-updated');
    const statusArea = document.getElementById('refresh-status-area');
    const progressFill = document.getElementById('refresh-progress-fill');
    const statusMsg = document.getElementById('refresh-status-msg');
    const refreshBtn = document.getElementById('refresh-btn');

    if (lastUpdatedEl && state.last_updated) {
        lastUpdatedEl.textContent = 'Last updated: ' + formatLastUpdatedDate(state.last_updated);
    }

    if (!statusArea) return;

    if (state.status === 'running') {
        statusArea.classList.add('is-visible');
        statusMsg.classList.remove('is-error');
        progressFill.style.width = Math.round((state.progress || 0) * 100) + '%';
        statusMsg.textContent = state.message || 'Refreshing...';
        if (refreshBtn) refreshBtn.disabled = true;

    } else if (state.status === 'done') {
        statusArea.classList.add('is-visible');
        statusMsg.classList.remove('is-error');
        progressFill.style.width = '100%';
        statusMsg.textContent = state.message || 'Refresh complete.';
        if (refreshBtn) refreshBtn.disabled = false;
        stopRefreshPolling();
        setTimeout(() => statusArea.classList.remove('is-visible'), 4000);

    } else if (state.status === 'error') {
        statusArea.classList.add('is-visible');
        statusMsg.classList.add('is-error');
        statusMsg.textContent = state.message || 'An error occurred.';
        if (refreshBtn) refreshBtn.disabled = false;
        stopRefreshPolling();

    } else {
        // idle
        statusArea.classList.remove('is-visible');
        if (refreshBtn) refreshBtn.disabled = false;
    }
}

function startRefreshPolling() {
    if (_refreshPollInterval) return;
    _refreshPollInterval = setInterval(async () => {
        const state = await fetchRefreshState();
        if (state) applyRefreshState(state);
        if (state && state.status !== 'running') stopRefreshPolling();
    }, 5000);
}

function stopRefreshPolling() {
    if (_refreshPollInterval) {
        clearInterval(_refreshPollInterval);
        _refreshPollInterval = null;
    }
}

async function handleRefresh() {
    const refreshBtn = document.getElementById('refresh-btn');
    const statusArea = document.getElementById('refresh-status-area');
    const progressFill = document.getElementById('refresh-progress-fill');
    const statusMsg = document.getElementById('refresh-status-msg');

    refreshBtn.disabled = true;
    statusArea.classList.add('is-visible');
    statusMsg.classList.remove('is-error');
    progressFill.style.width = '0%';
    statusMsg.textContent = 'Starting refresh...';

    try {
        const res = await fetch('/api/refresh', { method: 'POST' });

        if (res.status === 423) {
            statusMsg.textContent = 'A refresh is already running. Please wait.';
            refreshBtn.disabled = false;
            return;
        }

        if (res.status === 429) {
            const retryAfterSecs = parseInt(res.headers.get('Retry-After') || '3600', 10);
            const mins = Math.ceil(retryAfterSecs / 60);
            statusMsg.textContent = `Data was refreshed recently. Try again in ~${mins} minute${mins === 1 ? '' : 's'}.`;
            refreshBtn.disabled = false;
            return;
        }

        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            statusMsg.classList.add('is-error');
            statusMsg.textContent = 'Error: ' + (data.detail || 'Failed to start refresh.');
            refreshBtn.disabled = false;
            return;
        }

        // 202 Accepted — start polling
        startRefreshPolling();

    } catch (_) {
        statusMsg.classList.add('is-error');
        statusMsg.textContent = 'Network error starting refresh.';
        refreshBtn.disabled = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────

function formatTsDisplay(v) {
    if (v == null || Number.isNaN(v)) return '—';
    return (v <= 1.25 ? v * 100 : v).toFixed(1) + '%';
}

function formatUsgDisplay(v) {
    if (v == null || Number.isNaN(v)) return '—';
    return v.toFixed(1) + '%';
}

/** Whole days from last game date to now; 0 if same calendar day or future-dated. */
function wholeDaysSinceLastGame(lastDate) {
    const end = new Date();
    end.setHours(0, 0, 0, 0);
    const start = new Date(lastDate);
    if (Number.isNaN(start.getTime())) return null;
    start.setHours(0, 0, 0, 0);
    const ms = end - start;
    const days = Math.floor(ms / (1000 * 60 * 60 * 24));
    return Math.max(0, days);
}

/** Green (recent) → red (stale); saturates at 365+ days (~1 year). */
function recencyColorForDaysSince(days) {
    const t = Math.min(Math.max(days / 365, 0), 1);
    const g = [34, 197, 94];
    const r = [239, 68, 68];
    const mix = (a, b) => Math.round(a + (b - a) * t);
    return `rgb(${mix(g[0], r[0])}, ${mix(g[1], r[1])}, ${mix(g[2], r[2])})`;
}

function formatDaysSinceLabel(days) {
    if (days <= 0) return 'Today';
    if (days === 1) return '1 day ago';
    return `${days} days ago`;
}

function populateContextFromPredict(data, playerName, playerTeam) {
    const p = data.player || {};
    const o = data.opponent || {};
    const m = data.matchup || {};

    const roll10 = numOrNull(data.roll10_pts ?? p.roll10_pts);
    const roll30 = numOrNull(data.roll30_pts ?? p.roll30_pts);
    const ema5 = numOrNull(data.ema5_pts ?? p.ema5_pts);
    const usg = numOrNull(data.roll10_usg_pct ?? p.roll10_usg_pct);
    const ts = numOrNull(data.roll10_ts_pct ?? p.roll10_ts_pct);
    const bpm = numOrNull(data.roll10_bpm ?? p.roll10_bpm);
    const pm = numOrNull(data.roll10_plus_minus ?? p.roll10_plus_minus);
    const gmsc = numOrNull(data.roll10_gmsc ?? p.roll10_gmsc);
    const roll5Pts = numOrNull(data.roll5_pts ?? p.roll5_pts);
    const roll10Efg = numOrNull(data.roll10_efg ?? p.roll10_efg);
    const oppL10 = numOrNull(data.opp_roll10_pts_allowed ?? o.roll10_pts_allowed);
    const oppL30 = numOrNull(data.opp_roll30_pts_allowed ?? o.roll30_pts_allowed);
    const oppDrtg = numOrNull(data.opp_roll10_drtg ?? o.roll10_team_drtg);
    const matchupAvg = numOrNull(data.matchup_hist_pts ?? m.matchup_hist_pts);
    const matchupCount = data.matchup_hist_count ?? m.matchup_hist_count ?? 0;
    const lastGame = data.last_game_date ?? p.last_game_date;
    const oppTeam = data.opp_team ?? o.acronym;
    const isHome = data.is_home ?? m.is_home;
    const daysRest = data.days_rest ?? m.days_rest;

    if (roll10 == null || roll30 == null || ema5 == null) {
        console.warn('Predict response missing rolling stats; restart API or hard-refresh.', data);
        return;
    }

    // Player card
    document.getElementById('player-card-name').textContent = playerName || '—';
    document.getElementById('player-card-team').textContent = playerTeam || '—';
    const homeFlag = Number(isHome);
    const dr = daysRest != null ? Number(daysRest) : 0;
    document.getElementById('pc-venue').textContent = homeFlag === 1 ? 'Home' : 'Away';
    document.getElementById('pc-opp').textContent = `vs ${oppTeam || '—'}`;
    document.getElementById('pc-rest').textContent = `${dr} day${dr === 1 ? '' : 's'} rest`;

    // Interval display (floor low at 0)
    const lo = (() => { const v = numOrNull(data.interval_low); return v != null ? Math.max(0, v) : null; })();
    const hi = numOrNull(data.interval_high);
    document.getElementById('interval-display').textContent =
        lo != null && hi != null ? `80% interval: ${lo.toFixed(1)} — ${hi.toFixed(1)}` : '';

    // Player stats tiles
    document.getElementById('insight-l10-pts').textContent = roll10.toFixed(1);
    document.getElementById('insight-l5-pts').textContent = roll5Pts != null ? roll5Pts.toFixed(1) : '—';
    document.getElementById('insight-l30-pts').textContent = roll30.toFixed(1);
    document.getElementById('insight-ema5').textContent = ema5.toFixed(1);
    document.getElementById('insight-usg').textContent = formatUsgDisplay(usg);
    document.getElementById('insight-efg').textContent = formatTsDisplay(roll10Efg);
    document.getElementById('insight-ts').textContent = formatTsDisplay(ts);
    document.getElementById('insight-gmsc').textContent = gmsc != null ? gmsc.toFixed(1) : '—';
    document.getElementById('insight-bpm').textContent = bpm != null ? bpm.toFixed(1) : '—';

    const pmEl = document.getElementById('insight-pm');
    if (pm != null) {
        pmEl.textContent = (pm >= 0 ? '+' : '') + pm.toFixed(1);
        pmEl.style.color = pm >= 0 ? 'var(--over-color)' : 'var(--under-color)';
    } else {
        pmEl.textContent = '—';
        pmEl.style.removeProperty('color');
    }

    // Matchup tiles
    document.getElementById('insight-opp-l10').textContent = oppL10 != null ? oppL10.toFixed(1) : '—';
    document.getElementById('insight-opp-l30').textContent = oppL30 != null ? oppL30.toFixed(1) : '—';
    document.getElementById('insight-opp-drtg').textContent = oppDrtg != null ? oppDrtg.toFixed(1) : '—';

    const matchupAvgEl = document.getElementById('insight-matchup-avg');
    const matchupCountEl = document.getElementById('insight-matchup-count');
    if (matchupAvg != null && matchupCount > 0) {
        matchupAvgEl.textContent = matchupAvg.toFixed(1);
        matchupCountEl.textContent = `${matchupCount} game${matchupCount === 1 ? '' : 's'}`;
    } else {
        matchupAvgEl.textContent = '—';
        matchupCountEl.textContent = 'No history';
    }

    // Last game row
    const lastGameRow = document.getElementById('insight-last-game-row');
    const daysSinceEl = document.getElementById('insight-days-since');
    if (lastGame) {
        const safe = String(lastGame).includes('T') ? lastGame : `${lastGame}T12:00:00`;
        const d = new Date(safe);
        if (Number.isNaN(d.getTime())) {
            document.getElementById('insight-last-game').textContent = String(lastGame);
            daysSinceEl.textContent = '';
            daysSinceEl.style.removeProperty('color');
            lastGameRow.classList.add('insight-last-game-row--empty');
        } else {
            document.getElementById('insight-last-game').textContent = d.toLocaleDateString(undefined, {
                weekday: 'short', month: 'short', day: 'numeric', year: 'numeric'
            });
            const daysSince = wholeDaysSinceLastGame(d);
            if (daysSince != null) {
                daysSinceEl.textContent = formatDaysSinceLabel(daysSince);
                daysSinceEl.style.color = recencyColorForDaysSince(daysSince);
            }
            lastGameRow.classList.remove('insight-last-game-row--empty');
        }
    } else {
        document.getElementById('insight-last-game').textContent = '—';
        daysSinceEl.textContent = '';
        daysSinceEl.style.removeProperty('color');
        lastGameRow.classList.add('insight-last-game-row--empty');
    }
}

function numOrNull(v) {
    if (v == null || v === '') return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}

function buildSparkChart(points, opponents = []) {
    const canvas = document.getElementById('sparkChart');
    if (!canvas || !points || points.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (sparkChart) {
        sparkChart.destroy();
        sparkChart = null;
    }

    const labels = points.map((_, i) => String(i + 1));
    sparkChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'PTS',
                data: points,
                borderColor: '#f97316',
                backgroundColor: 'rgba(249, 115, 22, 0.08)',
                fill: true,
                tension: 0.35,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: '#f97316',
                pointBorderColor: 'rgba(255, 255, 255, 0.2)',
                pointBorderWidth: 1,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 480 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(8, 16, 30, 0.92)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    titleColor: '#5d7080',
                    bodyColor: '#eef2ff',
                    padding: 10,
                    callbacks: {
                        title: (items) => {
                            const opp = opponents[items[0].dataIndex];
                            return opp ? `vs ${opp}` : '';
                        },
                        label: (item) => `${item.raw} pts`,
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#5d7080', font: { size: 10 }, maxRotation: 0 },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: {
                    ticks: { color: '#5d7080', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    beginAtZero: false
                }
            }
        }
    });
}



function resolveTeamCode(playerRow) {
    if (playerRow.team && typeof playerRow.team === 'string') {
        return playerRow.team;
    }
    if (playerRow.team_acronym && typeof playerRow.team_acronym === 'string') {
        return playerRow.team_acronym;
    }
    const teamName = playerRow.team_name || playerRow.player_team || '';
    if (typeof teamName !== 'string' || !teamName.trim()) {
        return 'UNK';
    }
    return TEAM_ACRONYMS[teamName] || 'UNK';
}

async function handlePredict() {
    const playerInput = document.getElementById('player-input').value;
    const oppTeam = document.getElementById('opp-input').value;
    const isHome = selectedVenue;

    // Parse player input "LAL LeBron James" -> team="LAL", name="LeBron James"
    const parts = playerInput.trim().split(' ');
    if (parts.length < 2) {
        showError("Please select a valid player from the list.");
        return;
    }
    const playerTeam = parts[0];
    const playerName = parts.slice(1).join(' ');

    hideError();
    showLoading();

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                player: playerName,
                opp_team: oppTeam,
                is_home: isHome,
            })
        });
        
        const data = await res.json();
        
        if (!res.ok) {
            showError(data.detail || "Prediction failed.");
            hideLoading();
            return;
        }
        
        currentMu = data.expected_pts;
        currentSigma = Math.max(data.rmse, 0.1);
        distributionPoints = [];
        cachedMaxY = 0;

        if (distChart) {
            distChart.destroy();
            distChart = null;
        }
        if (sparkChart) {
            sparkChart.destroy();
            sparkChart = null;
        }


        populateContextFromPredict(data, playerName, playerTeam);

        document.getElementById('expected-pts-display').textContent = currentMu.toFixed(1);
        document.getElementById('rmse-display').textContent = currentSigma.toFixed(2);
        
        const slider = document.getElementById('line-slider');
        const { graphMax, sliderMax } = computeGraphAndSliderMax(currentMu, currentSigma);
        slider.min = '1';
        slider.max = String(sliderMax);
        let snapped = Math.min(sliderMax, Math.max(1, Math.round(currentMu)));
        slider.value = snapped;
        document.getElementById('line-display').textContent = String(snapped);

        cachedMinX = 0;
        cachedMaxX = graphMax;

        const viz = document.getElementById('viz-section');
        viz.classList.remove('viz-reveal');
        viz.style.display = 'block';
        requestAnimationFrame(() => {
            requestAnimationFrame(() => viz.classList.add('viz-reveal'));
        });

        hideLoading();

        const predictBtn = document.getElementById('predict-btn');
        predictBtn.classList.remove('btn-success-pulse');
        void predictBtn.offsetWidth;
        predictBtn.classList.add('btn-success-pulse');
        setTimeout(() => predictBtn.classList.remove('btn-success-pulse'), 800);

        updateViz();

        const recentPts = (data.player && data.player.recent_pts) || data.recent_pts;
        currentSparkOpponents = (data.player && data.player.recent_opponents) || data.recent_opponents || [];
        if (recentPts && recentPts.length) {
            requestAnimationFrame(() => {
                buildSparkChart(recentPts, currentSparkOpponents);
                setTimeout(() => { if (sparkChart) sparkChart.resize(); }, 580);
            });
        }
        
    } catch (e) {
        showError("Network error. Please try again.");
        hideLoading();
    }
}

function handleSliderMove(e) {
    document.getElementById('line-display').textContent = String(parseInt(e.target.value, 10));
    updateViz();
}

function updateViz() {
    const line = parseFloat(document.getElementById('line-slider').value);
    
    // Calculate probabilities using jStat
    const probUnder = jStat.normal.cdf(line, currentMu, currentSigma);
    const probOver = 1 - probUnder;
    
    document.getElementById('prob-under-display').textContent = (probUnder * 100).toFixed(1) + '%';
    document.getElementById('prob-over-display').textContent = (probOver * 100).toFixed(1) + '%';
    
    drawChart(line);
}

function drawChart(line) {
    const ctx = document.getElementById('distChart').getContext('2d');

    if (!distributionPoints.length) {
        const smax = parseFloat(document.getElementById('line-slider').max);
        cachedMinX = 0;
        cachedMaxX = Number.isFinite(smax) ? smax : computeGraphAndSliderMax(currentMu, currentSigma).graphMax;
        const step = (cachedMaxX - cachedMinX) / 160;
        distributionPoints = [];

        for (let x = cachedMinX; x <= cachedMaxX; x += step) {
            distributionPoints.push({x: x, y: jStat.normal.pdf(x, currentMu, currentSigma)});
        }
        cachedMaxY = Math.max(...distributionPoints.map(p => p.y)) * 1.1;
    }

    const pointsUnder = distributionPoints.filter(p => p.x <= line);
    const pointsOver = distributionPoints.filter(p => p.x >= line);

    if (distChart) {
        distChart.data.datasets[1].data = pointsUnder;
        distChart.data.datasets[2].data = pointsOver;
        distChart.update('none');
        return;
    }

    distChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Density base',
                    data: distributionPoints.slice(),
                    backgroundColor: 'rgba(255, 255, 255, 0.04)',
                    borderColor: 'rgba(255, 255, 255, 0.14)',
                    borderWidth: 1.5,
                    fill: 'origin',
                    pointRadius: 0,
                    tension: 0.35,
                    order: 0
                },
                {
                    label: 'Under Probability',
                    data: pointsUnder,
                    backgroundColor: 'rgba(239, 68, 68, 0.48)',
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0,
                    order: 1
                },
                {
                    label: 'Over Probability',
                    data: pointsOver,
                    backgroundColor: 'rgba(34, 197, 94, 0.48)',
                    borderColor: '#22c55e',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0,
                    order: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: () => `Probability density`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: cachedMinX,
                    max: cachedMaxX,
                    title: { display: true, text: 'Points Scored', color: '#5d7080' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#5d7080' }
                },
                y: {
                    display: false,
                    min: 0,
                    max: cachedMaxY
                }
            }
        }
    });
}

function showLoading() {
    document.getElementById('loading').style.display = 'block';
    const viz = document.getElementById('viz-section');
    viz.style.display = 'none';
    viz.classList.remove('viz-reveal');
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showError(msg) {
    const errObj = document.getElementById('error-msg');
    errObj.textContent = msg;
    errObj.style.display = 'block';
}

function hideError() {
    document.getElementById('error-msg').style.display = 'none';
}
