let currentMu = 0;
let currentSigma = 0;
let distChart = null;
let distributionPoints = [];
let cachedMinX = 0;
let cachedMaxX = 40;
let cachedMaxY = 0;

/** x where right-tail mass P(X>x) is negligible (~0.01% → reads ~0% past this point). */
function tailXNegligibleRightMass(mu, sigma) {
    return jStat.normal.inv(0.9999, mu, sigma);
}

/**
 * Right chart bound: (tail x + 10), rounded up to next multiple of 5.
 * Slider max: that bound minus 0.5 (half-integer lines). Capped at 80.5 / graph 81.
 */
function computeGraphAndSliderMax(mu, sigma) {
    const raw = tailXNegligibleRightMass(mu, sigma) + 10;
    let graphMax = Math.ceil(raw / 5) * 5;
    graphMax = Math.max(5, graphMax);
    let sliderMax = graphMax - 0.5;
    const SLIDER_CAP = 80.5;
    if (sliderMax > SLIDER_CAP) {
        sliderMax = SLIDER_CAP;
        graphMax = SLIDER_CAP + 0.5;
    }
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
        const [cfgRes, playersRes, teamsRes] = await Promise.all([
            fetch('/api/config'),
            fetch('/api/players'),
            fetch('/api/teams')
        ]);
        const cfg = await cfgRes.json();
        
        const players = await playersRes.json();
        const teams = await teamsRes.json();
        
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
            video.play().catch(() => {});
        }
        
    } catch (e) {
        console.error("Failed to load initial data", e);
    }
    
    // UI Elements
    const predictBtn = document.getElementById('predict-btn');
    const slider = document.getElementById('line-slider');
    
    predictBtn.addEventListener('click', handlePredict);
    slider.addEventListener('input', handleSliderMove);

    const restInput = document.getElementById('rest-input');
    const clampDaysRest = () => {
        const raw = restInput.value.trim();
        if (raw === '' || raw === '-') {
            restInput.value = '0';
            return;
        }
        let v = parseInt(restInput.value, 10);
        if (Number.isNaN(v)) {
            restInput.value = '0';
            return;
        }
        v = Math.min(10, Math.max(0, v));
        restInput.value = String(v);
    };
    const clampRestOnInput = () => {
        if (restInput.value === '' || restInput.value === '-') return;
        clampDaysRest();
    };
    restInput.addEventListener('input', clampRestOnInput);
    restInput.addEventListener('change', clampDaysRest);
    restInput.addEventListener('blur', clampDaysRest);
});

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
    const isHome = parseInt(document.getElementById('home-input').value);
    let daysRest = parseInt(document.getElementById('rest-input').value, 10);
    if (Number.isNaN(daysRest)) daysRest = 0;
    daysRest = Math.min(10, Math.max(0, daysRest));
    document.getElementById('rest-input').value = String(daysRest);
    
    // Parse player input "LAL LeBron James" -> "LeBron James"
    const parts = playerInput.trim().split(' ');
    if (parts.length < 2) {
        showError("Please select a valid player from the list.");
        return;
    }
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
                days_rest: daysRest
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
        
        document.getElementById('expected-pts-display').textContent = currentMu.toFixed(1);
        document.getElementById('rmse-display').textContent = currentSigma.toFixed(2);
        
        // Auto-set slider to nearest half-line; axis = tail+10 → next multiple of 5; slider = axis − 0.5
        const slider = document.getElementById('line-slider');
        const { graphMax, sliderMax } = computeGraphAndSliderMax(currentMu, currentSigma);
        slider.min = '0.5';
        slider.max = String(sliderMax);
        let snapped = Math.round(currentMu - 0.5) + 0.5;
        snapped = Math.min(sliderMax, Math.max(0.5, snapped));
        slider.value = snapped;
        document.getElementById('line-display').textContent = snapped.toFixed(1);

        cachedMinX = 0;
        cachedMaxX = graphMax;
        
        document.getElementById('viz-section').style.display = 'block';
        hideLoading();
        
        updateViz();
        
    } catch (e) {
        showError("Network error. Please try again.");
        hideLoading();
    }
}

function handleSliderMove(e) {
    document.getElementById('line-display').textContent = parseFloat(e.target.value).toFixed(1);
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
        cachedMaxX = Number.isFinite(smax) ? smax + 0.5 : computeGraphAndSliderMax(currentMu, currentSigma).graphMax;
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
        distChart.data.datasets[0].data = pointsUnder;
        distChart.data.datasets[1].data = pointsOver;
        distChart.update('none');
        return;
    }

    distChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Under Probability',
                    data: pointsUnder,
                    backgroundColor: 'rgba(239, 68, 68, 0.5)',
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true
                },
                {
                    label: 'Over Probability',
                    data: pointsOver,
                    backgroundColor: 'rgba(34, 197, 94, 0.5)',
                    borderColor: '#22c55e',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true
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
                    title: { display: true, text: 'Points Scored', color: '#c4b5fd' },
                    grid: { color: 'rgba(232, 121, 249, 0.08)' },
                    ticks: { color: '#c4b5fd' }
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
    document.getElementById('viz-section').style.display = 'none';
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
