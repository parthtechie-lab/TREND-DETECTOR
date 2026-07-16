/**
 * Signal Hunter AI — Frontend Client Logic
 * Handles WebSocket subscriptions, dynamic table rendering, UI card transitions,
 * status checks, and data filtering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const wsStatus = document.getElementById("ws-status");
  const wsDot = document.getElementById("ws-dot");
  const wsLabel = document.getElementById("ws-label");
  const currentTime = document.getElementById("current-time");

  // Metrics
  const metricItemsTotal = document.getElementById("metric-items-total");
  const metricAlerts = document.getElementById("metric-alerts");
  const metricRejections = document.getElementById("metric-rejections");
  const metricClusters = document.getElementById("metric-clusters");
  const trendItems = document.getElementById("trend-items");
  const trendAlerts = document.getElementById("trend-alerts");

  // Queue depths
  const qRawVal = document.getElementById("q-raw-val");
  const qRawBar = document.getElementById("q-raw-bar");
  const qUniqueVal = document.getElementById("q-unique-val");
  const qUniqueBar = document.getElementById("q-unique-bar");
  const qScoredVal = document.getElementById("q-scored-val");
  const qScoredBar = document.getElementById("q-scored-bar");

  // Feeds
  const sourceGrid = document.getElementById("source-grid");
  const alertsList = document.getElementById("alerts-list");
  const alertCountBadge = document.getElementById("alert-count-badge");
  const signalTbody = document.getElementById("signal-tbody");

  // Controls
  const btnRefreshSources = document.getElementById("btn-refresh-sources");
  const filterCategory = document.getElementById("filter-category");
  const filterSource = document.getElementById("filter-source");
  const filterCorroborated = document.getElementById("filter-corroborated");
  const toastContainer = document.getElementById("toast-container");

  // Console Drawer Elements
  const consoleDrawer = document.getElementById("console-drawer");
  const btnToggleConsole = document.getElementById("btn-toggle-console");
  const consoleLogs = document.getElementById("console-logs");
  const consoleContent = document.getElementById("console-content");

  // Local cache/state
  let allSignals = [];
  let alertCount = 0;
  let itemsCount = 0;
  let rejectsCount = 0;

  // Update Clock
  function updateClock() {
    const now = new Date();
    currentTime.textContent = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";
  }
  updateClock();
  setInterval(updateClock, 1000);

  // --- WebSocket Connection ---
  let socket = null;
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;

    wsDot.className = "status-badge__dot status-badge__dot--connecting";
    wsLabel.textContent = "Connecting…";

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      wsDot.className = "status-badge__dot status-badge__dot--connected";
      wsLabel.textContent = "Connected";
      showToast("System Connected", "Real-time socket link established.", "signal");
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "scored_item") {
          handleIncomingSignal(payload.data);
        } else if (payload.type === "log_entry") {
          handleIncomingLog(payload.data);
        }
      } catch (err) {
        console.error("Failed to parse socket packet:", err);
      }
    };

    socket.onclose = () => {
      wsDot.className = "status-badge__dot status-badge__dot--error";
      wsLabel.textContent = "Disconnected";
      // Auto-reconnect in 5s
      setTimeout(connectWebSocket, 5000);
    };

    socket.onerror = (err) => {
      console.error("WebSocket encountered an error:", err);
      socket.close();
    };
  }

  // --- REST Requests ---
  async function fetchSignals() {
    try {
      const category = filterCategory.value;
      const source = filterSource.value;
      const corroborated = filterCorroborated.checked;

      let url = "/api/items?limit=50";
      if (category) url += `&category=${encodeURIComponent(category)}`;
      if (source) url += `&source=${encodeURIComponent(source)}`;
      if (corroborated) url += `&corroborated_only=true`;

      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      allSignals = data;
      renderSignalTable(data);

      // Simple metric estimate from data
      if (data.length > 0) {
        metricClusters.textContent = data.length;
      }
      updateCharts();
    } catch (err) {
      console.error("Error fetching signals:", err);
      signalTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--c-error)">Failed to load signals.</td></tr>`;
    }
  }

  async function fetchSources() {
    try {
      const resp = await fetch("/api/sources");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const sources = await resp.json();
      renderSourceGrid(sources);
    } catch (err) {
      console.error("Error fetching source health:", err);
    }
  }

  async function fetchAlerts() {
    try {
      const resp = await fetch("/api/alerts");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const alerts = await resp.json();
      renderAlertsList(alerts);
    } catch (err) {
      console.error("Error fetching alert logs:", err);
    }
  }

  async function fetchHealth() {
    try {
      const resp = await fetch("/api/health");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      // Update queue depth visualizers
      const raw = data.queue_depths.raw || 0;
      const unique = data.queue_depths.unique || 0;
      const scored = data.queue_depths.scored || 0;

      qRawVal.textContent = raw;
      qRawBar.style.width = `${(raw / 500) * 100}%`;
      qRawBar.setAttribute("aria-valuenow", raw);

      qUniqueVal.textContent = unique;
      qUniqueBar.style.width = `${(unique / 200) * 100}%`;
      qUniqueBar.setAttribute("aria-valuenow", unique);

      qScoredVal.textContent = scored;
      qScoredBar.style.width = `${(scored / 200) * 100}%`;
      qScoredBar.setAttribute("aria-valuenow", scored);

      // Fetch precise counter metrics
      const metricsResp = await fetch("/api/metrics");
      if (metricsResp.ok) {
        const metricsData = await metricsResp.json();
        rejectsCount = metricsData.hallucination_rejections || 0;
        metricRejections.textContent = rejectsCount;

        if (metricsData.items_ingested !== undefined) {
          itemsCount = metricsData.items_ingested;
          metricItemsTotal.textContent = itemsCount;
          trendItems.textContent = `+${Math.round(itemsCount / 5)}/min`;
        }
        if (metricsData.alerts_sent !== undefined) {
          alertCount = metricsData.alerts_sent;
          metricAlerts.textContent = alertCount;
        }
      }
      updateCharts();
    } catch (err) {
      console.error("Error fetching pipeline health:", err);
    }
  }

  // --- Rendering Functions ---
  function getCategoryBadgeClass(cat) {
    switch (cat) {
      case "AI Tool": return "cat-badge--ai-tool";
      case "SaaS": return "cat-badge--saas";
      case "Hardware": return "cat-badge--hardware";
      case "Meme": return "cat-badge--meme";
      default: return "cat-badge--other";
    }
  }

  function getCategoryEmoji(cat) {
    switch (cat) {
      case "AI Tool": return "🤖";
      case "SaaS": return "💼";
      case "Hardware": return "🔩";
      case "Meme": return "😂";
      default: return "📦";
    }
  }

  function renderSignalTable(signals) {
    if (signals.length === 0) {
      signalTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding: 40px; color: var(--c-text-3)">No signals match current filters.</td></tr>`;
      return;
    }

    signalTbody.innerHTML = "";
    signals.forEach((sig) => {
      const tr = document.createElement("tr");

      // Confidence Bar
      const confPct = Math.round(sig.confidence * 100);
      let barColor = "conf-bar__fill--low";
      if (sig.confidence >= 0.8) barColor = "conf-bar__fill--high";
      else if (sig.confidence >= 0.5) barColor = "conf-bar__fill--medium";

      const confCell = `
        <div class="conf-cell">
          <div class="conf-bar">
            <div class="conf-bar__fill ${barColor}" style="width: ${confPct}%"></div>
          </div>
          <span class="conf-pct">${confPct}%</span>
        </div>
      `;

      // Product and category
      const prodName = sig.product_name || "Unknown Product";
      const catEmoji = getCategoryEmoji(sig.category);
      const catBadge = `<span class="cat-badge ${getCategoryBadgeClass(sig.category)}">${catEmoji} ${sig.category}</span>`;

      // Sources
      const sourcesText = Array.isArray(sig.sources) ? sig.sources.join(", ") : sig.sources || "unknown";
      const sourceCount = sig.source_count || 1;
      const sourceCell = `<span class="source-chip" title="${sourcesText}">${sig.sources[0] || "unknown"} ${sourceCount > 1 ? `+${sourceCount - 1}` : ""}</span>`;

      // Evidence quote
      const escapedQuote = (sig.evidence_quote || "").replace(/"/g, "&quot;");
      const evidenceCell = sig.evidence_quote
        ? `<div class="evidence-quote" title="${escapedQuote}">“${sig.evidence_quote}”</div>`
        : `<span style="color: var(--c-text-3)">None</span>`;

      // Corroboration
      const corroborationCell = sig.corroborated
        ? `<span class="corr-badge">✅ ${sourceCount} sources</span>`
        : `<span class="corr-badge corr-badge--no">❌ Anecdotal</span>`;

      // Time
      const timeStr = sig.classified_at
        ? sig.classified_at.replace("T", " ").substring(11, 19)
        : "Unknown";

      tr.innerHTML = `
        <td>${confCell}</td>
        <td style="font-weight: 600; color: var(--c-text-1)">
          ${sig.url ? `<a href="${sig.url}" target="_blank" style="color: inherit; text-decoration: none; border-bottom: 1px dashed var(--c-border-2)">${prodName}</a>` : prodName}
        </td>
        <td>${catBadge}</td>
        <td>${sourceCell}</td>
        <td>${evidenceCell}</td>
        <td>${corroborationCell}</td>
        <td class="time-cell">${timeStr}</td>
      `;
      signalTbody.appendChild(tr);
    });
  }

  function renderSourceGrid(sources) {
    if (sources.length === 0) {
      sourceGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--c-text-3)">No sources monitored.</div>`;
      return;
    }

    sourceGrid.innerHTML = "";
    sources.forEach((src) => {
      const card = document.createElement("div");
      const isDegraded = src.is_degraded;
      card.className = `source-card ${isDegraded ? "source-card--degraded" : "source-card--ok"}`;

      const statusDot = `<span class="source-status-dot ${isDegraded ? "source-status-dot--degraded" : "source-status-dot--ok"}"></span>`;

      const lastSuccessStr = src.last_success
        ? new Date(src.last_success).toISOString().substring(11, 19)
        : "Never";

      card.innerHTML = `
        <div class="source-card__name">
          ${statusDot}
          <span>${src.source_name.toUpperCase()}</span>
          <span class="source-card__tier">${src.source_name === "tiktok" ? "Tier C" : src.source_name === "product_hunt" ? "Tier B" : "Tier A"}</span>
        </div>
        <div class="source-card__stats">
          <div class="source-card__stat">Consecutive Failures: <span>${src.consecutive_failures}</span></div>
          <div class="source-card__stat">Last Success: <span>${lastSuccessStr} UTC</span></div>
          <div class="source-card__stat">24h Items Vol: <span>${src.total_items_24h}</span></div>
        </div>
      `;
      sourceGrid.appendChild(card);
    });
  }

  function renderAlertsList(alerts) {
    if (alerts.length === 0) {
      alertsList.innerHTML = `
        <div class="empty-state">
          <div class="empty-state__icon">🔍</div>
          <div class="empty-state__text">No alerts sent yet.</div>
        </div>
      `;
      alertCountBadge.textContent = "0";
      return;
    }

    alertCount = alerts.length;
    alertCountBadge.textContent = alertCount;
    metricAlerts.textContent = alertCount;

    alertsList.innerHTML = "";
    alerts.forEach((alert) => {
      const item = document.createElement("div");
      const isRealtime = alert.alert_type === "realtime";
      item.className = `alert-item ${isRealtime ? "alert-item--realtime" : "alert-item--digest"}`;

      const icon = isRealtime ? "🚨" : "📊";
      const timeStr = alert.sent_at
        ? alert.sent_at.replace("T", " ").substring(11, 19)
        : "Unknown";

      item.innerHTML = `
        <div class="alert-item__icon">${icon}</div>
        <div class="alert-item__body">
          <div class="alert-item__title">${alert.canonical_title}</div>
          <div class="alert-item__meta">
            <span>Type: ${alert.alert_type}</span>
            <span>Sent: ${timeStr} UTC</span>
            ${alert.telegram_message_id ? `<span>ID: #${alert.telegram_message_id}</span>` : ""}
          </div>
        </div>
        <span class="alert-item__type ${isRealtime ? "alert-item__type--realtime" : "alert-item__type--digest"}">${alert.alert_type.toUpperCase()}</span>
      `;
      alertsList.appendChild(item);
    });
  }

  // --- Real-time Updates ---
  function handleIncomingSignal(item) {
    // 1. Show toast message
    const isHighConf = item.confidence >= 0.8;
    const catEmoji = getCategoryEmoji(item.category);
    const prodName = item.product_name || "Unknown Product";
    showToast(
      `${catEmoji} New Signal: ${prodName}`,
      `Classified as ${item.category} with ${Math.round(item.confidence * 100)}% confidence.`,
      isHighConf ? "alert" : "signal"
    );

    // 2. Play subtle notification sound for high-confidence alerts
    if (isHighConf) {
      try {
        const audio = new Audio("https://assets.mixkit.co/active_storage/sfx/2869/2869-500.wav");
        audio.volume = 0.3;
        audio.play();
      } catch (err) {
        // block by browser autoplay rules
      }
    }

    // 3. Dynamic metrics increment
    itemsCount++;
    metricItemsTotal.textContent = itemsCount;
    trendItems.textContent = `+${Math.round(itemsCount / 5)}/min`;

    // 4. Update the table locally (prepend to maintain chronological order)
    // Find if already exists in local list, update it, or prepend
    const existingIdx = allSignals.findIndex((s) => s.cluster_id === item.cluster_id);
    if (existingIdx !== -1) {
      allSignals[existingIdx] = {
        ...allSignals[existingIdx],
        sources: Array.from(new Set([...allSignals[existingIdx].sources, item.source])),
        source_count: allSignals[existingIdx].source_count + 1,
        corroborated: item.corroborated,
        confidence: Math.max(allSignals[existingIdx].confidence, item.confidence),
      };
    } else {
      allSignals.unshift({
        id: item.raw_item_id,
        cluster_id: item.cluster_id,
        product_name: item.product_name,
        category: item.category,
        evidence_quote: item.evidence_quote,
        confidence: item.confidence,
        trend_signal_present: item.trend_signal_present,
        classified_at: item.fetched_at,
        canonical_title: item.title,
        sources: [item.source],
        source_count: 1,
        corroborated: item.corroborated,
        url: item.url,
      });
    }

    // Apply active UI filters to see if the new item should display
    const activeCategory = filterCategory.value;
    const activeSource = filterSource.value;
    const activeCorrobed = filterCorroborated.checked;

    let displaySignals = [...allSignals];
    if (activeCategory) displaySignals = displaySignals.filter((s) => s.category === activeCategory);
    if (activeSource) displaySignals = displaySignals.filter((s) => s.sources.includes(activeSource));
    if (activeCorrobed) displaySignals = displaySignals.filter((s) => s.corroborated);

    renderSignalTable(displaySignals);
    updateCharts();

    // Refresh telemetry
    fetchHealth();
    fetchAlerts();
    fetchSources();
  }

  // --- Toast Manager ---
  function showToast(title, message, type = "signal") {
    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;

    let icon = "📡";
    if (type === "alert") icon = "🚨";
    if (type === "error") icon = "⚠️";

    toast.innerHTML = `
      <div class="toast__icon">${icon}</div>
      <div class="toast__body">
        <div class="toast__title">${title}</div>
        <div class="toast__msg">${message}</div>
      </div>
      <button class="toast__close" aria-label="Close notification">&times;</button>
    `;

    toastContainer.appendChild(toast);

    // Close button click listener
    toast.querySelector(".toast__close").addEventListener("click", () => {
      dismissToast(toast);
    });

    // Auto dismiss after 5s
    setTimeout(() => {
      dismissToast(toast);
    }, 5000);
  }

  function dismissToast(toast) {
    if (toast.parentNode) {
      toast.classList.add("toast--exiting");
      toast.addEventListener("animationend", () => {
        if (toast.parentNode) {
          toastContainer.removeChild(toast);
        }
      });
    }
  }

  // --- Events and Polling ---
  btnRefreshSources.addEventListener("click", () => {
    fetchSources();
    showToast("Sources Refreshed", "Loaded latest agent pathway telemetry.", "signal");
  });

  // Filter bindings
  filterCategory.addEventListener("change", fetchSignals);
  filterSource.addEventListener("change", fetchSignals);
  filterCorroborated.addEventListener("change", fetchSignals);

  // Initialize
  connectWebSocket();
  fetchSignals();
  fetchSources();
  fetchAlerts();
  fetchHealth();

  // --- Collapsible Log Console Toggle listener ---
  if (btnToggleConsole) {
    btnToggleConsole.addEventListener("click", () => {
      consoleDrawer.classList.toggle("console-drawer--open");
    });
  }

  function handleIncomingLog(log) {
    if (!consoleLogs) return;
    const line = document.createElement("div");
    let levelClass = "log-line--info";
    if (log.level === "WARNING") levelClass = "log-line--warning";
    else if (log.level === "ERROR") levelClass = "log-line--error";
    else if (log.level === "CRITICAL") levelClass = "log-line--critical";

    line.className = `log-line ${levelClass}`;

    const cleanTime = log.timestamp ? log.timestamp.substring(11, 19) : "00:00:00";
    line.innerHTML = `<span class="log-time">[${cleanTime}]</span> <span class="log-name">${log.logger}:</span> ${log.message}`;
    consoleLogs.appendChild(line);

    // Maintain max 100 log lines in DOM
    while (consoleLogs.childNodes.length > 100) {
      consoleLogs.removeChild(consoleLogs.firstChild);
    }
    
    // Auto scroll content
    consoleContent.scrollTop = consoleContent.scrollHeight;
  }

  // --- Chart.js Analytics Objects ---
  let categoryCounts = { "AI Tool": 0, "SaaS": 0, "Hardware": 0, "Meme": 0, "Other": 0 };
  let confidenceDistribution = [0, 0, 0, 0, 0];
  let passesCount = 0;

  // Initialize Category Breakdown Doughnut Chart
  const ctxCategories = document.getElementById("chart-categories").getContext("2d");
  const chartCategories = new Chart(ctxCategories, {
    type: 'doughnut',
    data: {
      labels: ['AI Tool', 'SaaS', 'Hardware', 'Meme', 'Other'],
      datasets: [{
        data: [0, 0, 0, 0, 0],
        backgroundColor: ['#6366f1', '#06b6d4', '#f59e0b', '#ef4444', '#475569'],
        borderWidth: 1,
        borderColor: '#0d1220'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#94a3b8', font: { size: 9, family: 'Inter' } }
        }
      }
    }
  });

  // Initialize Grounding Rejection Rate Pie Chart
  const ctxRejections = document.getElementById("chart-rejections").getContext("2d");
  const chartRejections = new Chart(ctxRejections, {
    type: 'pie',
    data: {
      labels: ['Grounded', 'Hallucinated'],
      datasets: [{
        data: [1, 0],
        backgroundColor: ['#10b981', '#ef4444'],
        borderWidth: 1,
        borderColor: '#0d1220'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#94a3b8', font: { size: 9, family: 'Inter' } }
        }
      }
    }
  });

  // Initialize Confidence Distribution Bar Chart
  const ctxConfidence = document.getElementById("chart-confidence").getContext("2d");
  const chartConfidence = new Chart(ctxConfidence, {
    type: 'bar',
    data: {
      labels: ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
      datasets: [{
        label: 'Signals',
        data: [0, 0, 0, 0, 0],
        backgroundColor: 'rgba(99, 102, 241, 0.4)',
        borderColor: '#6366f1',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { display: false } },
        y: { ticks: { color: '#94a3b8', font: { size: 9 }, stepSize: 1 }, grid: { color: 'rgba(99,102,241,0.05)' } }
      },
      plugins: { legend: { display: false } }
    }
  });

  function updateCharts() {
    // Reset tallies
    categoryCounts = { "AI Tool": 0, "SaaS": 0, "Hardware": 0, "Meme": 0, "Other": 0 };
    confidenceDistribution = [0, 0, 0, 0, 0];

    allSignals.forEach(sig => {
      // Category normalization for matching the chart labels
      let cat = sig.category || "Other";
      if (cat === "AI_TOOL") cat = "AI Tool";
      else if (cat === "SAAS") cat = "SaaS";
      else if (cat === "HARDWARE") cat = "Hardware";
      else if (cat === "MEME") cat = "Meme";
      else if (cat === "OTHER") cat = "Other";

      if (cat in categoryCounts) {
        categoryCounts[cat]++;
      } else {
        categoryCounts["Other"]++;
      }

      // Confidence buckets
      const conf = sig.confidence || 0.0;
      const idx = Math.min(4, Math.floor(conf / 0.2));
      confidenceDistribution[idx]++;
    });

    // Update Category chart
    chartCategories.data.datasets[0].data = [
      categoryCounts["AI Tool"],
      categoryCounts["SaaS"],
      categoryCounts["Hardware"],
      categoryCounts["Meme"],
      categoryCounts["Other"]
    ];
    chartCategories.update();

    // Update Confidence chart
    chartConfidence.data.datasets[0].data = confidenceDistribution;
    chartConfidence.update();

    // Update Rejections chart
    passesCount = allSignals.length;
    chartRejections.data.datasets[0].data = [passesCount, rejectsCount];
    chartRejections.update();
  }

  // Slow polls for metrics/health (10s)
  setInterval(() => {
    fetchHealth();
    fetchSources();
  }, 10000);
});
