const state = {
    charts: {},
    records: [],
    selectedId: null,
    initializedOptions: false,
    refreshTimer: null,
    isFetching: false,
    pendingRefresh: false,
    pageContext: {
        slug: "overview",
        title: "Operational Safety Console",
        sources: [],
    },
};

const PAGE_FILTERS = {
    overview: ["reason", "sentiment", "emotion"],
    chat: ["reason", "sentiment"],
    video: ["reason", "emotion"],
    audio: ["reason"],
};

if (window.MWG_PAGE_CONTEXT && typeof window.MWG_PAGE_CONTEXT === "object") {
    state.pageContext = {
        ...state.pageContext,
        ...window.MWG_PAGE_CONTEXT,
    };
}

const API_BASE = (() => {
    const injected = String(window.MWG_API_BASE_URL || "").trim();
    if (injected && !injected.includes("{{")) {
        return injected.replace(/\/$/, "");
    }
    if (window.location.protocol === "file:") {
        return "http://127.0.0.1:8502";
    }
    return "";
})();

const URL_RUN_ID = (() => {
    const params = new URLSearchParams(window.location.search || "");
    return String(params.get("run_id") || "").trim();
})();

function apiUrl(path, queryString = "") {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    const base = API_BASE;
    if (queryString) {
        return `${base}${normalizedPath}?${queryString}`;
    }
    return `${base}${normalizedPath}`;
}

const elements = {
    runIdSelect: document.getElementById("runIdSelect"),
    sourceSelect: document.getElementById("sourceSelect"),
    sourceScopeHint: document.getElementById("sourceScopeHint"),
    statusSelect: document.getElementById("statusSelect"),
    severitySelect: document.getElementById("severitySelect"),
    categorySelect: document.getElementById("categorySelect"),
    reasonSelect: document.getElementById("reasonSelect"),
    sentimentSelect: document.getElementById("sentimentSelect"),
    emotionSelect: document.getElementById("emotionSelect"),
    confidenceMin: document.getElementById("confidenceMin"),
    confidenceMax: document.getElementById("confidenceMax"),
    startDate: document.getElementById("startDate"),
    endDate: document.getElementById("endDate"),
    searchInput: document.getElementById("searchInput"),
    limitInput: document.getElementById("limitInput"),
    autoRefreshToggle: document.getElementById("autoRefreshToggle"),
    refreshSeconds: document.getElementById("refreshSeconds"),
    applyBtn: document.getElementById("applyBtn"),
    resetBtn: document.getElementById("resetBtn"),
    exportCsvBtn: document.getElementById("exportCsvBtn"),
    exportJsonBtn: document.getElementById("exportJsonBtn"),
    statusText: document.getElementById("statusText"),
    snapshotText: document.getElementById("snapshotText"),
    sourcePanel: document.getElementById("sourcePanel"),
    semanticGroups: document.querySelectorAll("[data-filter-key]"),
    tableMeta: document.getElementById("tableMeta"),
    alertsTableBody: document.getElementById("alertsTableBody"),
    detailId: document.getElementById("detailId"),
    detailContent: document.getElementById("detailContent"),
    kpiTotal: document.getElementById("kpiTotal"),
    kpiFlagged: document.getElementById("kpiFlagged"),
    kpiSafe: document.getElementById("kpiSafe"),
    kpiFlagRate: document.getElementById("kpiFlagRate"),
    kpiHighCritical: document.getElementById("kpiHighCritical"),
    kpiAvgConf: document.getElementById("kpiAvgConf"),
    sourceSummaryStrip: document.getElementById("sourceSummaryStrip"),
};

function setStatus(text) {
    elements.statusText.textContent = text;
}

function getSelectedValues(selectElement) {
    return Array.from(selectElement.selectedOptions).map((option) => option.value);
}

function getSelectedFilterValues(selectElement) {
    const options = Array.from(selectElement.options);
    if (!options.length) {
        return [];
    }

    const selected = options
        .filter((option) => option.selected)
        .map((option) => option.value);

    if (selected.length === options.length) {
        return [];
    }

    return selected;
}

function getForcedSources() {
    const sourceList = state.pageContext.sources;
    if (!Array.isArray(sourceList)) {
        return [];
    }
    return sourceList
        .map((source) => String(source).trim())
        .filter((source) => source.length > 0);
}

function applyPageLayout() {
    const pageSlug = String(state.pageContext.slug || "overview").toLowerCase();
    const visibleFilters = new Set(PAGE_FILTERS[pageSlug] || PAGE_FILTERS.overview);

    elements.semanticGroups.forEach((group) => {
        const key = String(group.dataset.filterKey || "").trim();
        group.classList.toggle("is-hidden", !visibleFilters.has(key));
    });

    const forcedSources = getForcedSources();
    if (elements.sourcePanel) {
        elements.sourcePanel.classList.toggle("is-hidden", forcedSources.length === 1);
    }
}

function setMultiSelectOptions(selectElement, values, preserveSelection) {
    const existingSelection = preserveSelection
        ? new Set(getSelectedValues(selectElement))
        : new Set();

    selectElement.innerHTML = "";

    values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;

        if (preserveSelection) {
            option.selected = existingSelection.has(value);
        } else {
            option.selected = true;
        }

        selectElement.appendChild(option);
    });

    if (preserveSelection && getSelectedValues(selectElement).length === 0) {
        Array.from(selectElement.options).forEach((option) => {
            option.selected = true;
        });
    }
}

function updateFilterOptions(options) {
    const preserve = state.initializedOptions;
    const forcedSources = getForcedSources();

    if (URL_RUN_ID) {
        setMultiSelectOptions(elements.runIdSelect, [URL_RUN_ID], false);
        elements.runIdSelect.disabled = true;
    } else {
        setMultiSelectOptions(elements.runIdSelect, options.run_ids || [], preserve);
        elements.runIdSelect.disabled = false;
    }

    if (forcedSources.length) {
        setMultiSelectOptions(elements.sourceSelect, forcedSources, false);
        elements.sourceSelect.disabled = true;
        if (elements.sourceScopeHint) {
            elements.sourceScopeHint.textContent = `Locked to ${forcedSources.join(", ")} for this page.`;
        }
    } else {
        setMultiSelectOptions(elements.sourceSelect, options.sources || [], preserve);
        elements.sourceSelect.disabled = false;
        if (elements.sourceScopeHint) {
            elements.sourceScopeHint.textContent = "Select one or more sources.";
        }
    }

    setMultiSelectOptions(elements.severitySelect, options.severities || [], preserve);
    setMultiSelectOptions(elements.categorySelect, options.categories || [], preserve);
    setMultiSelectOptions(elements.reasonSelect, options.reasons || [], preserve);
    setMultiSelectOptions(elements.sentimentSelect, options.sentiments || [], preserve);
    setMultiSelectOptions(elements.emotionSelect, options.emotions || [], preserve);

    state.initializedOptions = true;
    applyPageLayout();
}

function buildQueryString() {
    const params = new URLSearchParams();
    const forcedSources = getForcedSources();

    if (URL_RUN_ID) {
        params.append("run_id", URL_RUN_ID);
    } else {
        getSelectedFilterValues(elements.runIdSelect).forEach((value) => params.append("run_id", value));
    }

    const sourceValues = forcedSources.length
        ? forcedSources
        : getSelectedFilterValues(elements.sourceSelect);
    sourceValues.forEach((value) => params.append("source", value));
    getSelectedFilterValues(elements.severitySelect).forEach((value) => params.append("severity", value));
    getSelectedFilterValues(elements.categorySelect).forEach((value) => params.append("category", value));
    getSelectedFilterValues(elements.reasonSelect).forEach((value) => params.append("reason", value));
    getSelectedFilterValues(elements.sentimentSelect).forEach((value) => params.append("sentiment", value));
    getSelectedFilterValues(elements.emotionSelect).forEach((value) => params.append("emotion", value));

    params.set("status", elements.statusSelect.value || "all");
    params.set("confidence_min", elements.confidenceMin.value || "0");
    params.set("confidence_max", elements.confidenceMax.value || "1");
    params.set("start_date", elements.startDate.value || "");
    params.set("end_date", elements.endDate.value || "");
    params.set("search", elements.searchInput.value || "");
    params.set("limit", elements.limitInput.value || "1000");

    return params.toString();
}

function renderKpis(metrics) {
    elements.kpiTotal.textContent = metrics.total ?? 0;
    elements.kpiFlagged.textContent = metrics.flagged ?? 0;
    elements.kpiSafe.textContent = metrics.safe ?? 0;
    elements.kpiFlagRate.textContent = `${(metrics.flag_rate ?? 0).toFixed(2)}%`;
    elements.kpiHighCritical.textContent = metrics.high_critical ?? 0;
    elements.kpiAvgConf.textContent = `Avg conf ${(metrics.avg_confidence_flagged ?? 0).toFixed(4)}`;
}

function renderCharts(chartData) {
    const sourceLabels = (chartData.sources || []).map((item) => item.label);
    const sourceValues = (chartData.sources || []).map((item) => item.value);
    if (!state.charts.source) {
        state.charts.source = new Chart(document.getElementById("sourceChart"), {
            type: "bar",
            data: {
                labels: sourceLabels,
                datasets: [{
                    label: "Records",
                    data: sourceValues,
                    backgroundColor: ["#0e8f83", "#d95d39", "#e8a619", "#11212d", "#4796ff"],
                    borderRadius: 8,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } },
            },
        });
    } else {
        state.charts.source.data.labels = sourceLabels;
        state.charts.source.data.datasets[0].data = sourceValues;
        state.charts.source.update("none");
    }

    const severityLabels = (chartData.severities || []).map((item) => item.label);
    const severityValues = (chartData.severities || []).map((item) => item.value);
    if (!state.charts.severity) {
        state.charts.severity = new Chart(document.getElementById("severityChart"), {
            type: "doughnut",
            data: {
                labels: severityLabels,
                datasets: [{
                    data: severityValues,
                    backgroundColor: ["#0e8f83", "#e8a619", "#d95d39", "#8b3dff", "#1f2937"],
                    borderWidth: 0,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "bottom" } },
            },
        });
    } else {
        state.charts.severity.data.labels = severityLabels;
        state.charts.severity.data.datasets[0].data = severityValues;
        state.charts.severity.update("none");
    }

    const reasonLabels = (chartData.reasons || []).map((item) => item.label);
    const reasonValues = (chartData.reasons || []).map((item) => item.value);
    if (!state.charts.reason) {
        state.charts.reason = new Chart(document.getElementById("reasonChart"), {
            type: "bar",
            data: {
                labels: reasonLabels,
                datasets: [{
                    label: "Count",
                    data: reasonValues,
                    backgroundColor: "#0e8f83",
                    borderRadius: 8,
                }],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { x: { beginAtZero: true } },
            },
        });
    } else {
        state.charts.reason.data.labels = reasonLabels;
        state.charts.reason.data.datasets[0].data = reasonValues;
        state.charts.reason.update("none");
    }

    const trendLabels = (chartData.trend || []).map((item) => item.minute);
    const trendFlagged = (chartData.trend || []).map((item) => item.flagged);
    const trendSafe = (chartData.trend || []).map((item) => item.safe);
    if (!state.charts.trend) {
        state.charts.trend = new Chart(document.getElementById("trendChart"), {
            type: "line",
            data: {
                labels: trendLabels,
                datasets: [
                    {
                        label: "Flagged",
                        data: trendFlagged,
                        borderColor: "#d95d39",
                        backgroundColor: "rgba(217, 93, 57, 0.18)",
                        fill: true,
                        tension: 0.28,
                    },
                    {
                        label: "Safe",
                        data: trendSafe,
                        borderColor: "#0e8f83",
                        backgroundColor: "rgba(14, 143, 131, 0.15)",
                        fill: true,
                        tension: 0.28,
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "bottom" } },
                scales: { y: { beginAtZero: true } },
            },
        });
    } else {
        state.charts.trend.data.labels = trendLabels;
        state.charts.trend.data.datasets[0].data = trendFlagged;
        state.charts.trend.data.datasets[1].data = trendSafe;
        state.charts.trend.update("none");
    }
}

function sourceLabel(sourceKey) {
    const key = String(sourceKey || "");
    if (key === "video_frame") {
        return "Video Frames";
    }
    if (key === "transcript") {
        return "Transcript";
    }
    if (key === "chat") {
        return "Chat";
    }
    if (key === "audio") {
        return "Audio";
    }
    return key || "Unknown";
}

function renderSourceSummary(summaryItems) {
    if (!elements.sourceSummaryStrip) {
        return;
    }

    if (!Array.isArray(summaryItems) || !summaryItems.length) {
        elements.sourceSummaryStrip.innerHTML =
            '<article class="source-pill source-pill-empty">No source data for current filters.</article>';
        return;
    }

    elements.sourceSummaryStrip.innerHTML = summaryItems
        .map((item) => {
            const source = sourceLabel(item.source);
            const total = Number(item.total || 0);
            const flagged = Number(item.flagged || 0);
            const safe = Number(item.safe || 0);
            const flagRate = Number(item.flag_rate || 0).toFixed(2);
            return `
                <article class="source-pill">
                    <h4>${escapeHtml(source)}</h4>
                    <p><strong>${total}</strong> total</p>
                    <p>${flagged} flagged | ${safe} safe</p>
                    <span>Flag rate ${flagRate}%</span>
                </article>
            `;
        })
        .join("");
}


function formatSeconds(value) {
    if (value == null || Number.isNaN(Number(value))) {
        return "n/a";
    }
    return `${Number(value).toFixed(3)}s`;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatEntities(entities) {
    if (!Array.isArray(entities) || !entities.length) {
        return "none";
    }
    const flattened = entities
        .map((entity) => {
            if (entity && typeof entity === "object" && "text" in entity) {
                return String(entity.text);
            }
            return String(entity);
        })
        .filter((item) => item.trim().length > 0);
    return flattened.join(", ") || "none";
}

function renderTable(records) {
    elements.alertsTableBody.innerHTML = "";

    if (!records.length) {
        const row = document.createElement("tr");
        row.innerHTML = '<td colspan="8">No rows match current filters.</td>';
        elements.alertsTableBody.appendChild(row);
        elements.tableMeta.textContent = "0 records";
        return;
    }

    records.forEach((record) => {
        const row = document.createElement("tr");
        if (record.id === state.selectedId) {
            row.classList.add("active");
        }

        const badgeClass = record.flagged ? "badge-flagged" : "badge-safe";
        const badgeText = record.flagged ? "FLAGGED" : "SAFE";

        row.innerHTML = `
            <td>${escapeHtml(record.id)}</td>
            <td>${escapeHtml(record.timestamp_display || record.timestamp || "")}</td>
            <td>${escapeHtml(record.source || "")}</td>
            <td><span class="badge ${badgeClass}">${badgeText}</span></td>
            <td>${escapeHtml(record.severity || "")}</td>
            <td>${Number(record.confidence || 0).toFixed(4)}</td>
            <td>${escapeHtml(record.reason_text || "")}</td>
            <td>${escapeHtml(record.preview || "")}</td>
        `;

        row.addEventListener("click", () => {
            state.selectedId = record.id;
            renderTable(state.records);
            renderDetail(record);
        });

        elements.alertsTableBody.appendChild(row);
    });

    elements.tableMeta.textContent = `${records.length} records shown`;
}

function renderDetail(record) {
    if (!record) {
        elements.detailId.textContent = "No selection";
        elements.detailContent.textContent = "Select a table row to view complete details.";
        return;
    }

    elements.detailId.textContent = `ID ${record.id} | ${record.source}`;

    let sourceSpecific = "";
    if (record.source === "chat" || record.source === "transcript") {
        let transcriptSegmentBlock = "";
        if (record.source === "transcript") {
            transcriptSegmentBlock = `
                <div class="detail-grid">
                    <div class="detail-tile"><h4>Segment Start</h4><p>${formatSeconds(record.segment_start_time)}</p></div>
                    <div class="detail-tile"><h4>Segment End</h4><p>${formatSeconds(record.segment_end_time)}</p></div>
                    <div class="detail-tile"><h4>Segment Duration</h4><p>${formatSeconds(record.segment_duration_sec)}</p></div>
                </div>
                <p><strong>Segment Confidence:</strong> ${record.segment_confidence == null ? "n/a" : Number(record.segment_confidence).toFixed(4)}</p>
            `;
        }

        sourceSpecific = `
            <div class="detail-grid">
                <div class="detail-tile"><h4>Profanity</h4><p>${record.has_profanity ? "Yes" : "No"}</p></div>
                <div class="detail-tile"><h4>PII</h4><p>${record.has_pii ? "Yes" : "No"}</p></div>
                <div class="detail-tile"><h4>Sentiment</h4><p>${escapeHtml(record.sentiment || "n/a")}</p></div>
            </div>
            <p><strong>PII Types:</strong> ${escapeHtml((record.pii_types || []).join(", ") || "none")}</p>
            <p><strong>Entities:</strong> ${escapeHtml(formatEntities(record.entities))}</p>
            ${transcriptSegmentBlock}
        `;
    } else if (record.source === "video_frame") {
        sourceSpecific = `
            <div class="detail-grid">
                <div class="detail-tile"><h4>Frame</h4><p>${record.frame_number ?? 0}</p></div>
                <div class="detail-tile"><h4>Second</h4><p>${Number(record.timestamp_sec || 0).toFixed(3)}</p></div>
                <div class="detail-tile"><h4>Emotion</h4><p>${escapeHtml(record.emotion || "none")}</p></div>
            </div>
            <p><strong>NSFW Label:</strong> ${escapeHtml(record.nsfw_label || "unknown")}</p>
            <p><strong>NSFW Score:</strong> ${Number(record.nsfw_score || 0).toFixed(4)}</p>
        `;
    } else if (record.source === "audio") {
        sourceSpecific = `
            <div class="detail-grid">
                <div class="detail-tile"><h4>Max dB</h4><p>${record.max_volume_db == null ? "n/a" : Number(record.max_volume_db).toFixed(2)}</p></div>
                <div class="detail-tile"><h4>Mean dB</h4><p>${record.mean_volume_db == null ? "n/a" : Number(record.mean_volume_db).toFixed(2)}</p></div>
                <div class="detail-tile"><h4>Silence Periods</h4><p>${record.silence_count ?? 0}</p></div>
            </div>
            <div class="detail-grid">
                <div class="detail-tile"><h4>Speech Rate</h4><p>${record.speech_rate_wpm == null ? "n/a" : Number(record.speech_rate_wpm).toFixed(1)}</p></div>
                <div class="detail-tile"><h4>Background dB</h4><p>${record.background_noise_db == null ? "n/a" : Number(record.background_noise_db).toFixed(2)}</p></div>
                <div class="detail-tile"><h4>Speakers</h4><p>${record.speaker_count ?? 0}</p></div>
            </div>
        `;
    }

    elements.detailContent.innerHTML = `
        <p><strong>Run ID:</strong> ${escapeHtml(record.run_id || "n/a")}</p>
        <p><strong>Timestamp:</strong> ${escapeHtml(record.timestamp_display || record.timestamp || "")}</p>
        <p><strong>Severity:</strong> ${escapeHtml(record.severity || "")}</p>
        <p><strong>Category:</strong> ${escapeHtml(record.category || "")}</p>
        <p><strong>Confidence:</strong> ${Number(record.confidence || 0).toFixed(4)}</p>
        <p><strong>Reasons:</strong> ${escapeHtml(record.reason_text || "safe")}</p>
        <p><strong>Health Status:</strong> ${escapeHtml(JSON.stringify(record.health_status || {}, null, 0) || "{}")}</p>
        <p><strong>Text:</strong> ${escapeHtml(record.content_text || "[no text available]")}</p>
        ${sourceSpecific}
        <h4>Raw Payload</h4>
        <pre>${escapeHtml(JSON.stringify(record, null, 2))}</pre>
    `;
}

function formatNow(isoString) {
    if (!isoString) {
        return "";
    }
    const dt = new Date(isoString);
    if (Number.isNaN(dt.getTime())) {
        return isoString;
    }
    return dt.toLocaleString();
}

async function fetchDashboardData() {
    if (state.isFetching) {
        state.pendingRefresh = true;
        return;
    }

    state.isFetching = true;
    state.pendingRefresh = false;
    setStatus("Loading data...");

    try {
        const queryString = buildQueryString();
        const response = await fetch(apiUrl("/api/dashboard-data", queryString));
        if (!response.ok) {
            throw new Error(`API error ${response.status}`);
        }

        const payload = await response.json();

        updateFilterOptions(payload.options || {});
        renderKpis(payload.metrics || {});
        renderSourceSummary(payload.source_summary || []);
        renderCharts(payload.chart_data || {});

        state.records = payload.records || [];

        if (state.selectedId) {
            const selected = state.records.find((record) => record.id === state.selectedId);
            renderDetail(selected || state.records[0]);
            if (!selected && state.records[0]) {
                state.selectedId = state.records[0].id;
            }
        } else if (state.records[0]) {
            state.selectedId = state.records[0].id;
            renderDetail(state.records[0]);
        } else {
            renderDetail(null);
        }

        renderTable(state.records);

        elements.snapshotText.textContent = `Showing ${payload.filtered_records} of ${payload.total_records} records | Updated ${formatNow(payload.generated_at)}`;
        setStatus("Updated successfully");
    } catch (error) {
        setStatus(`Failed to load: ${error.message}. Start backend with: python html_dashboard.py`);
    } finally {
        state.isFetching = false;
        if (state.pendingRefresh) {
            state.pendingRefresh = false;
            queueMicrotask(fetchDashboardData);
        }
    }
}

function downloadBlob(fileName, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function exportCsv() {
    if (!state.records.length) {
        setStatus("No records to export");
        return;
    }

    const headers = [
        "id",
        "timestamp",
        "source",
        "flagged",
        "severity",
        "category",
        "confidence",
        "reason_text",
        "preview",
    ];

    const lines = [headers.join(",")];
    state.records.forEach((record) => {
        const line = headers
            .map((key) => {
                const value = String(record[key] ?? "").replaceAll('"', '""');
                return `"${value}"`;
            })
            .join(",");
        lines.push(line);
    });

    downloadBlob(
        `melodywings_filtered_${Date.now()}.csv`,
        lines.join("\n"),
        "text/csv;charset=utf-8"
    );
    setStatus("CSV exported");
}

function exportJson() {
    if (!state.records.length) {
        setStatus("No records to export");
        return;
    }

    downloadBlob(
        `melodywings_filtered_${Date.now()}.json`,
        JSON.stringify(state.records, null, 2),
        "application/json;charset=utf-8"
    );
    setStatus("JSON exported");
}

function setupAutoRefresh() {
    if (state.refreshTimer) {
        clearInterval(state.refreshTimer);
        state.refreshTimer = null;
    }

    if (!elements.autoRefreshToggle.checked) {
        return;
    }

    const seconds = Math.max(2, Number(elements.refreshSeconds.value || 5));
    state.refreshTimer = setInterval(() => {
        fetchDashboardData();
    }, seconds * 1000);
}

function resetFilters() {
    [
        elements.runIdSelect,
        elements.sourceSelect,
        elements.severitySelect,
        elements.categorySelect,
        elements.reasonSelect,
        elements.sentimentSelect,
        elements.emotionSelect,
    ].forEach((selectElement) => {
        if (!selectElement || selectElement.disabled) {
            return;
        }
        Array.from(selectElement.options).forEach((option) => {
            option.selected = true;
        });
    });

    elements.statusSelect.value = "all";
    elements.confidenceMin.value = "0";
    elements.confidenceMax.value = "1";
    elements.startDate.value = "";
    elements.endDate.value = "";
    elements.searchInput.value = "";
    elements.limitInput.value = "1000";
    elements.autoRefreshToggle.checked = true;
    elements.refreshSeconds.value = "5";

    setupAutoRefresh();
    fetchDashboardData();
}

function registerEvents() {
    elements.applyBtn.addEventListener("click", () => {
        setupAutoRefresh();
        fetchDashboardData();
    });

    elements.resetBtn.addEventListener("click", resetFilters);
    elements.exportCsvBtn.addEventListener("click", exportCsv);
    elements.exportJsonBtn.addEventListener("click", exportJson);

    elements.autoRefreshToggle.addEventListener("change", setupAutoRefresh);
    elements.refreshSeconds.addEventListener("change", setupAutoRefresh);

    elements.searchInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            fetchDashboardData();
        }
    });
}

function init() {
    applyPageLayout();
    registerEvents();
    setupAutoRefresh();
    fetchDashboardData();
}

window.addEventListener("DOMContentLoaded", init);
