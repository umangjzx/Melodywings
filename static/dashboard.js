const state = {
    charts: {},
    records: [],
    selectedId: null,
    initializedOptions: false,
    refreshTimer: null,
    isFetching: false,
    pendingRefresh: false,
    activeRunId: null,
    statusTimer: null,
    flaggedItems: [],
    selectedFlagId: null,
    pageContext: {
        slug: "overview",
        title: "Operational Safety Console",
        sources: [],
    },
};

const PAGE_FILTERS = {
    overview: ["reason", "sentiment", "emotion"],
    upload: ["reason", "sentiment", "emotion"],
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

const API_AUTH_TOKEN = String(window.MWG_API_AUTH_TOKEN || "").trim();
const API_AUTH_REQUIRED = Boolean(window.MWG_API_AUTH_REQUIRED);
const API_AUTH_QUERY_PARAM = String(window.MWG_API_AUTH_QUERY_PARAM || "api_key").trim() || "api_key";
const MAX_UPLOAD_MB = Math.max(1, Number(window.MWG_MAX_UPLOAD_MB || 512));
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
const ALLOWED_UPLOAD_EXTENSIONS = new Set([".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]);

const URL_RUN_ID = (() => {
    const params = new URLSearchParams(window.location.search || "");
    return String(params.get("run_id") || "").trim();
})();

const PAGE_SLUG = String(state.pageContext.slug || "overview").toLowerCase();
const IS_UPLOAD_PAGE = PAGE_SLUG === "upload";

function apiUrl(path, queryString = "") {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    const base = API_BASE;
    if (queryString) {
        return `${base}${normalizedPath}?${queryString}`;
    }
    return `${base}${normalizedPath}`;
}

function buildApiHeaders(extraHeaders = {}) {
    const headers = { ...extraHeaders };
    if (API_AUTH_TOKEN) {
        headers.Authorization = `Bearer ${API_AUTH_TOKEN}`;
    }
    return headers;
}

function validateUploadFile(file) {
    if (!file) {
        return "Please choose a video file.";
    }

    const fileName = String(file.name || "").trim();
    const lowerName = fileName.toLowerCase();
    const extension = lowerName.includes(".") ? lowerName.slice(lowerName.lastIndexOf(".")) : "";
    if (!ALLOWED_UPLOAD_EXTENSIONS.has(extension)) {
        return `Unsupported file type. Allowed: ${Array.from(ALLOWED_UPLOAD_EXTENSIONS).join(", ")}.`;
    }

    if (Number(file.size || 0) > MAX_UPLOAD_BYTES) {
        return `File too large. Max allowed size is ${MAX_UPLOAD_MB} MB.`;
    }

    const mimeType = String(file.type || "").toLowerCase();
    if (mimeType && !mimeType.startsWith("video/")) {
        return `Invalid MIME type: ${mimeType}. Please upload a video file.`;
    }

    return null;
}

const elements = {
    uploadForm: document.getElementById("uploadForm"),
    videoFileInput: document.getElementById("videoFileInput"),
    uploadStatus: document.getElementById("uploadStatus"),
    uploadVideo: document.getElementById("uploadVideo"),
    videoStatus: document.getElementById("videoStatus"),
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
    runProgressMeta: document.getElementById("runProgressMeta"),
    runProgressFill: document.getElementById("runProgressFill"),
    runProgressDetail: document.getElementById("runProgressDetail"),
    flaggedItemsGrid: document.getElementById("flaggedItemsGrid"),
    flagFilterNsfw: document.getElementById("flagFilterNsfw"),
    flagFilterEmotion: document.getElementById("flagFilterEmotion"),
    flagFilterAudio: document.getElementById("flagFilterAudio"),
    flagFilterText: document.getElementById("flagFilterText"),
    frameInspectorMeta: document.getElementById("frameInspectorMeta"),
    frameInspectorContent: document.getElementById("frameInspectorContent"),
    transcriptList: document.getElementById("transcriptList"),
    transcriptSearch: document.getElementById("transcriptSearch"),
    transcriptFlaggedOnly: document.getElementById("transcriptFlaggedOnly"),
    transcriptRefreshBtn: document.getElementById("transcriptRefreshBtn"),
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
    const activeRunId = getActiveRunId();

    if (IS_UPLOAD_PAGE) {
        setMultiSelectOptions(elements.runIdSelect, activeRunId ? [activeRunId] : [], false);
        elements.runIdSelect.disabled = true;
    } else if (URL_RUN_ID) {
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
    const activeRunId = getActiveRunId();

    if (IS_UPLOAD_PAGE) {
        if (activeRunId) {
            params.append("run_id", activeRunId);
        }
    } else if (URL_RUN_ID) {
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

function formatChartMinute(isoString) {
    if (!isoString) {
        return "";
    }
    const dt = new Date(isoString);
    if (Number.isNaN(dt.getTime())) {
        return String(isoString);
    }
    return dt.toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
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

    const sourceFlagRateData = chartData.source_flag_rate || [];
    const sourceFlagRateLabels = sourceFlagRateData.map((item) => item.label);
    const sourceFlagRateValues = sourceFlagRateData.map((item) => item.flag_rate);
    if (!state.charts.flagRate) {
        state.charts.flagRate = new Chart(document.getElementById("flagRateChart"), {
            type: "bar",
            data: {
                labels: sourceFlagRateLabels,
                datasets: [{
                    label: "Flag rate (%)",
                    data: sourceFlagRateValues,
                    backgroundColor: "rgba(217, 93, 57, 0.82)",
                    borderColor: "#d95d39",
                    borderWidth: 1,
                    borderRadius: 8,
                }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const index = Number(context.dataIndex || 0);
                                const chartRows = context.chart.$sourceFlagRate || [];
                                const row = chartRows[index] || {};
                                const rate = Number(row.flag_rate || 0).toFixed(2);
                                return `${rate}% (${row.flagged || 0}/${row.total || 0} flagged)`;
                            },
                        },
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: (value) => `${value}%`,
                        },
                    },
                },
            },
        });
        state.charts.flagRate.$sourceFlagRate = sourceFlagRateData;
    } else {
        state.charts.flagRate.data.labels = sourceFlagRateLabels;
        state.charts.flagRate.data.datasets[0].data = sourceFlagRateValues;
        state.charts.flagRate.$sourceFlagRate = sourceFlagRateData;
        state.charts.flagRate.update("none");
    }

    const confidenceDistribution = chartData.confidence_distribution || [];
    const confidenceLabels = confidenceDistribution.map((item) => item.bucket);
    const confidenceFlagged = confidenceDistribution.map((item) => item.flagged);
    const confidenceSafe = confidenceDistribution.map((item) => item.safe);
    if (!state.charts.confidenceDist) {
        state.charts.confidenceDist = new Chart(document.getElementById("confidenceDistChart"), {
            type: "bar",
            data: {
                labels: confidenceLabels,
                datasets: [
                    {
                        label: "Flagged",
                        data: confidenceFlagged,
                        backgroundColor: "rgba(217, 93, 57, 0.82)",
                        borderColor: "#d95d39",
                        borderWidth: 1,
                        borderRadius: 6,
                    },
                    {
                        label: "Safe",
                        data: confidenceSafe,
                        backgroundColor: "rgba(14, 143, 131, 0.78)",
                        borderColor: "#0e8f83",
                        borderWidth: 1,
                        borderRadius: 6,
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "bottom" } },
                scales: {
                    x: { stacked: true },
                    y: { beginAtZero: true, stacked: true },
                },
            },
        });
    } else {
        state.charts.confidenceDist.data.labels = confidenceLabels;
        state.charts.confidenceDist.data.datasets[0].data = confidenceFlagged;
        state.charts.confidenceDist.data.datasets[1].data = confidenceSafe;
        state.charts.confidenceDist.update("none");
    }

    const confidenceTrend = chartData.confidence_trend || [];
    const confidenceTrendLabels = confidenceTrend.map((item) => formatChartMinute(item.minute));
    const avgConfidenceValues = confidenceTrend.map((item) => item.avg_confidence);
    const avgFlaggedConfidenceValues = confidenceTrend.map((item) => item.avg_flagged_confidence);
    if (!state.charts.confidenceTrend) {
        state.charts.confidenceTrend = new Chart(document.getElementById("confidenceTrendChart"), {
            type: "line",
            data: {
                labels: confidenceTrendLabels,
                datasets: [
                    {
                        label: "Avg confidence (all)",
                        data: avgConfidenceValues,
                        borderColor: "#1f6aa5",
                        backgroundColor: "rgba(31, 106, 165, 0.18)",
                        pointRadius: 2,
                        tension: 0.24,
                    },
                    {
                        label: "Avg confidence (flagged)",
                        data: avgFlaggedConfidenceValues,
                        borderColor: "#d95d39",
                        backgroundColor: "rgba(217, 93, 57, 0.2)",
                        pointRadius: 2,
                        tension: 0.24,
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "bottom" } },
                scales: {
                    y: {
                        beginAtZero: true,
                        suggestedMax: 1,
                    },
                },
            },
        });
    } else {
        state.charts.confidenceTrend.data.labels = confidenceTrendLabels;
        state.charts.confidenceTrend.data.datasets[0].data = avgConfidenceValues;
        state.charts.confidenceTrend.data.datasets[1].data = avgFlaggedConfidenceValues;
        state.charts.confidenceTrend.update("none");
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

function setUploadStatus(text, isError = false) {
    if (!elements.uploadStatus) {
        return;
    }
    elements.uploadStatus.textContent = text;
    elements.uploadStatus.classList.toggle("status-error", isError);
}

function setUploadVideoSource(runId) {
    if (!elements.uploadVideo) {
        return;
    }
    if (!runId) {
        elements.uploadVideo.removeAttribute("src");
        elements.uploadVideo.load();
        if (elements.videoStatus) {
            elements.videoStatus.textContent = "Waiting for upload";
        }
        return;
    }

    elements.uploadVideo.src = apiUrl(`/video/${runId}`);
    elements.uploadVideo.load();
    elements.uploadVideo.play().catch(() => {});
    if (elements.videoStatus) {
        elements.videoStatus.textContent = "Previewing uploaded video";
    }
}

function getActiveRunId() {
    if (IS_UPLOAD_PAGE) {
        return state.activeRunId || "";
    }
    return state.activeRunId || URL_RUN_ID || "";
}

function clearUploadRunViews() {
    state.records = [];
    state.selectedId = null;
    state.flaggedItems = [];
    state.selectedFlagId = null;

    if (elements.snapshotText) {
        elements.snapshotText.textContent = "Upload a video to start analysis. Old history is hidden on this page.";
    }

    setRunProgress(0, "Idle", "Upload a video to begin processing.");
    renderFlaggedItems([]);
    renderFrameInspector(null);
    renderTranscript([]);

    if (
        state.charts.source ||
        state.charts.severity ||
        state.charts.reason ||
        state.charts.trend ||
        state.charts.flagRate ||
        state.charts.confidenceDist ||
        state.charts.confidenceTrend
    ) {
        renderCharts({
            sources: [],
            severities: [],
            reasons: [],
            trend: [],
            source_flag_rate: [],
            confidence_distribution: [],
            confidence_trend: [],
        });
    }
}

function setRunProgress(percent, metaText, detailText) {
    if (elements.runProgressFill) {
        elements.runProgressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }
    if (elements.runProgressMeta) {
        elements.runProgressMeta.textContent = metaText || "";
    }
    if (elements.runProgressDetail) {
        elements.runProgressDetail.textContent = detailText || "";
    }
}

function getFlagFilterTypes() {
    const types = [];
    if (elements.flagFilterNsfw?.checked) {
        types.push("nsfw");
    }
    if (elements.flagFilterEmotion?.checked) {
        types.push("emotion");
    }
    if (elements.flagFilterAudio?.checked) {
        types.push("audio");
    }
    if (elements.flagFilterText?.checked) {
        types.push("text");
    }
    return types;
}

function formatTimestamp(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) {
        return "n/a";
    }
    return `${Number(seconds).toFixed(2)}s`;
}

function renderFlaggedItems(items) {
    if (!elements.flaggedItemsGrid) {
        return;
    }

    if (!items.length) {
        elements.flaggedItemsGrid.innerHTML = '<div class="flagged-empty">No flagged items for current filters.</div>';
        return;
    }

    elements.flaggedItemsGrid.innerHTML = items
        .map((item) => {
            const isVideo = item.item_type === "video_frame";
            const thumbnail = isVideo && item.frame_url
                ? `<img src="${escapeHtml(item.frame_url)}" alt="Flagged frame" />`
                : `<div class="flagged-placeholder">${escapeHtml(item.item_type)}</div>`;
            const validation = item.validation || {};
            const correct = Number(validation.correct || 0);
            const incorrect = Number(validation.incorrect || 0);
            return `
                <article class="flag-card" data-flag-id="${item.id}">
                    <div class="flag-thumb">${thumbnail}</div>
                    <div class="flag-meta">
                        <h4>${escapeHtml(item.label || item.item_type)}</h4>
                        ${isVideo ? `<p>Time ${formatTimestamp(item.timestamp_sec)}</p>` : ``}
                        <p>Confidence ${(Number(item.confidence || 0)).toFixed(4)}</p>
                        ${item.message ? `<p class="flag-message">${escapeHtml(item.message)}</p>` : ``}
                        ${item.reason_text ? `<p class="flag-reason">${escapeHtml(item.reason_text)}</p>` : ``}
                        ${isVideo ? `<p class="flag-validation">${correct} correct · ${incorrect} incorrect</p>` : ``}
                    </div>
                </article>
            `;
        })
        .join("");

    Array.from(elements.flaggedItemsGrid.querySelectorAll(".flag-card")).forEach((card) => {
        card.addEventListener("click", () => {
            const id = Number(card.dataset.flagId || 0);
            const selected = items.find((item) => Number(item.id) === id);
            state.selectedFlagId = id;
            renderFrameInspector(selected || null);
        });
    });
}

function renderTranscript(items) {
    if (!elements.transcriptList) {
        return;
    }

    if (!items.length) {
        elements.transcriptList.innerHTML = '<div class="transcript-empty">No transcript segments yet.</div>';
        return;
    }

    elements.transcriptList.innerHTML = items
        .map((item) => {
            const badgeClass = item.flagged ? "badge-flagged" : "badge-safe";
            const badgeText = item.flagged ? "FLAGGED" : "SAFE";
            const start = formatSeconds(item.segment_start_time);
            const end = formatSeconds(item.segment_end_time);
            return `
                <article class="transcript-item ${item.flagged ? "is-flagged" : ""}">
                    <div class="transcript-meta">
                        <span class="transcript-time">${start} - ${end}</span>
                        <span class="badge ${badgeClass}">${badgeText}</span>
                        <span class="transcript-confidence">Conf ${(Number(item.segment_confidence || 0)).toFixed(3)}</span>
                    </div>
                    <p class="transcript-text">${escapeHtml(item.segment_text || "")}</p>
                    <p class="transcript-reasons">${escapeHtml(item.reason_text || "safe")}</p>
                </article>
            `;
        })
        .join("");
}

async function fetchTranscript() {
    if (!elements.transcriptList) {
        return;
    }

    const params = new URLSearchParams();
    const runId = getActiveRunId();
    if (IS_UPLOAD_PAGE && !runId) {
        renderTranscript([]);
        return;
    }

    if (runId) {
        params.set("run_id", runId);
    }
    const search = elements.transcriptSearch?.value || "";
    if (search) {
        params.set("search", search);
    }
    if (elements.transcriptFlaggedOnly?.checked) {
        params.set("flagged", "true");
    }

    try {
        const response = await fetch(apiUrl("/api/transcript", params.toString()), {
            headers: buildApiHeaders(),
        });
        if (!response.ok) {
            if (response.status === 401 && API_AUTH_REQUIRED) {
                throw new Error(`Unauthorized. Reopen dashboard with ?${API_AUTH_QUERY_PARAM}=<token>.`);
            }
            throw new Error(`API error ${response.status}`);
        }
        const payload = await response.json();
        renderTranscript(payload.items || []);
    } catch (error) {
        elements.transcriptList.innerHTML = `
            <div class="transcript-empty">Unable to load transcript (${escapeHtml(error.message)}).</div>
        `;
    }
}

function renderFrameInspector(item) {
    if (!elements.frameInspectorContent || !elements.frameInspectorMeta) {
        return;
    }

    if (!item) {
        elements.frameInspectorMeta.textContent = "No selection";
        elements.frameInspectorContent.textContent = "Select a flagged item to inspect details.";
        return;
    }

    elements.frameInspectorMeta.textContent = `ID ${item.id} · ${item.item_type}`;

    if (item.item_type !== "video_frame") {
        elements.frameInspectorContent.innerHTML = `
            <p><strong>Type:</strong> ${escapeHtml(item.item_type)}</p>
            <p><strong>Confidence:</strong> ${(Number(item.confidence || 0)).toFixed(4)}</p>
            <p><strong>Details:</strong> ${escapeHtml(item.message || item.reason_text || "")}</p>
        `;
        return;
    }

    const validation = item.validation || {};
    const correct = Number(validation.correct || 0);
    const incorrect = Number(validation.incorrect || 0);

    elements.frameInspectorContent.innerHTML = `
        <div class="frame-inspector-grid">
            <div class="frame-preview">
                <img src="${escapeHtml(item.frame_url)}" alt="Flagged frame preview" />
            </div>
            <div class="frame-details">
                <p><strong>Timestamp:</strong> ${formatTimestamp(item.timestamp_sec)}</p>
                <p><strong>Label:</strong> ${escapeHtml(item.label || "flagged")}</p>
                <p><strong>Confidence:</strong> ${(Number(item.confidence || 0)).toFixed(4)}</p>
                <p><strong>Validation:</strong> ${correct} correct · ${incorrect} incorrect</p>
                <div class="validation-actions">
                    <button class="btn btn-primary" data-feedback="correct">Correct</button>
                    <button class="btn btn-secondary" data-feedback="incorrect">Incorrect</button>
                </div>
                <div id="validationStatus" class="status-text"></div>
            </div>
        </div>
    `;

    const buttons = elements.frameInspectorContent.querySelectorAll("[data-feedback]");
    buttons.forEach((button) => {
        button.addEventListener("click", async () => {
            const feedback = button.dataset.feedback;
            const statusEl = elements.frameInspectorContent.querySelector("#validationStatus");
            if (statusEl) {
                statusEl.textContent = "Saving feedback...";
            }

            try {
                const response = await fetch(apiUrl("/validate"), {
                    method: "POST",
                    headers: buildApiHeaders({ "Content-Type": "application/json" }),
                    body: JSON.stringify({ frame_id: item.id, user_feedback: feedback }),
                });
                if (!response.ok) {
                    throw new Error(`API error ${response.status}`);
                }
                const payload = await response.json();
                if (statusEl) {
                    statusEl.textContent = "Feedback saved.";
                }
                const summary = payload.summary || {};
                item.validation = summary;
                renderFrameInspector(item);
                fetchFlaggedItems();
            } catch (error) {
                if (statusEl) {
                    statusEl.textContent = `Failed to save: ${error.message}`;
                }
            }
        });
    });
}

async function fetchFlaggedItems() {
    if (!elements.flaggedItemsGrid) {
        return;
    }

    const params = new URLSearchParams();
    const runId = getActiveRunId();
    if (IS_UPLOAD_PAGE && !runId) {
        state.flaggedItems = [];
        state.selectedFlagId = null;
        renderFlaggedItems([]);
        renderFrameInspector(null);
        return;
    }

    if (runId) {
        params.set("run_id", runId);
    }
    getFlagFilterTypes().forEach((type) => params.append("type", type));

    try {
        const response = await fetch(apiUrl("/api/flagged-items", params.toString()), {
            headers: buildApiHeaders(),
        });
        if (!response.ok) {
            throw new Error(`API error ${response.status}`);
        }
        const payload = await response.json();
        state.flaggedItems = payload.items || [];
        renderFlaggedItems(state.flaggedItems);

        if (state.selectedFlagId) {
            const selected = state.flaggedItems.find((item) => Number(item.id) === state.selectedFlagId);
            if (selected) {
                renderFrameInspector(selected);
            } else {
                state.selectedFlagId = null;
                renderFrameInspector(null);
            }
        }
    } catch (error) {
        elements.flaggedItemsGrid.innerHTML = `
            <div class="flagged-empty">Unable to load flagged items (${escapeHtml(error.message)}).</div>
        `;
    }
}

async function uploadVideo(file) {
    const uploadError = validateUploadFile(file);
    if (uploadError) {
        setUploadStatus(uploadError, true);
        return;
    }

    const formData = new FormData();
    formData.append("video", file);

    setUploadStatus("Uploading...", false);

    try {
        const response = await fetch(apiUrl("/upload"), {
            method: "POST",
            headers: buildApiHeaders(),
            body: formData,
        });

        if (!response.ok) {
            if (response.status === 401 && API_AUTH_REQUIRED) {
                throw new Error(`Unauthorized. Reopen dashboard with ?${API_AUTH_QUERY_PARAM}=<token>.`);
            }
            throw new Error(`API error ${response.status}`);
        }

        const payload = await response.json();
        state.activeRunId = payload.run_id;
        setUploadStatus(`Upload complete. Run ID ${payload.run_id}`, false);
        setUploadVideoSource(payload.run_id);
        if (elements.runIdSelect) {
            setMultiSelectOptions(elements.runIdSelect, [payload.run_id], false);
            elements.runIdSelect.disabled = true;
        }
        setupAutoRefresh();
        startStatusPolling();
        fetchDashboardData();
    } catch (error) {
        setUploadStatus(`Upload failed: ${error.message}`, true);
    }
}

async function fetchRunStatus() {
    const runId = getActiveRunId();
    if (!runId) {
        return;
    }

    try {
        const response = await fetch(apiUrl(`/status/${runId}`), {
            headers: buildApiHeaders(),
        });
        if (!response.ok) {
            if (response.status === 401 && API_AUTH_REQUIRED) {
                throw new Error(`Unauthorized. Reopen dashboard with ?${API_AUTH_QUERY_PARAM}=<token>.`);
            }
            throw new Error(`API error ${response.status}`);
        }

        const status = await response.json();
        const progress = Number(status.progress || 0);
        const processed = status.processed_frames ?? 0;
        const total = status.total_frames ?? 0;
        const stage = status.stage || "running";
        const current = status.current_frame;

        let detail = `Stage: ${stage}`;
        if (current) {
            detail = `Frame ${current.frame_number} @ ${formatTimestamp(current.timestamp_sec)} | ${current.label || ""}`;
        }
        setRunProgress(progress, `${processed}/${total} frames`, detail);

        if (status.status === "complete") {
            setRunProgress(100, `${processed}/${total} frames`, "Processing complete");
            stopStatusPolling();
            fetchDashboardData();
        }
        if (status.status === "error") {
            setRunProgress(progress, "Error", status.message || "Processing failed");
            stopStatusPolling();
        }
    } catch (error) {
        setRunProgress(0, "Status unavailable", error.message);
    }
}

function startStatusPolling() {
    stopStatusPolling();
    state.statusTimer = setInterval(fetchRunStatus, 2000);
    fetchRunStatus();
}

function stopStatusPolling() {
    if (state.statusTimer) {
        clearInterval(state.statusTimer);
        state.statusTimer = null;
    }
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
    if (IS_UPLOAD_PAGE && !getActiveRunId()) {
        clearUploadRunViews();
        return;
    }

    if (state.isFetching) {
        state.pendingRefresh = true;
        return;
    }

    state.isFetching = true;
    state.pendingRefresh = false;
    setStatus("Loading data...");

    try {
        const queryString = buildQueryString();
        const response = await fetch(apiUrl("/api/dashboard-data", queryString), {
            headers: buildApiHeaders(),
        });
        if (!response.ok) {
            if (response.status === 401 && API_AUTH_REQUIRED) {
                throw new Error(`Unauthorized. Reopen dashboard with ?${API_AUTH_QUERY_PARAM}=<token>.`);
            }
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
        fetchFlaggedItems();
        fetchTranscript();
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

    if (IS_UPLOAD_PAGE && !getActiveRunId()) {
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
    if (elements.uploadForm) {
        elements.uploadForm.addEventListener("submit", (event) => {
            event.preventDefault();
            const file = elements.videoFileInput?.files?.[0];
            uploadVideo(file);
        });
    }

    if (elements.videoFileInput) {
        elements.videoFileInput.addEventListener("change", () => {
            const file = elements.videoFileInput?.files?.[0];
            if (file) {
                const uploadError = validateUploadFile(file);
                if (uploadError) {
                    setUploadStatus(uploadError, true);
                } else {
                    const fileMb = (Number(file.size || 0) / (1024 * 1024)).toFixed(2);
                    setUploadStatus(`Selected: ${file.name} (${fileMb} MB)`, false);
                }
            }
        });
    }
    elements.applyBtn.addEventListener("click", () => {
        setupAutoRefresh();
        fetchDashboardData();
    });

    elements.resetBtn.addEventListener("click", resetFilters);
    elements.exportCsvBtn.addEventListener("click", exportCsv);
    elements.exportJsonBtn.addEventListener("click", exportJson);

    elements.autoRefreshToggle.addEventListener("change", setupAutoRefresh);
    elements.refreshSeconds.addEventListener("change", setupAutoRefresh);

    [
        elements.flagFilterNsfw,
        elements.flagFilterEmotion,
        elements.flagFilterAudio,
        elements.flagFilterText,
    ].forEach((checkbox) => {
        if (!checkbox) {
            return;
        }
        checkbox.addEventListener("change", () => {
            fetchFlaggedItems();
        });
    });

    elements.searchInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            fetchDashboardData();
        }
    });

    if (elements.transcriptSearch) {
        elements.transcriptSearch.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                fetchTranscript();
            }
        });
    }

    if (elements.transcriptFlaggedOnly) {
        elements.transcriptFlaggedOnly.addEventListener("change", () => {
            fetchTranscript();
        });
    }

    if (elements.transcriptRefreshBtn) {
        elements.transcriptRefreshBtn.addEventListener("click", () => {
            fetchTranscript();
        });
    }

    if (elements.runIdSelect) {
        elements.runIdSelect.addEventListener("change", () => {
            const selected = getSelectedValues(elements.runIdSelect);
            state.activeRunId = selected.length === 1 ? selected[0] : null;
            fetchFlaggedItems();
            fetchTranscript();
            setUploadVideoSource(state.activeRunId);
            if (state.activeRunId) {
                startStatusPolling();
            } else {
                stopStatusPolling();
            }
        });
    }
}

function init() {
    applyPageLayout();

    if (elements.uploadStatus && API_AUTH_REQUIRED && !API_AUTH_TOKEN) {
        setUploadStatus(`Auth enabled. Open with ?${API_AUTH_QUERY_PARAM}=<token> if API calls fail.`, false);
    }

    const uploadHint = document.getElementById("uploadHint");
    if (uploadHint) {
        uploadHint.textContent = `Accepted formats: ${Array.from(ALLOWED_UPLOAD_EXTENSIONS).join(", ")} | Max ${MAX_UPLOAD_MB} MB.`;
    }

    registerEvents();
    setupAutoRefresh();

    if (IS_UPLOAD_PAGE && !getActiveRunId()) {
        clearUploadRunViews();
        setUploadVideoSource("");
        stopStatusPolling();
        return;
    }

    fetchDashboardData();
    fetchFlaggedItems();
    fetchTranscript();
    if (getActiveRunId()) {
        setUploadVideoSource(getActiveRunId());
        startStatusPolling();
    }
}

window.addEventListener("DOMContentLoaded", init);
