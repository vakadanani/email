// ═══════════════════════════════════════════════
//  SpamShield AI — app.js (Full Enhanced Version)
// ═══════════════════════════════════════════════

const API_BASE = "http://127.0.0.1:5000";

// ─── DOM References ──────────────────────────
const emailText       = document.getElementById("emailText");
const charCount       = document.getElementById("charCount");
const scanBtn         = document.getElementById("scanBtn");
const resultSection   = document.getElementById("resultSection");
const resultCard      = document.getElementById("resultCard");
const resultIcon      = document.getElementById("resultIcon");
const resultVerdict   = document.getElementById("resultVerdict");
const resultDesc      = document.getElementById("resultDesc");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceFill  = document.getElementById("confidenceFill");
const accuracyValue   = document.getElementById("accuracyValue");
const accuracyFill    = document.getElementById("accuracyFill");
const textareaWrapper = document.getElementById("textareaWrapper");
const soundToggleBtn  = document.getElementById("soundToggleBtn");
const soundIconOn     = document.getElementById("soundIconOn");
const soundIconOff    = document.getElementById("soundIconOff");

// File upload
const uploadZone    = document.getElementById("uploadZone");
const fileInput     = document.getElementById("fileInput");
const uploadContent = document.getElementById("uploadContent");
const uploadSuccess = document.getElementById("uploadSuccess");
const uploadFilename= document.getElementById("uploadFilename");

// Image OCR scan
const imageInput       = document.getElementById("imageInput");
const checkImageBtn    = document.getElementById("checkImageBtn");
const imageErrorMsg    = document.getElementById("imageErrorMsg");
const imageResultPanel = document.getElementById("imageResultPanel");
const extractedTextBox = document.getElementById("extractedTextBox");
const imageVerdict     = document.getElementById("imageVerdict");
const imageConfidence  = document.getElementById("imageConfidence");
const imageCombinedNote = document.getElementById("imageCombinedNote");

// Feedback
const feedbackSection = document.getElementById("feedbackSection");
const fbCorrect       = document.getElementById("fbCorrect");
const fbWrong         = document.getElementById("fbWrong");
const feedbackMsg     = document.getElementById("feedbackMsg");

// Explainable AI
const explainCard = document.getElementById("explainCard");
const explainList = document.getElementById("explainList");

// Spam words
const spamWordsCard = document.getElementById("spamWordsCard");
const spamWordsList = document.getElementById("spamWordsList");

// URL analysis
const urlCard = document.getElementById("urlCard");
const urlList = document.getElementById("urlList");

// Sender reputation
const reputationCard = document.getElementById("reputationCard");
const reputationInfo = document.getElementById("reputationInfo");

const modelBadge = document.getElementById("modelBadge");
const predictImageAttach = document.getElementById("predictImageAttach");

// Dashboard
const statTotal  = document.getElementById("statTotal");
const statSpam   = document.getElementById("statSpam");
const statHam    = document.getElementById("statHam");
const historyBody= document.getElementById("historyBody");

// Analytics
const analyticsEmpty = document.getElementById("analyticsEmpty");

// User nav
const userAvatar = document.getElementById("userAvatar");
const userNameEl = document.getElementById("userName");

// State
let isSoundEnabled = true;
let currentScanId = null;
let currentPrediction = null;
let currentUser = null;
let trendChart = null;
let volumeChart = null;

// ─── Audio ───────────────────────────────────
const safeTone    = new Audio('safe_mail_tone.mp3');
const warningBeep = new Audio('warning_beep.mp3');

soundToggleBtn.addEventListener("click", () => {
    isSoundEnabled = !isSoundEnabled;
    soundToggleBtn.classList.toggle("muted", !isSoundEnabled);
    soundIconOn.style.display  = isSoundEnabled ? "block" : "none";
    soundIconOff.style.display = isSoundEnabled ? "none"  : "block";
    if (!isSoundEnabled) {
        safeTone.pause(); safeTone.currentTime = 0;
        warningBeep.pause(); warningBeep.currentTime = 0;
    }
});

function playFeedbackSound(type) {
    if (!isSoundEnabled) return;
    safeTone.pause(); safeTone.currentTime = 0;
    warningBeep.pause(); warningBeep.currentTime = 0;
    const p = (type === "spam") ? warningBeep.play() : safeTone.play();
    if (p !== undefined) p.catch(() => {});
}

// ─── Character counter ───────────────────────
emailText.addEventListener("input", () => {
    const len = emailText.value.length;
    charCount.textContent = `${len.toLocaleString()} / 5,000`;
    charCount.classList.toggle("warn", len > 4000 && len <= 4800);
    charCount.classList.toggle("danger", len > 4800);
});

// ═══════════════════════════════════════════════
//  TAB NAVIGATION
// ═══════════════════════════════════════════════
const tabBtns   = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        const tab = btn.dataset.tab;
        tabBtns.forEach(b => b.classList.remove("active"));
        tabPanels.forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("panel" + tab.charAt(0).toUpperCase() + tab.slice(1)).classList.add("active");

        // Load data when switching tabs
        if (tab === "dashboard") loadDashboard();
        if (tab === "analytics") loadAnalytics();
    });
});

// ═══════════════════════════════════════════════
//  FILE UPLOAD
// ═══════════════════════════════════════════════
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileUpload(files[0]);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFileUpload(fileInput.files[0]);
});

function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    uploadContent.style.display = "none";
    uploadSuccess.style.display = "flex";
    uploadFilename.textContent = `Uploading ${file.name}…`;

    fetch(`${API_BASE}/upload`, { method: "POST", body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                uploadFilename.textContent = `Error: ${data.error}`;
                setTimeout(() => resetUploadZone(), 3000);
                return;
            }
            uploadFilename.textContent = `✓ ${data.filename} loaded`;
            emailText.value = data.text;
            charCount.textContent = `${data.text.length.toLocaleString()} / 5,000`;
        })
        .catch(() => {
            uploadFilename.textContent = "Upload failed";
            setTimeout(() => resetUploadZone(), 3000);
        });
}

function resetUploadZone() {
    uploadContent.style.display = "flex";
    uploadSuccess.style.display = "none";
    fileInput.value = "";
}

// ═══════════════════════════════════════════════
//  IMAGE OCR + SPAM PREDICTION
// ═══════════════════════════════════════════════
if (checkImageBtn && imageInput) {
    checkImageBtn.addEventListener("click", checkImageSpam);
}

function checkImageSpam() {
    if (!imageInput || !checkImageBtn) return;

    imageErrorMsg.hidden = true;
    imageErrorMsg.textContent = "";

    if (!imageInput.files || imageInput.files.length === 0) {
        imageErrorMsg.textContent = "Please choose a JPG or PNG image first.";
        imageErrorMsg.hidden = false;
        return;
    }

    const file = imageInput.files[0];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["jpg", "jpeg", "png"].includes(ext)) {
        imageErrorMsg.textContent = "Unsupported file type. Use JPG, JPEG, or PNG.";
        imageErrorMsg.hidden = false;
        return;
    }

    const formData = new FormData();
    formData.append("image", file);
    const emailExtra = emailText.value.trim();
    if (emailExtra) formData.append("email_text", emailExtra);

    checkImageBtn.classList.add("loading");
    imageResultPanel.hidden = true;

    fetch(`${API_BASE}/predict_image`, { method: "POST", body: formData })
        .then(async (res) => {
            const data = await res.json().catch(() => ({}));
            return { ok: res.ok, data };
        })
        .then(({ ok, data }) => {
            checkImageBtn.classList.remove("loading");
            if (!ok || data.error) {
                imageErrorMsg.textContent = data.error || "Could not analyze the image. Try again.";
                imageErrorMsg.hidden = false;
                return;
            }

            imageResultPanel.hidden = false;
            extractedTextBox.textContent = data.extracted_text || "";

            const isSpam = data.prediction === "Spam";
            imageVerdict.textContent = data.prediction;
            imageVerdict.className = "image-verdict " + (isSpam ? "spam" : "ham");
            const clsName = data.model || data.classifier;
            const modelPart = clsName ? ` · ${clsName}` : "";
            imageConfidence.textContent = `Confidence: ${data.confidence}%${modelPart}`;

            if (data.combined_with_email) {
                imageCombinedNote.textContent = "Analysis used your email text above plus text from the image.";
                imageCombinedNote.hidden = false;
            } else {
                imageCombinedNote.hidden = true;
            }

            currentScanId = data.id;
            currentPrediction = data.prediction_raw;

            const conf01 = typeof data.confidence === "number" ? data.confidence / 100 : 0;
            playFeedbackSound(isSpam ? "spam" : "safe");
            showResult(data.prediction_raw, conf01, data.model_accuracy, data.model || data.classifier);
            showExplanation(data.explanation || []);
            showSpamWords(data.spam_words || [], data.prediction_raw);
            showUrlAnalysis(data.urls || []);
            showReputation(data.sender || "");

            feedbackMsg.textContent = "";
            fbCorrect.disabled = false;
            fbWrong.disabled = false;
            feedbackSection.style.display = "flex";

            imageResultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
        })
        .catch(() => {
            checkImageBtn.classList.remove("loading");
            imageErrorMsg.textContent = "Could not reach the server. Is server.py running?";
            imageErrorMsg.hidden = false;
        });
}

// ═══════════════════════════════════════════════
//  MAIN SCAN FUNCTION
// ═══════════════════════════════════════════════
function checkSpam() {
    const text = emailText.value.trim();

    if (text === "") {
        textareaWrapper.classList.add("shake");
        textareaWrapper.style.borderColor = "var(--red)";
        setTimeout(() => {
            textareaWrapper.classList.remove("shake");
            textareaWrapper.style.borderColor = "";
        }, 600);
        return;
    }

    scanBtn.classList.add("loading");
    resultSection.classList.remove("visible");

    const hasAttach = predictImageAttach && predictImageAttach.files && predictImageAttach.files.length > 0;
    const req = hasAttach
        ? (() => {
            const fd = new FormData();
            fd.append("message", text);
            fd.append("image", predictImageAttach.files[0]);
            return fetch(`${API_BASE}/predict`, { method: "POST", body: fd });
        })()
        : fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });

    req
    .then(res => {
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        return res.json();
    })
    .then(data => {
        scanBtn.classList.remove("loading");
        if (data.error) { showError(data.error); return; }

        currentScanId = data.id;
        currentPrediction = data.prediction;

        showResult(data.prediction, data.confidence, data.model_accuracy, data.model || data.classifier);
        showExplanation(data.explanation || []);
        showSpamWords(data.spam_words || [], data.prediction);
        showUrlAnalysis(data.urls || []);
        showReputation(data.sender);

        // Reset feedback
        feedbackMsg.textContent = "";
        fbCorrect.disabled = false;
        fbWrong.disabled = false;
        feedbackSection.style.display = "flex";
    })
    .catch(err => {
        console.error("Prediction error:", err);
        scanBtn.classList.remove("loading");
        showError("Could not reach the backend. Is server.py running?");
    });
}

// ─── Display result ──────────────────────────
function showResult(prediction, confidence, modelAccuracy, modelUsed) {
    const isSpam = prediction === "spam";
    const pct = confidence ? (confidence * 100) : 0;

    playFeedbackSound(isSpam ? "spam" : "safe");

    if (modelBadge) {
        if (modelUsed) {
            modelBadge.hidden = false;
            const isBert = String(modelUsed).toUpperCase().includes("BERT");
            modelBadge.className = "model-badge " + (isBert ? "bert" : "legacy");
            modelBadge.textContent = isBert ? "Model: BERT" : `Model: ${modelUsed}`;
        } else {
            modelBadge.hidden = true;
            modelBadge.textContent = "";
        }
    }

    resultCard.className = `result-card ${isSpam ? "spam" : "ham"}`;
    resultIcon.innerHTML = isSpam ? "🚨" : "✅";
    resultVerdict.textContent = isSpam ? "Spam Detected" : "Email is Safe";
    resultDesc.textContent = isSpam
        ? "This message exhibits patterns commonly found in unsolicited or malicious emails."
        : "This message does not match known spam patterns. It appears to be legitimate.";

    confidenceValue.textContent = `${pct.toFixed(1)}%`;
    confidenceFill.className = "confidence-fill";
    if (pct < 50) confidenceFill.classList.add("low");
    else if (pct < 70) confidenceFill.classList.add("medium");
    else if (pct < 90) confidenceFill.classList.add("high");
    else confidenceFill.classList.add("very-high");

    confidenceFill.style.width = "0%";
    requestAnimationFrame(() => {
        requestAnimationFrame(() => { confidenceFill.style.width = `${pct}%`; });
    });

    resultSection.classList.add("visible");
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });

    if (modelAccuracy) {
        const accPct = modelAccuracy * 100;
        accuracyValue.textContent = `${accPct.toFixed(2)}%`;
        accuracyFill.style.width = "0%";
        requestAnimationFrame(() => {
            requestAnimationFrame(() => { accuracyFill.style.width = `${accPct}%`; });
        });
    } else {
        accuracyValue.textContent = "—";
        accuracyFill.style.width = "0%";
    }
}

// ─── Explainable AI ──────────────────────────
function showExplanation(reasons) {
    explainList.innerHTML = "";
    if (reasons.length === 0) {
        explainCard.style.display = "none";
        return;
    }
    explainCard.style.display = "block";
    reasons.forEach(r => {
        const li = document.createElement("li");
        li.textContent = r;
        explainList.appendChild(li);
    });
}

// ─── Spam Word Highlights ────────────────────
function showSpamWords(words, prediction) {
    spamWordsList.innerHTML = "";
    if (words.length === 0 || prediction !== "spam") {
        spamWordsCard.style.display = "none";
        return;
    }
    spamWordsCard.style.display = "block";
    words.forEach(w => {
        const span = document.createElement("span");
        span.className = "spam-word";
        span.textContent = w;
        spamWordsList.appendChild(span);
    });
}

// ─── URL / Phishing Analysis ─────────────────
function showUrlAnalysis(urls) {
    urlList.innerHTML = "";
    if (urls.length === 0) {
        urlCard.style.display = "none";
        return;
    }
    urlCard.style.display = "block";
    urls.forEach(u => {
        const div = document.createElement("div");
        div.className = `url-item ${u.suspicious ? "suspicious" : "safe"}`;
        div.innerHTML = `
            <span class="url-status">${u.suspicious ? "⚠️" : "✅"}</span>
            <div class="url-details">
                <span class="url-domain">${escapeHtml(u.domain)}</span>
                ${u.flags.length > 0 ? `<span class="url-flags">${u.flags.join(" · ")}</span>` : '<span class="url-flags">No issues detected</span>'}
            </div>
        `;
        urlList.appendChild(div);
    });
}

// ─── Sender Reputation ───────────────────────
function showReputation(sender) {
    if (!sender) {
        reputationCard.style.display = "none";
        return;
    }
    fetch(`${API_BASE}/reputation?sender=${encodeURIComponent(sender)}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) { reputationCard.style.display = "none"; return; }
            reputationCard.style.display = "block";
            const levelClass = {
                trusted: "rep-trusted",
                suspicious: "rep-suspicious",
                dangerous: "rep-dangerous",
                unknown: "rep-unknown"
            }[data.level] || "rep-unknown";

            const levelLabels = {
                trusted: "✅ Trusted",
                suspicious: "⚠️ Suspicious",
                dangerous: "🚨 Dangerous",
                unknown: "❔ Unknown"
            };

            reputationInfo.innerHTML = `
                <div class="rep-sender">${escapeHtml(data.sender)}</div>
                <div class="rep-badge ${levelClass}">${levelLabels[data.level]}</div>
                <div class="rep-stats">
                    ${data.total_scans} scan(s) · ${data.spam_count} spam · ${data.ham_count} safe
                    ${data.total_scans > 0 ? ` · ${data.spam_ratio}% spam rate` : ''}
                </div>
            `;
        })
        .catch(() => { reputationCard.style.display = "none"; });
}

// ─── Feedback ────────────────────────────────
function sendFeedback(type) {
    if (!currentScanId) return;
    const fb = (type === "correct") ? currentPrediction : (currentPrediction === "spam" ? "ham" : "spam");

    fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: currentScanId, feedback: fb })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            feedbackMsg.textContent = "Thank you for your feedback!";
            feedbackMsg.className = "feedback-msg success";
        } else {
            feedbackMsg.textContent = "Could not save feedback.";
            feedbackMsg.className = "feedback-msg error";
        }
        fbCorrect.disabled = true;
        fbWrong.disabled = true;
    })
    .catch(() => {
        feedbackMsg.textContent = "Network error.";
        feedbackMsg.className = "feedback-msg error";
    });
}

// ─── Error ───────────────────────────────────
function showError(message) {
    if (modelBadge) {
        modelBadge.hidden = true;
        modelBadge.textContent = "";
    }
    resultCard.className = "result-card spam";
    resultIcon.innerHTML = "❌";
    resultVerdict.textContent = "Error";
    resultDesc.textContent = message;
    confidenceValue.textContent = "—";
    confidenceFill.style.width = "0%";
    accuracyValue.textContent = "—";
    accuracyFill.style.width = "0%";
    explainCard.style.display = "none";
    spamWordsCard.style.display = "none";
    urlCard.style.display = "none";
    reputationCard.style.display = "none";
    feedbackSection.style.display = "none";
    resultSection.classList.add("visible");
}

// ═══════════════════════════════════════════════
//  DASHBOARD
// ═══════════════════════════════════════════════
function loadDashboard() {
    fetch(`${API_BASE}/history`)
        .then(r => r.json())
        .then(data => {
            if (data.error) return;
            const s = data.stats;
            animateCounter(statTotal, s.total);
            animateCounter(statSpam, s.spam);
            animateCounter(statHam, s.ham);

            if (data.history.length === 0) {
                historyBody.innerHTML = '<tr><td colspan="5" class="empty-state">No scans yet. Go to the Scan tab to analyze an email.</td></tr>';
                return;
            }

            historyBody.innerHTML = data.history.map(h => `
                <tr class="${h.prediction === 'spam' ? 'row-spam' : 'row-ham'}">
                    <td>${h.id}</td>
                    <td class="preview-col">${escapeHtml(h.message_preview)}</td>
                    <td><span class="badge-result ${h.prediction}">${h.prediction === 'spam' ? '🚨 Spam' : '✅ Safe'}</span></td>
                    <td>${(h.confidence * 100).toFixed(1)}%</td>
                    <td>${formatTime(h.created_at)}</td>
                </tr>
            `).join('');
        })
        .catch(() => {});
}

function animateCounter(el, target) {
    const duration = 600;
    const start = parseInt(el.textContent) || 0;
    const diff = target - start;
    if (diff === 0) { el.textContent = target; return; }
    const startTime = performance.now();
    function step(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(start + diff * ease);
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function formatTime(ts) {
    if (!ts) return "—";
    const d = new Date(ts + "Z");
    return d.toLocaleDateString() + " " + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

// ═══════════════════════════════════════════════
//  ANALYTICS
// ═══════════════════════════════════════════════
function loadAnalytics() {
    fetch(`${API_BASE}/analytics`)
        .then(r => r.json())
        .then(data => {
            if (data.error) return;
            const daily = data.daily || [];
            if (daily.length === 0) {
                analyticsEmpty.style.display = "block";
                return;
            }
            analyticsEmpty.style.display = "none";

            const labels = daily.map(d => d.day);
            const spamData = daily.map(d => d.spam);
            const hamData = daily.map(d => d.ham);
            const totalData = daily.map(d => d.total);

            // Chart defaults
            Chart.defaults.color = '#94a3b8';
            Chart.defaults.borderColor = 'rgba(99,102,241,0.08)';

            // Trend chart
            const trendCtx = document.getElementById("trendChart").getContext("2d");
            if (trendChart) trendChart.destroy();
            trendChart = new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [
                        {
                            label: 'Spam',
                            data: spamData,
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239,68,68,0.1)',
                            fill: true,
                            tension: 0.4,
                            borderWidth: 2,
                            pointRadius: 3,
                            pointBackgroundColor: '#ef4444'
                        },
                        {
                            label: 'Safe',
                            data: hamData,
                            borderColor: '#22c55e',
                            backgroundColor: 'rgba(34,197,94,0.1)',
                            fill: true,
                            tension: 0.4,
                            borderWidth: 2,
                            pointRadius: 3,
                            pointBackgroundColor: '#22c55e'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { usePointStyle: true, padding: 16 } } },
                    scales: {
                        y: { beginAtZero: true, ticks: { stepSize: 1 } }
                    }
                }
            });

            // Volume chart
            const volCtx = document.getElementById("volumeChart").getContext("2d");
            if (volumeChart) volumeChart.destroy();
            volumeChart = new Chart(volCtx, {
                type: 'bar',
                data: {
                    labels,
                    datasets: [{
                        label: 'Total Scans',
                        data: totalData,
                        backgroundColor: 'rgba(99,102,241,0.5)',
                        borderColor: '#6366f1',
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { usePointStyle: true, padding: 16 } } },
                    scales: {
                        y: { beginAtZero: true, ticks: { stepSize: 1 } }
                    }
                }
            });
        })
        .catch(() => {});
}

// ─── Helpers ─────────────────────────────────
function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
}

// ─── Auth check & Initial load ───────────────
async function checkAuth() {
    try {
        const res = await fetch(`${API_BASE}/api/me`);
        if (res.status === 401) {
            window.location.href = '/login';
            return;
        }
        const data = await res.json();
        if (data.error) {
            window.location.href = '/login';
            return;
        }
        currentUser = data;
        // Update user nav
        if (userNameEl) userNameEl.textContent = data.username;
        if (userAvatar) {
            userAvatar.textContent = data.username.charAt(0).toUpperCase();
        }
        // Personalize dashboard heading
        const dashH = document.getElementById('dashboardHeading');
        if (dashH) dashH.innerHTML = `Welcome, <span class="gradient-text">${escapeHtml(data.username)}</span>`;
        const dashSub = document.getElementById('dashboardSubtitle');
        if (dashSub) dashSub.textContent = 'Your personal email scanning dashboard — stats, history, and threat analysis.';

        loadDashboard();
    } catch {
        window.location.href = '/login';
    }
}

checkAuth();