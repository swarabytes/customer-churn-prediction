/**
 * NexIQ Customer Intelligence Platform
 * Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {

    // Mobile navigation toggle
    const navToggleBtn = document.getElementById('nav-toggle-btn');
    const navLinksMenu = document.getElementById('nav-links-menu');
    if (navToggleBtn && navLinksMenu) {
        navToggleBtn.addEventListener('click', () => {
            navLinksMenu.classList.toggle('open');
        });
    }

    // ============================================================
    // GLOBAL UTILITIES: Toast & Loader
    // ============================================================

    window.showToast = function (message, type = 'info', duration = 3500) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || 'ℹ'}</span>
            <span class="toast-msg">${message}</span>
            <span class="toast-close">×</span>
        `;

        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });

        container.appendChild(toast);

        setTimeout(() => {
            if (toast.parentElement) toast.remove();
        }, duration);
    };

    window.showLoader = function (text = 'Loading...') {
        const overlay = document.getElementById('global-loader');
        if (!overlay) return;
        const txt = overlay.querySelector('.loader-text');
        if (txt) txt.textContent = text;
        overlay.classList.add('active');
    };

    window.hideLoader = function () {
        const overlay = document.getElementById('global-loader');
        if (overlay) overlay.classList.remove('active');
    };


    // ============================================================
    // PREDICT PAGE
    // ============================================================

    const SAMPLE_DATA = {
        recency: 42,
        frequency: 3,
        monetary: 320.50,
        average_order_value: 106.83,
        category_diversity: 2,
        average_review_score: 4.5,
        payment_count: 3,
        delivery_delay: 0,
        inactivity_days: 42
    };

    const predictForm = document.getElementById('predict-form') || document.getElementById('churn-form');
    if (predictForm) {
        const btnAutofill = document.getElementById('btn-autofill');
        const btnPasteJson = document.getElementById('btn-paste-json');
        const btnParseJson = document.getElementById('btn-parse-json');
        const jsonInput = document.getElementById('json-input');

        // Dynamic Field Validation Setup
        const numericInputs = predictForm.querySelectorAll('.form-group input');
        numericInputs.forEach(input => {
            const errMsg = document.createElement('div');
            errMsg.className = 'field-error-msg';
            errMsg.id = `err-${input.name}`;
            input.parentElement.appendChild(errMsg);

            // Live validation on blur
            input.addEventListener('blur', () => validateField(input));
            input.addEventListener('input', () => {
                if (input.classList.contains('input-error')) validateField(input);
            });
        });

        // Autofill
        btnAutofill.addEventListener('click', () => {
            for (const key in SAMPLE_DATA) {
                const input = document.getElementById(key);
                if (input) {
                    input.value = SAMPLE_DATA[key];
                    input.classList.remove('input-error');
                    input.classList.add('input-valid');
                    const errEl = document.getElementById(`err-${key}`);
                    if (errEl) errEl.classList.remove('visible');
                }
            }
            showToast('Sample data applied!', 'success', 2000);
        });

        // Toggle JSON Textarea
        btnPasteJson.addEventListener('click', () => {
            const isHidden = jsonInput.style.display === 'none';
            jsonInput.style.display = isHidden ? 'block' : 'none';
            btnParseJson.style.display = isHidden ? 'inline-block' : 'none';
        });

        // Parse JSON
        btnParseJson.addEventListener('click', () => {
            try {
                const data = JSON.parse(jsonInput.value);
                for (const key in data) {
                    const input = document.getElementById(key);
                    if (input) input.value = data[key];
                }
                jsonInput.style.display = 'none';
                btnParseJson.style.display = 'none';
                showToast('JSON applied successfully!', 'success', 2000);
            } catch (e) {
                showToast('Invalid JSON format. Please check your input.', 'error');
            }
        });

        // Form Submit
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Validate all fields first
            let isValid = true;
            predictForm.querySelectorAll('.form-group input').forEach(input => {
                if (!validateField(input)) isValid = false;
            });

            if (!isValid) {
                showToast('Please fix the validation errors before submitting.', 'error');
                return;
            }

            // Build payload
            const formData = new FormData(predictForm);
            const payload = {};
            formData.forEach((value, key) => {
                if (key === 'model_type') {
                    payload[key] = value;   // keep as string
                } else {
                    payload[key] = parseFloat(value);
                }
            });
            // Ensure model_type is present even if not in FormData
            if (!payload.model_type) {
                payload.model_type = document.getElementById('model_type')?.value || 'logistic_regression';
            }

            const btnPredict = document.getElementById('btn-predict');
            btnPredict.disabled = true;
            showLoader('Analyzing customer risk profile...');

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const result = await res.json();
                if (result.error) throw new Error(result.error);

                showResult(result);
                showToast(`Analysis complete — Risk Level: ${result.risk_level}`, 'success');
                document.getElementById('result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
            } catch (err) {
                showToast('Analysis failed: ' + err.message, 'error');
            } finally {
                hideLoader();
                btnPredict.disabled = false;
            }
        });
    }

    function validateField(input) {
        const errEl = document.getElementById(`err-${input.name}`);
        const val = input.value.trim();
        let error = '';

        if (val === '' || val === null) {
            error = 'This field is required.';
        } else if (isNaN(parseFloat(val))) {
            error = 'Must be a valid number.';
        } else if (parseFloat(val) < 0) {
            error = 'Value cannot be negative.';
        }

        if (error) {
            input.classList.add('input-error');
            input.classList.remove('input-valid');
            if (errEl) { errEl.textContent = error; errEl.classList.add('visible'); }
            return false;
        } else {
            input.classList.remove('input-error');
            input.classList.add('input-valid');
            if (errEl) errEl.classList.remove('visible');
            return true;
        }
    }

    function showResult(result) {
        const section = document.getElementById('result-section');
        section.style.display = 'block';

        document.getElementById('res-cust-id').innerText = result.customer_id;

        // Outcome & model metadata
        const outcomeEl = document.getElementById('res-outcome');
        if (outcomeEl) outcomeEl.innerText = result.outcome_label || (result.prediction === 1 ? 'Churned' : 'Retained');
        const modelEl = document.getElementById('res-model-used');
        if (modelEl) modelEl.innerText = (result.model_used || 'logistic_regression').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

        const badge = document.getElementById('res-badge');
        badge.className = `badge ${result.risk_level.toLowerCase()}`;
        badge.innerText = `${result.risk_level} Risk`;

        const probPercent = Math.round(result.probability * 100);
        const probRetainedPercent = typeof result.probability_retained === 'number'
            ? Math.round(result.probability_retained * 100)
            : Math.round((1 - result.probability) * 100);

        const probAnim = document.getElementById('res-prob-anim');
        const circProg = document.getElementById('circular-progress');

        const churnValEl = document.getElementById('res-prob-churn-val');
        if (churnValEl) churnValEl.innerText = `${probPercent}%`;
        const retValEl = document.getElementById('res-prob-retained-val');
        if (retValEl) retValEl.innerText = `${probRetainedPercent}%`;

        let color = '#10b981';
        if (result.risk_level === 'Medium') color = '#f59e0b';
        if (result.risk_level === 'High') color = '#ef4444';

        let cur = 0;
        const interval = setInterval(() => {
            cur++;
            probAnim.innerText = `${cur}%`;
            circProg.style.background = `conic-gradient(${color} ${cur * 3.6}deg, rgba(255,255,255,0.07) 0deg)`;
            if (cur >= probPercent) clearInterval(interval);
        }, 15);

        // Use server-calculated health_score if available, else derive from probability
        const healthScore = typeof result.health_score === 'number'
            ? Math.round(result.health_score)
            : (100 - probPercent);
        setTimeout(() => {
            document.getElementById('res-health-fill').style.width = `${healthScore}%`;
        }, 300);

        document.getElementById('res-insight').innerText =
            result.insight || 'Customer shows stable purchasing behaviour.';

        const actions = Array.isArray(result.recommended_action)
            ? result.recommended_action
            : [result.recommended_action || 'Continue standard engagement.'];
        document.getElementById('res-actions').innerHTML =
            actions.map(a => `<li>${a}</li>`).join('');
    }


    // ============================================================
    // DASHBOARD PAGE
    // ============================================================
    if (document.getElementById('kpi-total')) {
        setupDashboardPage();
    }

    // ---- Pagination State ----
    let allPredictions = [];
    let filteredData = [];
    let currentPage = 1;
    const PAGE_SIZE = 10;

    // Registry of Chart instances to prevent canvas collision / leaks on refresh
    const chartInstances = {};

    function destroyChart(id) {
        if (chartInstances[id]) {
            try { chartInstances[id].destroy(); } catch (e) { }
            delete chartInstances[id];
        }
    }

    async function setupDashboardPage() {
        const refreshBtn = document.getElementById('btn-refresh-dashboard');
        const refreshIcon = document.getElementById('refresh-icon');

        async function loadDashboardData(isRefresh = false) {
            try {
                if (isRefresh && refreshIcon) refreshIcon.classList.add('spinning');
                showLoader(isRefresh ? 'Refreshing intelligence data...' : 'Loading intelligence data...');

                const res = await fetch('/api/dashboard-data');
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                allPredictions = data.recent_predictions || [];

                // KPIs
                animateValue('kpi-total', 0, data.stats.total_predictions || 0, 1000);
                animateValue('kpi-high-risk', 0, data.stats.high_risk_count || 0, 1000);
                const avgPct = Math.round((data.stats.avg_risk_probability || 0) * 100);
                const avgRiskEl = document.getElementById('kpi-avg-risk');
                if (avgRiskEl) avgRiskEl.innerText = `${avgPct}%`;
                const revRiskEl = document.getElementById('kpi-rev-risk');
                if (revRiskEl) revRiskEl.innerText = `$${Number(data.stats.revenue_at_risk || 0).toLocaleString()}`;

                // Executive Summary
                const total = data.stats.total_predictions || 0;
                const pctHigh = total > 0 ? Math.round((data.stats.high_risk_count / total) * 100) : 0;
                const execEl = document.getElementById('exec-summary-text');
                if (execEl) {
                    const alertCount = (data.alerts && data.alerts.length) ? data.alerts.length : 0;
                    execEl.innerText =
                        `${pctHigh}% of your customers are identified as High Risk. ` +
                        `Revenue loss potential stands at $${Number(data.stats.revenue_at_risk || 0).toLocaleString()}. ` +
                        `${alertCount} active alert${alertCount !== 1 ? 's' : ''} require attention.`;
                }

                // Alerts
                renderAlerts(data.alerts || []);

                // Charts
                buildCharts(allPredictions);

                // Table with search/filter/pagination
                filteredData = [...allPredictions];
                renderTable();
                setupTableControls();

                hideLoader();
                if (isRefresh) showToast('Dashboard refreshed successfully!', 'success');
            } catch (err) {
                hideLoader();
                showToast('Failed to load dashboard data: ' + err.message, 'error');
                console.error('Dashboard load error:', err);
            } finally {
                if (refreshIcon) refreshIcon.classList.remove('spinning');
            }
        }

        // Initial Load
        loadDashboardData(false);

        // Refresh Button Listener
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => loadDashboardData(true));
        }

        // Side panel close
        document.getElementById('close-panel')?.addEventListener('click', closeSidePanel);
        document.getElementById('panel-overlay')?.addEventListener('click', closeSidePanel);
    }

    function renderAlerts(alerts) {
        const list = document.getElementById('alerts-list');
        const countBadge = document.getElementById('alerts-count');
        if (!list) return;

        if (countBadge) countBadge.textContent = alerts ? alerts.length : 0;

        if (!alerts || alerts.length === 0) {
            list.innerHTML = '<p style="color:var(--text-secondary); font-size:0.9rem;">✅ No active alerts — all systems normal.</p>';
            return;
        }

        list.innerHTML = '';
        alerts.forEach(alert => {
            const div = document.createElement('div');
            div.className = `alert-item ${alert.type || 'info'}`;
            div.innerHTML = `
                <div class="alert-item-cust">${alert.customer_id || 'System'}</div>
                <div class="alert-item-msg">${alert.type === 'danger' ? '🔴' : '🟡'} ${alert.message}</div>
            `;
            list.appendChild(div);
        });
    }

    // ---- Search, Filter, Pagination ----
    function setupTableControls() {
        const searchInput = document.getElementById('searchInput');
        const filterRisk = document.getElementById('filterRisk');
        const filterPrio = document.getElementById('filterPriority');
        const btnPrev = document.getElementById('btn-prev');
        const btnNext = document.getElementById('btn-next');

        [searchInput, filterRisk, filterPrio].forEach(el => {
            if (el) el.addEventListener('input', () => { currentPage = 1; applyFilters(); });
        });

        if (btnPrev) btnPrev.addEventListener('click', () => { if (currentPage > 1) { currentPage--; renderTable(); } });
        if (btnNext) btnNext.addEventListener('click', () => {
            const totalPages = Math.ceil(filteredData.length / PAGE_SIZE);
            if (currentPage < totalPages) { currentPage++; renderTable(); }
        });
    }

    function applyFilters() {
        const query = (document.getElementById('searchInput')?.value || '').toLowerCase();
        const risk = (document.getElementById('filterRisk')?.value || '');
        const priority = (document.getElementById('filterPriority')?.value || '');

        filteredData = allPredictions.filter(p => {
            const matchSearch = !query || String(p.customer_id).toLowerCase().includes(query);
            const matchRisk = !risk || p.risk_level === risk;
            const matchPrio = !priority || p.priority === priority;
            return matchSearch && matchRisk && matchPrio;
        });

        renderTable();
    }

    function renderTable() {
        const tbody = document.getElementById('customer-table-body');
        const noMsg = document.getElementById('no-data-msg');
        if (!tbody) return;

        tbody.innerHTML = '';

        if (filteredData.length === 0) {
            if (noMsg) noMsg.style.display = 'block';
            updatePagination(0);
            return;
        }
        if (noMsg) noMsg.style.display = 'none';

        const totalPages = Math.ceil(filteredData.length / PAGE_SIZE);
        if (currentPage > totalPages) currentPage = totalPages;

        const start = (currentPage - 1) * PAGE_SIZE;
        const pageRows = filteredData.slice(start, start + PAGE_SIZE);

        pageRows.forEach((p, i) => {
            const tr = document.createElement('tr');
            tr.style.animationDelay = `${i * 40}ms`;
            tr.classList.add('anim-fade-in');

            const riskClass = (p.risk_level || 'low').toLowerCase();
            const probPct = Math.round((p.probability || 0) * 100);

            let prioClass = 'low';
            if (p.priority === 'Critical') prioClass = 'high';
            if (p.priority === 'Important') prioClass = 'medium';

            tr.innerHTML = `
                <td><strong>${p.customer_id}</strong></td>
                <td>${p.recency} days</td>
                <td>$${Number(p.monetary).toLocaleString()}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:0.6rem;">
                        <div style="width:50px; height:6px; background:var(--border-light); border-radius:3px; overflow:hidden;">
                            <div style="width:${probPct}%; height:100%; background:${riskClass === 'high' ? 'var(--danger)' : riskClass === 'medium' ? 'var(--warning)' : 'var(--success)'}; border-radius:3px;"></div>
                        </div>
                        ${probPct}%
                    </div>
                </td>
                <td><span class="badge ${riskClass}">${p.risk_level}</span></td>
                <td><span class="badge ${prioClass}">${p.priority || 'Monitor'}</span></td>
                <td><button class="btn-expand" data-id="${p.id}">View →</button></td>
            `;
            tbody.appendChild(tr);
        });

        // Re-bind expand buttons
        document.querySelectorAll('.btn-expand').forEach(btn => {
            btn.addEventListener('click', e => {
                const id = e.currentTarget.getAttribute('data-id');
                const rowData = filteredData.find(p => String(p.id) === String(id));
                if (rowData) openSidePanel(rowData);
            });
        });

        updatePagination(filteredData.length);
    }

    function updatePagination(total) {
        const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
        const start = Math.min((currentPage - 1) * PAGE_SIZE + 1, total);
        const end = Math.min(currentPage * PAGE_SIZE, total);

        const infoEl = document.getElementById('pagination-info');
        if (infoEl) infoEl.textContent = total > 0 ? `Showing ${start}–${end} of ${total}` : 'No results';

        const indEl = document.getElementById('page-indicator');
        if (indEl) indEl.textContent = `Page ${currentPage} of ${totalPages}`;

        const btnPrev = document.getElementById('btn-prev');
        const btnNext = document.getElementById('btn-next');
        if (btnPrev) btnPrev.disabled = currentPage <= 1;
        if (btnNext) btnNext.disabled = currentPage >= totalPages;
    }


    // ---- Side Panel ----
    function openSidePanel(data) {
        const panel = document.getElementById('side-panel');
        const overlay = document.getElementById('panel-overlay');
        const content = document.getElementById('panel-content');

        let insight = 'Customer shows stable purchasing behavior.';
        let action = 'Continue standard engagement campaigns.';

        if (data.inactivity_days > 90) {
            insight = 'Customer inactive for long period → High churn risk.';
            action = 'Send win-back email campaign with exclusive discount.';
        } else if (data.monetary > 500 && data.probability > 0.7) {
            insight = 'High value customer is at significant risk of leaving.';
            action = 'Direct VIP outreach via dedicated account manager.';
        } else if (data.average_review_score < 3.0) {
            insight = 'Consistently poor reviews → Bad experience risk.';
            action = 'Escalate to customer support for immediate resolution.';
        } else if (data.risk_level === 'Medium') {
            insight = 'Moderate drop in engagement detected.';
            action = 'Send personalized recommendations & loyalty reward.';
        }

        const healthScore = Math.round((1 - data.probability) * 100);
        const riskColor = data.risk_level === 'High' ? 'var(--danger)' : data.risk_level === 'Medium' ? 'var(--warning)' : 'var(--success)';

        content.innerHTML = `
            <div style="margin-top:1.2rem;">
                <p style="color:var(--text-secondary); font-size:0.8rem; margin-bottom:0.2rem;">CUSTOMER ID</p>
                <h4 style="font-size:1.1rem; margin-bottom:1.5rem; letter-spacing:0.02em;">${data.customer_id}</h4>
                
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; margin-bottom:1.5rem;">
                    <div style="background:rgba(0,0,0,0.25); padding:1rem; border-radius:10px; text-align:center;">
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; display:block; margin-bottom:0.3rem;">Risk Level</span>
                        <span class="badge ${data.risk_level.toLowerCase()}" style="display:inline-block;">${data.risk_level}</span>
                    </div>
                    <div style="background:rgba(0,0,0,0.25); padding:1rem; border-radius:10px; text-align:center;">
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; display:block; margin-bottom:0.3rem;">Probability</span>
                        <span style="font-weight:700; font-size:1.4rem; color:${riskColor};">${Math.round(data.probability * 100)}%</span>
                    </div>
                    <div style="background:rgba(0,0,0,0.25); padding:1rem; border-radius:10px; text-align:center;">
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; display:block; margin-bottom:0.3rem;">Health Score</span>
                        <span style="font-weight:700; font-size:1.4rem; color:var(--success);">${healthScore}/100</span>
                    </div>
                    <div style="background:rgba(0,0,0,0.25); padding:1rem; border-radius:10px; text-align:center;">
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; display:block; margin-bottom:0.3rem;">Priority</span>
                        <span class="badge ${data.priority === 'Critical' ? 'high' : data.priority === 'Important' ? 'medium' : 'low'}" style="display:inline-block;">${data.priority || 'Monitor'}</span>
                    </div>
                </div>

                <div style="background:rgba(239,68,68,0.1); padding:1rem; border-radius:10px; border-left:3px solid var(--danger); margin-bottom:1rem;">
                    <h5 style="color:var(--danger); margin-bottom:0.4rem; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em;">⚡ Risk Explanation</h5>
                    <p style="font-size:0.9rem; line-height:1.5;">${insight}</p>
                </div>
                <div style="background:rgba(16,185,129,0.1); padding:1rem; border-radius:10px; border-left:3px solid var(--success); margin-bottom:1.5rem;">
                    <h5 style="color:var(--success); margin-bottom:0.4rem; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em;">✅ Suggested Action</h5>
                    <p style="font-size:0.9rem; line-height:1.5;">${action}</p>
                </div>

                <h5 style="border-bottom:1px solid var(--border-light); padding-bottom:0.6rem; margin-bottom:1rem; font-size:0.82rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em;">Behavioral Metrics</h5>
                <ul style="list-style:none; display:flex; flex-direction:column; gap:0.6rem;">
                    ${[
                ['Recency', `${data.recency} days`],
                ['Frequency', `${data.frequency} orders`],
                ['Monetary Value', `$${Number(data.monetary).toLocaleString()}`],
                ['Avg Order Value', `$${data.average_order_value}`],
                ['Category Diversity', data.category_diversity],
                ['Avg Review Score', `${data.average_review_score} / 5`],
                ['Payment Count', data.payment_count],
                ['Delivery Delay', `${data.delivery_delay} days`],
                ['Inactivity Days', `${data.inactivity_days} days`],
            ].map(([label, val]) => `
                        <li style="display:flex; justify-content:space-between; align-items:center; font-size:0.88rem;">
                            <span style="color:var(--text-secondary);">${label}</span>
                            <strong>${val}</strong>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;

        panel.classList.add('open');
        overlay.classList.add('active');
    }

    function closeSidePanel() {
        document.getElementById('side-panel')?.classList.remove('open');
        document.getElementById('panel-overlay')?.classList.remove('active');
    }


    // ---- Animate Counter ----
    function animateValue(id, start, end, duration) {
        const obj = document.getElementById(id);
        if (!obj) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) window.requestAnimationFrame(step);
        };
        window.requestAnimationFrame(step);
    }

    // ---- Build Charts (Original 3 Charts - Upgraded Aesthetics & Animations) ----
    function buildCharts(predictions) {
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";

        const totalCount = predictions.length;

        // 1. Risk Distribution Doughnut (Enhanced with center text & glowing gradient hover)
        const riskCounts = { Low: 0, Medium: 0, High: 0 };
        predictions.forEach(p => { if (riskCounts[p.risk_level] !== undefined) riskCounts[p.risk_level]++; });

        const ctxDist = document.getElementById('riskDistChart');
        if (ctxDist) {
            destroyChart('riskDistChart');

            // Center text plugin for doughnut
            const centerTextPlugin = {
                id: 'centerTextPlugin',
                beforeDraw: (chart) => {
                    const { width, height, ctx } = chart;
                    ctx.save();
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    const centerX = width / 2;
                    const centerY = height / 2 - 8;

                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 1.6rem Inter, sans-serif';
                    ctx.fillText(totalCount.toLocaleString(), centerX, centerY);

                    ctx.fillStyle = '#94a3b8';
                    ctx.font = '500 0.75rem Inter, sans-serif';
                    ctx.fillText('CUSTOMERS', centerX, centerY + 24);
                    ctx.restore();
                }
            };

            chartInstances['riskDistChart'] = new Chart(ctxDist, {
                type: 'doughnut',
                data: {
                    labels: ['Low Risk (<30%)', 'Medium Risk (30%–70%)', 'High Risk (>70%)'],
                    datasets: [{
                        data: [riskCounts.Low, riskCounts.Medium, riskCounts.High],
                        backgroundColor: [
                            'rgba(16, 185, 129, 0.88)',
                            'rgba(245, 158, 11, 0.88)',
                            'rgba(239, 68, 68, 0.88)'
                        ],
                        borderColor: '#0f172a',
                        borderWidth: 3,
                        hoverOffset: 12,
                        hoverBorderWidth: 4,
                        hoverBorderColor: '#ffffff'
                    }]
                },
                options: {
                    cutout: '74%',
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        animateRotate: true,
                        animateScale: true,
                        duration: 1400,
                        easing: 'easeOutQuart'
                    },
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#e2e8f0',
                                padding: 18,
                                font: { size: 12, weight: '500' },
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.95)',
                            titleColor: '#ffffff',
                            bodyColor: '#e2e8f0',
                            borderColor: 'rgba(255, 255, 255, 0.12)',
                            borderWidth: 1,
                            padding: 12,
                            boxPadding: 6,
                            usePointStyle: true,
                            callbacks: {
                                label: (ctx) => {
                                    const val = ctx.parsed;
                                    const pct = totalCount > 0 ? ((val / totalCount) * 100).toFixed(1) : 0;
                                    return `  ${ctx.label}: ${val.toLocaleString()} (${pct}%)`;
                                }
                            }
                        }
                    }
                },
                plugins: [centerTextPlugin]
            });
        }

        // 2. Spend vs Churn Probability Bar (Gradients, rounded caps, animated scale)
        const spendSum = { Low: 0, Medium: 0, High: 0 };
        predictions.forEach(p => { if (spendSum[p.risk_level] !== undefined) spendSum[p.risk_level] += (p.monetary || 0); });

        const ctxSpend = document.getElementById('spendChurnChart');
        if (ctxSpend) {
            destroyChart('spendChurnChart');
            const spendContext = ctxSpend.getContext('2d');

            // Create rich vertical gradients for the bars
            const gradLow = spendContext.createLinearGradient(0, 0, 0, 300);
            gradLow.addColorStop(0, 'rgba(16, 185, 129, 0.95)');
            gradLow.addColorStop(1, 'rgba(16, 185, 129, 0.25)');

            const gradMed = spendContext.createLinearGradient(0, 0, 0, 300);
            gradMed.addColorStop(0, 'rgba(245, 158, 11, 0.95)');
            gradMed.addColorStop(1, 'rgba(245, 158, 11, 0.25)');

            const gradHigh = spendContext.createLinearGradient(0, 0, 0, 300);
            gradHigh.addColorStop(0, 'rgba(239, 68, 68, 0.95)');
            gradHigh.addColorStop(1, 'rgba(239, 68, 68, 0.25)');

            chartInstances['spendChurnChart'] = new Chart(ctxSpend, {
                type: 'bar',
                data: {
                    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                    datasets: [{
                        label: 'Total Spend ($)',
                        data: [spendSum.Low, spendSum.Medium, spendSum.High],
                        backgroundColor: [gradLow, gradMed, gradHigh],
                        borderColor: ['#10b981', '#f59e0b', '#ef4444'],
                        borderWidth: 1.5,
                        borderRadius: 10,
                        borderSkipped: false,
                        barThickness: 54,
                        hoverBorderWidth: 2.5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 1300,
                        easing: 'easeOutQuart'
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.95)',
                            titleColor: '#ffffff',
                            bodyColor: '#e2e8f0',
                            borderColor: 'rgba(255, 255, 255, 0.12)',
                            borderWidth: 1,
                            padding: 12,
                            callbacks: {
                                label: (ctx) => ` Total Revenue: $${Number(ctx.parsed.y).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                color: '#94a3b8',
                                font: { size: 11 },
                                callback: v => `$${Number(v).toLocaleString()}`
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawBorder: false
                            }
                        },
                        x: {
                            ticks: {
                                color: '#e2e8f0',
                                font: { size: 12, weight: '600' }
                            },
                            grid: { display: false }
                        }
                    }
                }
            });
        }

        // 3. Activity Trend Line (Smooth bezier curve, glow line, soft area gradient)
        const ctxTrend = document.getElementById('activityTrendChart');
        if (ctxTrend) {
            destroyChart('activityTrendChart');
            const trendData = predictions.slice(0, 25).reverse().map(p => Math.round(p.probability * 100));
            const labels = trendData.map((_, i) => `#${i + 1}`);

            const trendContext = ctxTrend.getContext('2d');
            const areaGrad = trendContext.createLinearGradient(0, 0, 0, 280);
            areaGrad.addColorStop(0, 'rgba(59, 130, 246, 0.35)');
            areaGrad.addColorStop(0.7, 'rgba(99, 102, 241, 0.12)');
            areaGrad.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

            chartInstances['activityTrendChart'] = new Chart(ctxTrend, {
                type: 'line',
                data: {
                    labels: labels.length ? labels : ['#1'],
                    datasets: [{
                        label: 'Predicted Churn Risk %',
                        data: trendData.length ? trendData : [0],
                        borderColor: '#60a5fa',
                        borderWidth: 3,
                        backgroundColor: areaGrad,
                        tension: 0.45,
                        fill: true,
                        pointBackgroundColor: '#3b82f6',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 4.5,
                        pointHoverRadius: 8,
                        pointHoverBackgroundColor: '#60a5fa',
                        pointHoverBorderColor: '#ffffff',
                        pointHoverBorderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 1500,
                        easing: 'easeOutCubic'
                    },
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#cbd5e1',
                                font: { size: 12, weight: '500' }
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.95)',
                            titleColor: '#ffffff',
                            bodyColor: '#60a5fa',
                            borderColor: 'rgba(59, 130, 246, 0.35)',
                            borderWidth: 1.5,
                            padding: 12,
                            callbacks: {
                                label: (ctx) => ` Churn Probability: ${ctx.parsed.y}%`
                            }
                        }
                    },
                    scales: {
                        y: {
                            min: 0,
                            max: 100,
                            ticks: {
                                color: '#94a3b8',
                                font: { size: 11 },
                                callback: v => `${v}%`
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawBorder: false
                            }
                        },
                        x: {
                            ticks: {
                                color: '#94a3b8',
                                font: { size: 11 }
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.03)'
                            }
                        }
                    }
                }
            });
        }
    }

    // ---- Deletion & Reset Handlers ----
    window.deletePrediction = async function(id) {
        if (!confirm("Are you sure you want to delete this prediction?")) return;
        
        try {
            const res = await fetch(`/api/prediction/${id}`, { method: 'DELETE' });
            const data = await res.json();
            if (data.status === 'success') {
                showToast('Prediction deleted successfully.', 'success');
                allPredictions = allPredictions.filter(p => String(p.id) !== String(id));
                applyFilters();
            } else {
                throw new Error(data.message);
            }
        } catch (err) {
            showToast('Error deleting: ' + err.message, 'error');
        }
    };

    window.resetCompanyData = async function() {
        if (!confirm("⚠️ WARNING: This will permanently delete ALL prediction and batch data for your company. This action cannot be undone. Are you sure?")) return;
        
        try {
            const res = await fetch('/api/reset', { method: 'DELETE' });
            const data = await res.json();
            if (data.status === 'success') {
                showToast('All data reset successfully.', 'success');
                setTimeout(() => window.location.reload(), 1500);
            } else {
                throw new Error(data.message);
            }
        } catch (err) {
            showToast('Error resetting data: ' + err.message, 'error');
        }
    };

});
