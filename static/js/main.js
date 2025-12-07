// AMPidentifier Web Portal - Interactive Features

// Subtle mouse tracking for gentle glow effect
document.addEventListener('mousemove', (e) => {
    const cards = document.querySelectorAll('.glass-card');

    cards.forEach(card => {
        const rect = card.getBoundingClientRect();

        // Only apply subtle effect when hovering over the card
        if (e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top && e.clientY <= rect.bottom) {
            card.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        } else {
            card.style.borderColor = '';
        }
    });
});

// Prediction form handler
const predictionForm = document.getElementById('prediction-form');
if (predictionForm) {
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const fastaSequence = document.getElementById('fasta-input').value;
        const modelChoice = document.getElementById('model-select').value;

        // Show loading state
        showLoading();
        hideResults();
        hideError();

        try {
            const formData = new FormData();
            formData.append('fasta_sequence', fastaSequence);
            formData.append('model', modelChoice);

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            // Display results
            displayResults(data);
            hideLoading();

        } catch (error) {
            console.error('Prediction error:', error);
            showError(error.message);
            hideLoading();
        }
    });
}

// Load example FASTA
const loadExampleBtn = document.getElementById('load-example');
if (loadExampleBtn) {
    loadExampleBtn.addEventListener('click', () => {
        const exampleFasta = loadExampleBtn.dataset.example;
        document.getElementById('fasta-input').value = exampleFasta;

        // Animate the textarea
        const textarea = document.getElementById('fasta-input');
        textarea.style.transform = 'scale(1.02)';
        setTimeout(() => {
            textarea.style.transform = '';
        }, 200);
    });
}

// Display prediction results
function displayResults(data) {
    const resultsContainer = document.getElementById('results-container');
    const predictionsTable = document.getElementById('predictions-table');
    const featuresTable = document.getElementById('features-table');

    // Clear previous results
    predictionsTable.innerHTML = '';
    featuresTable.innerHTML = '';

    // Create predictions table
    if (data.predictions && data.predictions.length > 0) {
        const predTable = createTable(data.predictions, data.predictions_columns);
        predictionsTable.appendChild(predTable);
    }

    // Create features table (show first 10 columns for readability)
    if (data.features && data.features.length > 0) {
        const displayColumns = data.features_columns.slice(0, 10);
        const limitedFeatures = data.features.map(row => {
            const limited = {};
            displayColumns.forEach(col => {
                limited[col] = row[col];
            });
            return limited;
        });
        const featTable = createTable(limitedFeatures, displayColumns);
        featuresTable.appendChild(featTable);

        // Add note about truncated columns
        if (data.features_columns.length > 10) {
            const note = document.createElement('p');
            note.className = 'text-muted mt-2';
            note.textContent = `Showing 10 of ${data.features_columns.length} features. Download CSV for complete data.`;
            featuresTable.appendChild(note);
        }
    }

    // Store data for download
    window.currentPredictions = data.predictions;
    window.currentFeatures = data.features;

    // Show results
    resultsContainer.classList.remove('hidden');

    // Smooth scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create HTML table from data
function createTable(data, columns) {
    const table = document.createElement('table');
    table.className = 'data-table';

    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body
    const tbody = document.createElement('tbody');
    data.forEach(row => {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            let value = row[col];

            // Format numbers
            if (typeof value === 'number') {
                value = value.toFixed(4);
            }

            // Highlight AMP predictions
            if (col.toLowerCase().includes('prediction') || col.toLowerCase().includes('amp')) {
                if (value === 1 || value === 'AMP' || value === true) {
                    td.innerHTML = `<span style="color: #43e97b; font-weight: 600;">✓ AMP</span>`;
                } else if (value === 0 || value === 'Non-AMP' || value === false) {
                    td.innerHTML = `<span style="color: #f5576c;">✗ Non-AMP</span>`;
                } else {
                    td.textContent = value;
                }
            } else {
                td.textContent = value;
            }

            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    return table;
}

// Download functions
function downloadPredictions(format) {
    if (!window.currentPredictions) {
        showError('No predictions to download');
        return;
    }

    if (format === 'csv') {
        downloadCSV(window.currentPredictions, 'predictions');
    }
}

function downloadFeatures(format) {
    if (!window.currentFeatures) {
        showError('No features to download');
        return;
    }

    if (format === 'csv') {
        downloadCSV(window.currentFeatures, 'features');
    }
}

function downloadCSV(data, filename) {
    if (!data || data.length === 0) return;

    // Convert to CSV
    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];

    data.forEach(row => {
        const values = headers.map(header => {
            const value = row[header];
            // Escape quotes and wrap in quotes if contains comma
            const escaped = String(value).replace(/"/g, '""');
            return escaped.includes(',') ? `"${escaped}"` : escaped;
        });
        csvRows.push(values.join(','));
    });

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ampidentifier_${filename}_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// UI Helper Functions
function showLoading() {
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
        loadingEl.classList.remove('hidden');
    }
}

function hideLoading() {
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
        loadingEl.classList.add('hidden');
    }
}

function showResults() {
    const resultsEl = document.getElementById('results-container');
    if (resultsEl) {
        resultsEl.classList.remove('hidden');
    }
}

function hideResults() {
    const resultsEl = document.getElementById('results-container');
    if (resultsEl) {
        resultsEl.classList.add('hidden');
    }
}

function showError(message) {
    const errorEl = document.getElementById('error-message');
    if (errorEl) {
        errorEl.textContent = message;
        errorEl.classList.remove('hidden');
        errorEl.scrollIntoView({ behavior: 'smooth' });
    }
}

function hideError() {
    const errorEl = document.getElementById('error-message');
    if (errorEl) {
        errorEl.classList.add('hidden');
    }
}

// Animated counter for statistics
function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target.toFixed(2);
            clearInterval(timer);
        } else {
            element.textContent = current.toFixed(2);
        }
    }, 16);
}

// Initialize counters on page load
document.addEventListener('DOMContentLoaded', () => {
    const counters = document.querySelectorAll('[data-counter]');
    counters.forEach(counter => {
        const target = parseFloat(counter.dataset.counter);
        animateCounter(counter, target);
    });
});

// Add ripple effect to buttons
document.querySelectorAll('.btn').forEach(button => {
    button.addEventListener('click', function (e) {
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;

        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple');

        this.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
    });
});
