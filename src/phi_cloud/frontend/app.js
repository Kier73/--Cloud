document.addEventListener('DOMContentLoaded', () => {
    const API_URL = 'http://localhost:8000';

    const uploadForm = document.getElementById('upload-form');
    const generateRandomBtn = document.getElementById('generate-random');
    const jobIdEl = document.getElementById('job-id');
    const jobStatusEl = document.getElementById('job-status');
    const downloadResultBtn = document.getElementById('download-result');
    const eyelidCanvas = document.getElementById('eyelid-canvas');
    const perfCanvas = document.getElementById('performance-canvas');
    const eyelidCtx = eyelidCanvas.getContext('2d');
    const perfCtx = perfCanvas.getContext('2d');

    let currentJobId = null;
    let pollInterval = null;

    // --- API Functions ---
    async function submitJob(formData) {
        try {
            const response = await fetch(`${API_URL}/genesis`, {
                method: 'POST',
                headers: {
                    'X-API-Key': formData.get('api_key')
                },
                body: formData
            });

            if (response.status === 202) {
                const data = await response.json();
                startPolling(data.job_id);
            } else {
                const error = await response.json();
                updateStatus(`Error: ${error.detail}`, 'error');
            }
        } catch (error) {
            updateStatus(`Error: ${error.message}`, 'error');
        }
    }

    async function checkJobStatus(jobId) {
        const apiKey = document.getElementById('api_key').value;
        try {
            const response = await fetch(`${API_URL}/collapse/${jobId}`, {
                headers: { 'X-API-Key': apiKey }
            });

            if (response.status === 200) {
                const data = await response.json();
                updateStatus(data.status);
                if (data.timings) {
                    drawPerformanceChart(data.timings);
                }
                if (data.status === 'completed') {
                    downloadResultBtn.disabled = false;
                    downloadResultBtn.onclick = () => {
                         fetch(`${API_URL}/collapse/${jobId}?download=true`, { headers: { 'X-API-Key': apiKey } })
                            .then(res => res.blob())
                            .then(blob => {
                                const url = window.URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.style.display = 'none';
                                a.href = url;
                                a.download = 'result.csv';
                                document.body.appendChild(a);
                                a.click();
                                window.URL.revokeObjectURL(url);
                            });
                    };
                    stopPolling();
                } else if (data.status === 'failed') {
                    updateStatus(`Failed: ${data.error}`, 'error');
                    stopPolling();
                }
            } else {
                const error = await response.json();
                updateStatus(`Error: ${error.detail}`, 'error');
            }
        } catch (error) {
            updateStatus(`Error: ${error.message}`, 'error');
        }
    }

    async function fetchPreview(jobId) {
        const apiKey = document.getElementById('api_key').value;
        try {
            const response = await fetch(`${API_URL}/preview/${jobId}`, {
                headers: { 'X-API-Key': apiKey }
            });

            if (response.ok) {
                const data = await response.json();
                drawHeatmap(data.heatmap);
            }
        } catch (error) {
            // It's okay if this fails, we'll just try again
            console.error('Preview fetch failed:', error);
        }
    }

    // --- UI and Canvas ---
    function updateStatus(status, type = 'info') {
        jobStatusEl.textContent = status;
        jobStatusEl.className = type; // for styling if needed
    }

    function drawHeatmap(heatmapData) {
        const sliceSize = Math.sqrt(heatmapData.length);
        const imageData = eyelidCtx.createImageData(sliceSize, sliceSize);

        for (let i = 0; i < heatmapData.length; i++) {
            const value = heatmapData[i];
            const j = i * 4;
            imageData.data[j] = value;       // Red
            imageData.data[j + 1] = value / 2; // Green
            imageData.data[j + 2] = 255 - value; // Blue
            imageData.data[j + 3] = 255;     // Alpha
        }

        // Scale the small heatmap to the larger canvas
        createImageBitmap(imageData).then(bitmap => {
            eyelidCtx.clearRect(0, 0, eyelidCanvas.width, eyelidCanvas.height);
            eyelidCtx.drawImage(bitmap, 0, 0, eyelidCanvas.width, eyelidCanvas.height);
        });
    }

    function drawPerformanceChart(timings) {
        const { standard_cpu_time_est, holographic_time } = timings;
        const maxValue = Math.max(standard_cpu_time_est, holographic_time);

        perfCtx.clearRect(0, 0, perfCanvas.width, perfCanvas.height);

        // Draw bars
        perfCtx.fillStyle = '#00aaff';
        const holoHeight = (holographic_time / maxValue) * (perfCanvas.height - 20);
        perfCtx.fillRect(50, perfCanvas.height - holoHeight - 10, 80, holoHeight);

        perfCtx.fillStyle = '#ff00ff';
        const cpuHeight = (standard_cpu_time_est / maxValue) * (perfCanvas.height - 20);
        perfCtx.fillRect(150, perfCanvas.height - cpuHeight - 10, 80, cpuHeight);

        // Draw labels
        perfCtx.fillStyle = '#e0e0e0';
        perfCtx.font = '12px Roboto Mono';
        perfCtx.fillText('ΦΦ-Cloud', 55, perfCanvas.height - holoHeight - 15);
        perfCtx.fillText(`${holographic_time.toFixed(2)}s`, 55, perfCanvas.height - holoHeight - 30);

        perfCtx.fillText('Std. CPU (est)', 155, perfCanvas.height - cpuHeight - 15);
        perfCtx.fillText(`${standard_cpu_time_est.toFixed(2)}s`, 155, perfCanvas.height - cpuHeight - 30);
    }

    function startPolling(jobId) {
        currentJobId = jobId;
        jobIdEl.textContent = jobId;
        updateStatus('Running...');
        downloadResultBtn.disabled = true;

        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(() => {
            checkJobStatus(currentJobId);
            fetchPreview(currentJobId);
        }, 2000);
    }

    function stopPolling() {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = null;
    }

    async function submitRandomJob() {
        const nSize = document.getElementById('n_size').value;
        const apiKey = document.getElementById('api_key').value;
        const formData = new FormData();
        formData.append('n_size', nSize);

        try {
            const response = await fetch(`${API_URL}/genesis/random`, {
                method: 'POST',
                headers: {
                    'X-API-Key': apiKey
                },
                body: formData
            });

            if (response.status === 202) {
                const data = await response.json();
                startPolling(data.job_id);
            } else {
                const error = await response.json();
                updateStatus(`Error: ${error.detail}`, 'error');
            }
        } catch (error) {
            updateStatus(`Error: ${error.message}`, 'error');
        }
    }

    // --- Event Listeners ---
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);
        submitJob(formData);
    });

    generateRandomBtn.addEventListener('click', () => {
        submitRandomJob();
    });
});
