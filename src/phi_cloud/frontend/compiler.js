document.addEventListener('DOMContentLoaded', () => {
    const API_URL = 'http://localhost:8000';

    const compilerForm = document.getElementById('compiler-form');
    const statusEl = document.getElementById('compile-status');
    const resultsContainer = document.getElementById('results-container');
    const resultImage = document.getElementById('result-image');
    const lossCanvas = document.getElementById('loss-canvas');
    const verificationSummary = document.getElementById('verification-summary');
    const verificationTableBody = document.querySelector('#verification-table tbody');

    compilerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        statusEl.textContent = 'Compiling...';
        resultsContainer.classList.add('hidden');

        const apiKey = document.getElementById('api_key').value;
        const inputPorts = document.getElementById('input-ports').value.split(',').map(Number);
        const outputPorts = document.getElementById('output-ports').value.split(',').map(Number);
        const truthTable = JSON.parse(document.getElementById('truth-table').value);

        const requestBody = {
            input_ports: inputPorts,
            output_ports: outputPorts,
            truth_table: truthTable,
        };

        try {
            const response = await fetch(`${API_URL}/compile-hologram`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': apiKey,
                },
                body: JSON.stringify(requestBody),
            });

            if (response.ok) {
                const data = await response.json();
                displayResults(data);
            } else {
                const error = await response.json();
                statusEl.textContent = `Error: ${error.detail}`;
            }
        } catch (error) {
            statusEl.textContent = `Error: ${error.message}`;
        }
    });

    function displayResults(data) {
        statusEl.textContent = `Compilation Complete (Accuracy: ${data.verification.accuracy}%)`;
        resultImage.src = `${API_URL}${data.image_url}`;

        drawLossChart(data.loss_history);

        verificationSummary.textContent = `Correct: ${data.verification.correct_cases}/${data.verification.total_cases}`;

        verificationTableBody.innerHTML = '';
        data.verification_cases.forEach(c => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${JSON.stringify(c.input)}</td>
                <td>${JSON.stringify(c.target)}</td>
                <td>${JSON.stringify(c.output)}</td>
                <td class="${c.is_correct ? 'pass' : 'fail'}">${c.is_correct ? 'PASS' : 'FAIL'}</td>
            `;
            verificationTableBody.appendChild(row);
        });

        resultsContainer.classList.remove('hidden');
    }

    function drawLossChart(lossHistory) {
        const ctx = lossCanvas.getContext('2d');
        ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

        const labels = Array.from({ length: lossHistory.length }, (_, i) => i);
        const maxLoss = Math.max(...lossHistory);

        ctx.strokeStyle = '#00aaff';
        ctx.fillStyle = '#e0e0e0';
        ctx.font = '10px Roboto Mono';

        // Draw Y axis labels
        ctx.fillText(maxLoss.toExponential(2), 5, 10);
        ctx.fillText('0', 5, lossCanvas.height - 5);

        // Draw path
        ctx.beginPath();
        ctx.moveTo(40, lossCanvas.height - 10 - (Math.log1p(lossHistory[0]) / Math.log1p(maxLoss) * (lossCanvas.height - 20)));
        lossHistory.forEach((loss, index) => {
            const x = 40 + (index / (lossHistory.length - 1)) * (lossCanvas.width - 50);
            const y = lossCanvas.height - 10 - (Math.log1p(loss) / Math.log1p(maxLoss) * (lossCanvas.height - 20));
            ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
});
