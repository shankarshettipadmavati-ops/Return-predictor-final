// ─── WASM Path Fix for GitHub Pages ───────────────────────────────────────────
// ORT tries to load .wasm files relative to the JS bundle location (CDN),
// which fails on GitHub Pages due to CORS/MIME. We override the path to load
// from the same CDN URL so the browser fetches them correctly.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';

// ─── Config ────────────────────────────────────────────────────────────────────
const MODEL_PATH = './model.onnx';

// Pre-load the session once (not on every click)
let sessionPromise = null;

function getSession() {
    if (!sessionPromise) {
        sessionPromise = ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm'],
        }).catch(err => {
            sessionPromise = null; // allow retry
            throw err;
        });
    }
    return sessionPromise;
}

// ─── UI Helpers ────────────────────────────────────────────────────────────────
function setLoading(isLoading) {
    const btn = document.getElementById('predict-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    btn.disabled = isLoading;
    btnText.textContent = isLoading ? 'Analyzing…' : 'Analyze Return Risk';
    btnLoader.classList.toggle('hidden', !isLoading);
}

function showResult(prediction, probability) {
    const resultDiv = document.getElementById('result-container');
    const resultIcon = document.getElementById('result-icon');
    const resultLabel = document.getElementById('result-label');
    const resultText = document.getElementById('result-text');
    const confidenceWrap = document.getElementById('confidence-bar-wrap');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidencePct = document.getElementById('confidence-pct');

    resultDiv.classList.remove('hidden', 'result-high', 'result-low');

    if (prediction === 1 || prediction === 1n) {
        resultIcon.textContent = '⚠️';
        resultLabel.textContent = 'High Return Risk';
        resultText.textContent = 'This order has a high likelihood of being returned.';
        resultDiv.classList.add('result-high');
    } else {
        resultIcon.textContent = '✅';
        resultLabel.textContent = 'Low Return Risk';
        resultText.textContent = 'This order is likely to be kept by the customer.';
        resultDiv.classList.add('result-low');
    }

    // Show confidence bar if probability available
    if (probability !== null) {
        const pct = Math.round(probability * 100);
        confidencePct.textContent = `${pct}%`;
        confidenceBar.style.width = `${pct}%`;
        confidenceWrap.classList.remove('hidden');
    } else {
        confidenceWrap.classList.add('hidden');
    }

    // Animate in
    resultDiv.style.opacity = '0';
    resultDiv.style.transform = 'translateY(8px)';
    requestAnimationFrame(() => {
        resultDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        resultDiv.style.opacity = '1';
        resultDiv.style.transform = 'translateY(0)';
    });
}

// ─── Core Predict Function ─────────────────────────────────────────────────────
async function predict() {
    setLoading(true);

    try {
        const session = await getSession();

        // Feature order must match Python training script
        const inputs = [
            parseFloat(document.getElementById('category').value),
            parseFloat(document.getElementById('price').value),
            parseFloat(document.getElementById('qty').value),
            parseFloat(document.getElementById('discount').value),
            parseFloat(document.getElementById('shipping').value),
            parseFloat(document.getElementById('payment').value),
            parseFloat(document.getElementById('age').value),
            parseFloat(document.getElementById('gender').value),
        ];

        const tensorData = Float32Array.from(inputs);
        const tensor = new ort.Tensor('float32', tensorData, [1, 8]);

        // Input name confirmed from model: 'float_input'
        const feeds = { float_input: tensor };
        const results = await session.run(feeds);

        // ── Output names confirmed from model ──────────────────────────────────
        // output_label  → 0 or 1
        // output_probability → { '0': p0, '1': p1 }  (map output)
        const prediction = results['output_label'].data[0];

        // Extract return probability (class 1) if available
        let returnProb = null;
        if (results['output_probability']) {
            try {
                // sklearn ZipMap output: stored as a flat sequence in ORT
                // Each pair: [p_class0, p_class1, p_class0, p_class1, ...]
                const probData = results['output_probability'].data;
                if (probData && probData.length >= 2) {
                    returnProb = probData[1]; // second value = P(class=1)
                }
            } catch (_) { /* probability display is optional */ }
        }

        showResult(prediction, returnProb);

    } catch (err) {
        console.error('Inference failed:', err);
        const resultDiv = document.getElementById('result-container');
        resultDiv.classList.remove('hidden', 'result-high', 'result-low');
        resultDiv.classList.add('result-error');
        document.getElementById('result-icon').textContent = '❌';
        document.getElementById('result-label').textContent = 'Model Load Error';
        document.getElementById('result-text').textContent =
            'Could not load model.onnx. Ensure it is committed to your GitHub repo and GitHub Pages is enabled. Check the browser console for details.';
        document.getElementById('confidence-bar-wrap').classList.add('hidden');
    } finally {
        setLoading(false);
    }
}

document.getElementById('predict-btn').addEventListener('click', predict);

// Pre-warm the session on page load (non-blocking)
getSession().catch(() => {});
