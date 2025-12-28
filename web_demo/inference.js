/**
 * Essay Prompt GPT-ability Analyzer
 * 
 * This module handles the inference pipeline for predicting how "GPT-able" an essay prompt is.
 * 
 * Pipeline:
 * 1. Text -> Transformers.js (all-MiniLM-L12-v2) -> 384-dim embedding
 * 2. Embedding -> ONNX Runtime Web (regression head) -> Score [0, 2]
 * 
 * Lower scores = Human-distinctive prompts (LLMs struggle to imitate)
 * Higher scores = GPT-able prompts (LLMs can effectively imitate humans)
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2';

// ONNX Runtime Web is loaded via script tag in index.html and available as window.ort
// This is the recommended approach for browser usage without a bundler
const ort = window.ort;

// Configure Transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

// Model configuration
const ENCODER_MODEL = 'Xenova/all-MiniLM-L12-v2';
const REGRESSOR_PATH = './models/regressor.onnx';

// Global model instances
let extractor = null;
let regressorSession = null;
let isModelLoaded = false;

// DOM elements
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const progressBar = document.getElementById('progressBar');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const promptInput = document.getElementById('promptInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultContainer = document.getElementById('resultContainer');
const scoreValue = document.getElementById('scoreValue');
const scoreLabel = document.getElementById('scoreLabel');
const scoreBarFill = document.getElementById('scoreBarFill');
const interpretation = document.getElementById('interpretation');
const samplePrompts = document.getElementById('samplePrompts');

/**
 * Update loading progress UI
 */
function updateProgress(progress, text) {
    progressBar.style.width = `${progress}%`;
    loadingText.textContent = text;
}

/**
 * Update status bar UI
 */
function updateStatus(isReady, text) {
    statusText.textContent = text;
    if (isReady) {
        statusDot.classList.remove('loading');
        statusDot.classList.add('ready');
    } else {
        statusDot.classList.add('loading');
        statusDot.classList.remove('ready');
    }
}

/**
 * Load the sentence transformer model and regression head
 */
async function loadModels() {
    loadingOverlay.classList.add('visible');
    
    try {
        // Load sentence transformer (with progress tracking)
        updateProgress(10, 'Loading sentence transformer model...');
        updateStatus(false, 'Loading models...');
        
        extractor = await pipeline('feature-extraction', ENCODER_MODEL, {
            dtype: 'q8',  // Use quantized model for faster loading and inference
            progress_callback: (progress) => {
                if (progress.status === 'progress') {
                    const pct = Math.round(progress.progress);
                    updateProgress(10 + pct * 0.7, `Loading encoder: ${pct}%`);
                }
            }
        });
        
        updateProgress(85, 'Loading regression head...');
        
        // Configure ONNX Runtime for WebAssembly
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        
        // Load regression head
        regressorSession = await ort.InferenceSession.create(REGRESSOR_PATH, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        updateProgress(100, 'Models loaded!');
        isModelLoaded = true;
        
        // Update UI
        analyzeBtn.disabled = false;
        updateStatus(true, 'Ready - Models loaded');
        
        // Hide loading overlay after a short delay
        setTimeout(() => {
            loadingOverlay.classList.remove('visible');
        }, 500);
        
    } catch (error) {
        console.error('Error loading models:', error);
        updateProgress(0, `Error: ${error.message}`);
        updateStatus(false, 'Error loading models');
        loadingText.style.color = '#e94560';
    }
}

/**
 * Run inference on a text prompt
 * @param {string} text - The prompt text to analyze
 * @returns {Promise<number>} - The predicted score (0-2)
 */
async function predict(text) {
    if (!isModelLoaded) {
        throw new Error('Models not loaded');
    }
    
    // Get embeddings from sentence transformer
    // The pipeline returns embeddings with mean pooling
    const output = await extractor(text, { 
        pooling: 'mean', 
        normalize: true 
    });
    
    // Convert to Float32Array for ONNX Runtime
    const embeddingData = new Float32Array(output.data);
    
    // Create input tensor (batch_size=1, embedding_dim=384)
    const inputTensor = new ort.Tensor('float32', embeddingData, [1, 384]);
    
    // Run regression head
    const results = await regressorSession.run({ 
        embedding: inputTensor 
    });
    
    // Get the score and clamp to [0, 2] range
    const rawScore = results.score.data[0];
    const clampedScore = Math.max(0, Math.min(2, rawScore));
    
    return clampedScore;
}

/**
 * Get interpretation text based on score
 * @param {number} score - The predicted score (0-2)
 * @returns {object} - Label, color, and interpretation text
 */
function getInterpretation(score) {
    // in practice, the range is much tighter
    // above 0.5 is most gpt-able
    // below 0.4 is most human-distinctive
    // in between is a mix

    if (score < 0.4) {
        return {
            label: 'Human-Distinctive',
            color: '#4ecca3',
            text: `This prompt elicits distinctive human responses that LLMs struggle to replicate effectively. 
                   Prompts like this often involve philosophical depth, specific historical/cultural contexts, 
                   or complex emotional scenarios that require genuine human experience to address authentically.`
        };
    } else if (score < 0.5) {
        return {
            label: 'Mixed',
            color: '#ffc947',
            text: `This prompt shows moderate differentiation between human and LLM responses. 
                   While an LLM can produce reasonable responses, careful reading may reveal differences 
                   in depth, authenticity, or engagement compared to human writing.`
        };
    } else {
        return {
            label: 'GPT-able',
            color: '#e94560',
            text: `This prompt is vulnerable to LLM completion - AI can effectively imitate human responses. 
                   Generic dialogue prompts, open-ended scenarios without specific constraints, and prompts 
                   lacking unique context tend to fall in this category. Consider adding more specificity 
                   or philosophical depth to make responses more distinctively human.`
        };
    }
}

/**
 * Display the analysis result
 * @param {number} score - The predicted score (0-2)
 */
function displayResult(score) {
    const interp = getInterpretation(score);
    
    // Update score display
    scoreValue.textContent = score.toFixed(2);
    scoreValue.style.color = interp.color;
    
    scoreLabel.textContent = interp.label;
    scoreLabel.style.color = interp.color;
    
    // Update progress bar
    const percentage = ((score - 0.2) / 0.4) * 100;
    scoreBarFill.style.width = `${percentage}%`;
    scoreBarFill.style.background = interp.color;
    
    // Update interpretation text
    interpretation.innerHTML = `<p>${interp.text}</p>`;
    
    // Show result container
    resultContainer.classList.add('visible');
    
    // Scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Handle analyze button click
 */
async function handleAnalyze() {
    const text = promptInput.value.trim();
    
    if (!text) {
        alert('Please enter a prompt to analyze.');
        return;
    }
    
    // Disable button and show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span>Analyzing...</span>';
    updateStatus(false, 'Analyzing prompt...');
    
    try {
        const score = await predict(text);
        displayResult(score);
        updateStatus(true, 'Analysis complete');
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Error during analysis: ${error.message}`);
        updateStatus(true, 'Ready - Models loaded');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span>Analyze Prompt</span>';
    }
}

/**
 * Handle clear button click
 */
function handleClear() {
    promptInput.value = '';
    resultContainer.classList.remove('visible');
}

/**
 * Handle sample prompt click
 */
function handleSampleClick(event) {
    const button = event.target.closest('.sample-prompt');
    if (button) {
        const prompt = button.dataset.prompt;
        promptInput.value = prompt;
        promptInput.focus();
        
        // Auto-analyze if models are loaded
        if (isModelLoaded) {
            handleAnalyze();
        }
    }
}

// Event listeners
analyzeBtn.addEventListener('click', handleAnalyze);
clearBtn.addEventListener('click', handleClear);
samplePrompts.addEventListener('click', handleSampleClick);

// Handle Enter key in textarea (Ctrl+Enter to analyze)
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        if (!analyzeBtn.disabled) {
            handleAnalyze();
        }
    }
});

// Load models on page load
loadModels();
