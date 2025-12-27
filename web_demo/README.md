# Essay Prompt GPT-ability Analyzer - Web Demo

A client-side web application that predicts how "GPT-able" an essay or creative writing prompt is. This tool helps educators design assignments that are more resilient to LLM-based completion.

## What It Does

This tool analyzes writing prompts and predicts how well LLMs can imitate human responses:

- **Low scores (0-0.7)**: "Human-Distinctive" - Prompts that elicit unique human responses that LLMs struggle to replicate
- **Medium scores (0.7-1.3)**: "Mixed" - Moderate differentiation between human and LLM responses
- **High scores (1.3-2.0)**: "GPT-able" - Prompts where LLMs can effectively imitate human writing

## How to Run Locally

The demo requires a local HTTP server because it uses ES modules. You cannot simply open the HTML file directly in a browser.

### Option 1: Python (recommended)

```bash
cd web_demo
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Option 2: Node.js

```bash
cd web_demo
npx serve .
```

### Option 3: VS Code Live Server

If you're using VS Code, install the "Live Server" extension and right-click on `index.html` -> "Open with Live Server".

## Technical Architecture

The inference pipeline runs entirely in your browser using WebAssembly:

```
User Input (text)
    │
    ▼
Transformers.js (all-MiniLM-L12-v2)
    │ 384-dimensional embedding
    ▼
ONNX Runtime Web (regression head)
    │
    ▼
Score [0, 2]
```

### Models Used

| Model | Size | Source |
|-------|------|--------|
| Sentence Encoder | ~34MB (quantized) | [Xenova/all-MiniLM-L12-v2](https://huggingface.co/Xenova/all-MiniLM-L12-v2) |
| Regression Head | ~62KB (quantized) | Custom trained (see main project) |

### Dependencies (loaded via CDN)

- [@huggingface/transformers](https://www.npmjs.com/package/@huggingface/transformers) v3.1.2
- [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) v1.20.1

## Performance

- **Initial load**: ~5-10 seconds (downloads ~34MB encoder model)
- **Subsequent loads**: <1 second (models cached in IndexedDB)
- **Inference time**: ~50-100ms per prediction

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 90+
- Safari 15+
- Edge 90+

Requires WebAssembly support (available in all modern browsers).

## Integration with Other Projects

To integrate this into your own project:

1. Copy the `models/` directory and `inference.js` to your project
2. Import the inference module:
   ```javascript
   import { pipeline, env } from '@huggingface/transformers';
   import * as ort from 'onnxruntime-web';
   ```
3. Use the `predict()` function from `inference.js` or adapt it to your needs

### Minimal Example

```javascript
import { pipeline } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';

// Load models
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L12-v2', { dtype: 'q8' });
const session = await ort.InferenceSession.create('./models/regressor_quantized.onnx');

// Run inference
async function predict(text) {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    const tensor = new ort.Tensor('float32', new Float32Array(output.data), [1, 384]);
    const result = await session.run({ embedding: tensor });
    return Math.max(0, Math.min(2, result.score.data[0]));
}

const score = await predict("Your prompt here...");
console.log(`GPT-ability score: ${score}`);
```

## File Structure

```
web_demo/
├── index.html              # Main HTML page with UI
├── inference.js            # JavaScript inference pipeline
├── models/
│   ├── regressor.onnx          # Full precision model (226KB)
│   └── regressor_quantized.onnx # Quantized model (62KB, used by default)
└── README.md               # This file
```

## Re-exporting the Model

If you retrain the model and need to re-export:

```bash
# From the main project directory
uv run python export_onnx.py --checkpoint model_output/best_model.pt --output-dir web_demo/models
```

## Research Background

This demo is based on research analyzing essay prompts by their ability to differentiate LLM and human responses. For more details, see:

- [Paper (PDF)](../paper.pdf)
- [GitHub Repository](https://github.com/iamr-gh/essay-gptability)

## License

MIT License - See the main project repository for details.
