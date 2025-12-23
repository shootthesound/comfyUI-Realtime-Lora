import { app } from "../../../scripts/app.js";

// Extension for dynamic image inputs (training nodes)
app.registerExtension({
    name: "RealtimeLoraTrainer.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to trainer nodes only
        if (!["RealtimeLoraTrainer", "SDXLLoraTrainer", "SD15LoraTrainer", "MusubiZImageLoraTrainer", "MusubiQwenImageLoraTrainer", "MusubiWanLoraTrainer"].includes(nodeData.name)) {
            return;
        }

        nodeType.prototype.onNodeCreated = function () {
            // Create the button widget
            const button = this.addWidget("button", "Update inputs", null, () => {
                if (!this.inputs) {
                    this.inputs = [];
                }

                const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
                const num_image_inputs = this.inputs.filter(input => input.type === "IMAGE").length;

                if (target_number_of_inputs === num_image_inputs) return;

                if (target_number_of_inputs < num_image_inputs) {
                    // Remove excess inputs (from the end) - both image and caption
                    for (let i = num_image_inputs; i > target_number_of_inputs; i--) {
                        // Find and remove caption_N first, then image_N
                        for (let j = this.inputs.length - 1; j >= 0; j--) {
                            if (this.inputs[j].name === `caption_${i}`) {
                                this.removeInput(j);
                                break;
                            }
                        }
                        for (let j = this.inputs.length - 1; j >= 0; j--) {
                            if (this.inputs[j].name === `image_${i}`) {
                                this.removeInput(j);
                                break;
                            }
                        }
                    }
                } else {
                    // Add new inputs interleaved (image then caption for each)
                    for (let i = num_image_inputs + 1; i <= target_number_of_inputs; ++i) {
                        this.addInput(`image_${i}`, "IMAGE");
                        this.addInput(`caption_${i}`, "STRING");
                    }
                }
            });

            // Move button to right after inputcount widget
            const inputcountIndex = this.widgets.findIndex(w => w.name === "inputcount");
            if (inputcountIndex !== -1) {
                // Remove button from end and insert after inputcount
                this.widgets.pop();
                this.widgets.splice(inputcountIndex + 1, 0, button);
            }
        };
    }
});

// ============================================================================
// STRENGTH SCHEDULE PRESETS (must match Python SCHEDULE_PRESETS)
// ============================================================================
const SCHEDULE_PRESETS = {
    // === No Schedule ===
    "Custom": "",
    "Constant 1.0 (No Change)": "0:1, 1:1",

    // === Basic Fades ===
    "Linear In (0→1)": "0:0, 1:1",
    "Linear Out (1→0)": "0:1, 1:0",

    // === Ease Curves ===
    "Ease In (slow start)": "0:0, 0.3:0.1, 0.7:0.5, 1:1",
    "Ease Out (slow end)": "0:1, 0.3:0.5, 0.7:0.9, 1:0",
    "Ease In-Out": "0:0, 0.3:0.1, 0.7:0.9, 1:1",

    // === Bell Curves ===
    "Bell Curve (peak middle)": "0:0, 0.5:1, 1:0",
    "Wide Bell": "0:0, 0.3:0.8, 0.7:0.8, 1:0",
    "Sharp Bell": "0:0, 0.4:0.2, 0.5:1, 0.6:0.2, 1:0",

    // === Structure LoRA Favorites ===
    "High Start → Cut Low": "0:1, 0.3:1, 0.35:0.2, 1:0.2",
    "High Start → Cut Off": "0:1, 0.3:1, 0.35:0, 1:0",
    "High Early → Fade": "0:1, 0.2:0.8, 0.5:0.3, 1:0",

    // === Detail LoRA Favorites ===
    "Low Start → Ramp Late": "0:0, 0.6:0.1, 0.8:0.7, 1:1",
    "Off → Kick In Late": "0:0, 0.7:0, 0.75:0.8, 1:1",
    "Low → Boost End": "0:0.2, 0.5:0.2, 0.7:0.6, 1:1",

    // === Step Functions ===
    "Step Up Mid": "0:0.2, 0.45:0.2, 0.55:0.8, 1:0.8",
    "Step Down Mid": "0:0.8, 0.45:0.8, 0.55:0.2, 1:0.2",
    "Two Steps Up": "0:0, 0.3:0, 0.35:0.5, 0.65:0.5, 0.7:1, 1:1",

    // === Pulses ===
    "Pulse Early": "0:1, 0.25:1, 0.35:0.2, 1:0.2",
    "Pulse Mid": "0:0.2, 0.4:0.2, 0.45:1, 0.55:1, 0.6:0.2, 1:0.2",
    "Pulse Late": "0:0.2, 0.65:0.2, 0.75:1, 1:1",

    // === Constant (for testing) ===
    "Constant Half": "0:0.5, 1:0.5",
    "Constant Low": "0:0.3, 1:0.3",

    // =========================================================================
    // INVERTED VERSIONS (same order as above)
    // =========================================================================

    // === Basic Fades (Inverted) ===
    "INV: Linear In (1→0)": "0:1, 1:0",
    "INV: Linear Out (0→1)": "0:0, 1:1",

    // === Ease Curves (Inverted) ===
    "INV: Ease In": "0:1, 0.3:0.9, 0.7:0.5, 1:0",
    "INV: Ease Out": "0:0, 0.3:0.5, 0.7:0.1, 1:1",
    "INV: Ease In-Out": "0:1, 0.3:0.9, 0.7:0.1, 1:0",

    // === Bell Curves (Inverted) ===
    "INV: Bell (dip middle)": "0:1, 0.5:0, 1:1",
    "INV: Wide Bell": "0:1, 0.3:0.2, 0.7:0.2, 1:1",
    "INV: Sharp Bell": "0:1, 0.4:0.8, 0.5:0, 0.6:0.8, 1:1",

    // === Structure Inverted ===
    "INV: Low Start → Boost High": "0:0, 0.3:0, 0.35:0.8, 1:0.8",
    "INV: Low Start → Full On": "0:0, 0.3:0, 0.35:1, 1:1",
    "INV: Low Early → Build": "0:0, 0.2:0.2, 0.5:0.7, 1:1",

    // === Detail Inverted ===
    "INV: High Start → Drop Late": "0:1, 0.6:0.9, 0.8:0.3, 1:0",
    "INV: Full → Cut Off Late": "0:1, 0.7:1, 0.75:0.2, 1:0",
    "INV: High → Drop End": "0:0.8, 0.5:0.8, 0.7:0.4, 1:0",

    // === Step Functions (Inverted) ===
    "INV: Step Down Mid": "0:0.8, 0.45:0.8, 0.55:0.2, 1:0.2",
    "INV: Step Up Mid": "0:0.2, 0.45:0.2, 0.55:0.8, 1:0.8",
    "INV: Two Steps Down": "0:1, 0.3:1, 0.35:0.5, 0.65:0.5, 0.7:0, 1:0",

    // === Pulses (Inverted) ===
    "INV: Dip Early": "0:0, 0.25:0, 0.35:0.8, 1:0.8",
    "INV: Dip Mid": "0:0.8, 0.4:0.8, 0.45:0, 0.55:0, 0.6:0.8, 1:0.8",
    "INV: Dip Late": "0:0.8, 0.65:0.8, 0.75:0, 1:0",
};

// Impact score color gradient (10% ranges, blue=low to red=high)
function getImpactColor(score) {
    // score is 0-100
    if (score < 10) return "#0066ff";      // Deep blue
    if (score < 20) return "#0088ff";      // Blue
    if (score < 30) return "#00aaff";      // Light blue
    if (score < 40) return "#00cccc";      // Cyan
    if (score < 50) return "#00cc66";      // Teal/green
    if (score < 60) return "#88cc00";      // Yellow-green
    if (score < 70) return "#cccc00";      // Yellow
    if (score < 80) return "#ff9900";      // Orange
    if (score < 90) return "#ff6600";      // Orange-red
    return "#ff3300";                       // Red
}

// Get analysis data from connected analysis_json input or stored on node
function getAnalysisFromInput(node) {
    // First check if this node has stored analysis data
    if (node._analysisData) {
        return node._analysisData;
    }

    if (!node.inputs) return null;

    // Find the analysis_json input
    const analysisInput = node.inputs.find(input => input.name === "analysis_json");
    if (!analysisInput || !analysisInput.link) return null;

    // Get the link and find the source node
    const link = node.graph?.links?.[analysisInput.link];
    if (!link) return null;

    const sourceNode = node.graph?.getNodeById(link.origin_id);
    if (!sourceNode) return null;

    // Check if source node (analyzer) has stored analysis data
    if (sourceNode._lastAnalysisData) {
        return sourceNode._lastAnalysisData;
    }

    return null;
}

// Extension for LoRA Analyzer to store analysis data after execution
app.registerExtension({
    name: "LoRAAnalyzer.StoreAnalysis",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to both V1 and V2 analyzers
        if (nodeData.name !== "LoRALoaderWithAnalysis" && nodeData.name !== "LoRALoaderWithAnalysisV2") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(output) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }
            // Store the analysis_json output for connected nodes to read
            if (output && output.analysis_json && output.analysis_json[0]) {
                try {
                    this._lastAnalysisData = JSON.parse(output.analysis_json[0]);
                    // Trigger redraw of connected nodes
                    if (this.graph) {
                        this.graph.setDirtyCanvas(true);
                    }
                } catch (e) {
                    // Silent fail - analysis coloring is optional
                }
            }
        };
    }
});

// Preset definitions for each selective loader
const SELECTIVE_LOADER_PRESETS = {
    "SDXLSelectiveLoRALoader": {
        blocks: ["text_encoder_1", "text_encoder_2", "input_4", "input_5", "input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3", "output_4", "output_5", "other_weights"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "UNet Only": { enabled: ["input_4", "input_5", "input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3", "output_4", "output_5", "other_weights"], strength: 1.0 },
            "High Impact": { enabled: ["input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2"], strength: 1.0 },
            "Text Encoders Only": { enabled: ["text_encoder_1", "text_encoder_2"], strength: 1.0 },
            "Decoders Only": { enabled: ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"], strength: 1.0 },
            "Encoders Only": { enabled: ["input_4", "input_5", "input_7", "input_8"], strength: 1.0 },
            "Style Focus": { enabled: ["output_1", "output_2"], strength: 1.0 },
            "Composition Focus": { enabled: ["input_8", "unet_mid", "output_0"], strength: 1.0 },
            "Face Focus": { enabled: ["input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3"], strength: 1.0 },
        }
    },
    "ZImageSelectiveLoRALoader": {
        blocks: [...Array.from({length: 30}, (_, i) => `layer_${i}`), "other_weights"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (20-29)": { enabled: [...Array.from({length: 10}, (_, i) => `layer_${i + 20}`), "other_weights"], strength: 1.0 },
            "Mid-Late (15-29)": { enabled: [...Array.from({length: 15}, (_, i) => `layer_${i + 15}`), "other_weights"], strength: 1.0 },
            "Skip Early (10-29)": { enabled: [...Array.from({length: 20}, (_, i) => `layer_${i + 10}`), "other_weights"], strength: 1.0 },
            "Mid Only (10-19)": { enabled: Array.from({length: 10}, (_, i) => `layer_${i + 10}`), strength: 1.0 },
            "Early Only (0-9)": { enabled: Array.from({length: 10}, (_, i) => `layer_${i}`), strength: 1.0 },
            "Peak Impact (18-25)": { enabled: Array.from({length: 8}, (_, i) => `layer_${i + 18}`), strength: 1.0 },
            "Face Priority (16-24)": { enabled: Array.from({length: 9}, (_, i) => `layer_${i + 16}`), strength: 1.0 },
            "Face Priority Aggressive (14-25)": { enabled: Array.from({length: 12}, (_, i) => `layer_${i + 14}`), strength: 1.0 },
            "Evens Only": { enabled: Array.from({length: 15}, (_, i) => `layer_${i * 2}`), strength: 1.0 },
            "Odds Only": { enabled: Array.from({length: 15}, (_, i) => `layer_${i * 2 + 1}`), strength: 1.0 },
        }
    },
    "FLUXSelectiveLoRALoader": {
        blocks: [
            ...Array.from({length: 19}, (_, i) => `double_${i}`),
            ...Array.from({length: 38}, (_, i) => `single_${i}`),
            "other_weights"
        ],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Double Blocks Only": { enabled: [...Array.from({length: 19}, (_, i) => `double_${i}`), "other_weights"], strength: 1.0 },
            "Single Blocks Only": { enabled: [...Array.from({length: 38}, (_, i) => `single_${i}`), "other_weights"], strength: 1.0 },
            "High Impact Double": { enabled: Array.from({length: 13}, (_, i) => `double_${i + 6}`), strength: 1.0 },
            "Core Double": { enabled: Array.from({length: 10}, (_, i) => `double_${i + 8}`), strength: 1.0 },
            "Face Focus": { enabled: ["double_7", "double_12", "double_16", "single_7", "single_12", "single_16", "single_20"], strength: 1.0 },
            "Face Aggressive": { enabled: ["double_4", "double_7", "double_8", "double_12", "double_15", "double_16", "single_4", "single_7", "single_8", "single_12", "single_15", "single_16", "single_19", "single_20"], strength: 1.0 },
            "Style Only (No Face)": {
                enabled: [
                    ...Array.from({length: 19}, (_, i) => `double_${i}`),
                    ...Array.from({length: 38}, (_, i) => `single_${i}`)
                ].filter(b => !["double_7", "double_12", "double_16", "single_7", "single_12", "single_16", "single_20"].includes(b)),
                strength: 1.0
            },
            "Evens Only": { enabled: [...Array.from({length: 10}, (_, i) => `double_${i * 2}`), ...Array.from({length: 19}, (_, i) => `single_${i * 2}`)], strength: 1.0 },
            "Odds Only": { enabled: [...Array.from({length: 9}, (_, i) => `double_${i * 2 + 1}`), ...Array.from({length: 19}, (_, i) => `single_${i * 2 + 1}`)], strength: 1.0 },
        }
    },
    "WanSelectiveLoRALoader": {
        blocks: [...Array.from({length: 40}, (_, i) => `block_${i}`), "other_weights"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (30-39)": { enabled: [...Array.from({length: 10}, (_, i) => `block_${i + 30}`), "other_weights"], strength: 1.0 },
            "Mid-Late (20-39)": { enabled: [...Array.from({length: 20}, (_, i) => `block_${i + 20}`), "other_weights"], strength: 1.0 },
            "Skip Early (10-39)": { enabled: [...Array.from({length: 30}, (_, i) => `block_${i + 10}`), "other_weights"], strength: 1.0 },
            "Mid Only (15-25)": { enabled: Array.from({length: 11}, (_, i) => `block_${i + 15}`), strength: 1.0 },
            "Early Only (0-19)": { enabled: Array.from({length: 20}, (_, i) => `block_${i}`), strength: 1.0 },
            "Evens Only": { enabled: Array.from({length: 20}, (_, i) => `block_${i * 2}`), strength: 1.0 },
            "Odds Only": { enabled: Array.from({length: 20}, (_, i) => `block_${i * 2 + 1}`), strength: 1.0 },
        }
    },
    "QwenSelectiveLoRALoader": {
        blocks: [...Array.from({length: 60}, (_, i) => `block_${i}`), "other_weights"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (45-59)": { enabled: [...Array.from({length: 15}, (_, i) => `block_${i + 45}`), "other_weights"], strength: 1.0 },
            "Mid-Late (30-59)": { enabled: [...Array.from({length: 30}, (_, i) => `block_${i + 30}`), "other_weights"], strength: 1.0 },
            "Skip Early (15-59)": { enabled: [...Array.from({length: 45}, (_, i) => `block_${i + 15}`), "other_weights"], strength: 1.0 },
            "Mid Only (20-40)": { enabled: Array.from({length: 21}, (_, i) => `block_${i + 20}`), strength: 1.0 },
            "Early Only (0-29)": { enabled: Array.from({length: 30}, (_, i) => `block_${i}`), strength: 1.0 },
            "Evens Only": { enabled: Array.from({length: 30}, (_, i) => `block_${i * 2}`), strength: 1.0 },
            "Odds Only": { enabled: Array.from({length: 30}, (_, i) => `block_${i * 2 + 1}`), strength: 1.0 },
        }
    },
    // V2 Combined Analyzer + Selective Loaders
    "ZImageAnalyzerSelectiveLoaderV2": {
        blocks: [...Array.from({length: 30}, (_, i) => `layer_${i}`), "context_refiner", "noise_refiner", "final_layer", "x_embedder", "other_weights"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (20-29)": { enabled: [...Array.from({length: 10}, (_, i) => `layer_${i + 20}`), "other_weights"], strength: 1.0 },
            "Mid-Late (15-29)": { enabled: [...Array.from({length: 15}, (_, i) => `layer_${i + 15}`), "other_weights"], strength: 1.0 },
            "Skip Early (10-29)": { enabled: [...Array.from({length: 20}, (_, i) => `layer_${i + 10}`), "other_weights"], strength: 1.0 },
            "Mid Only (10-19)": { enabled: Array.from({length: 10}, (_, i) => `layer_${i + 10}`), strength: 1.0 },
            "Early Only (0-9)": { enabled: Array.from({length: 10}, (_, i) => `layer_${i}`), strength: 1.0 },
            "Peak Impact (18-25)": { enabled: Array.from({length: 8}, (_, i) => `layer_${i + 18}`), strength: 1.0 },
            "Face Priority (16-24)": { enabled: Array.from({length: 9}, (_, i) => `layer_${i + 16}`), strength: 1.0 },
            "Face Priority Aggressive (14-25)": { enabled: Array.from({length: 12}, (_, i) => `layer_${i + 14}`), strength: 1.0 },
            "Evens Only": { enabled: Array.from({length: 15}, (_, i) => `layer_${i * 2}`), strength: 1.0 },
            "Odds Only": { enabled: Array.from({length: 15}, (_, i) => `layer_${i * 2 + 1}`), strength: 1.0 },
        }
    },
    "SDXLAnalyzerSelectiveLoaderV2": {
        blocks: ["text_encoder_1", "text_encoder_2", "input_4", "input_5", "input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3", "output_4", "output_5", "other_weights"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "UNet Only": { enabled: ["input_4", "input_5", "input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3", "output_4", "output_5", "other_weights"], strength: 1.0 },
            "High Impact": { enabled: ["input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2"], strength: 1.0 },
            "Text Encoders Only": { enabled: ["text_encoder_1", "text_encoder_2"], strength: 1.0 },
            "Decoders Only": { enabled: ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"], strength: 1.0 },
            "Encoders Only": { enabled: ["input_4", "input_5", "input_7", "input_8"], strength: 1.0 },
            "Style Focus": { enabled: ["output_1", "output_2"], strength: 1.0 },
            "Composition Focus": { enabled: ["input_8", "unet_mid", "output_0"], strength: 1.0 },
            "Face Focus": { enabled: ["input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3"], strength: 1.0 },
        }
    },
    "FLUXAnalyzerSelectiveLoaderV2": {
        blocks: [
            ...Array.from({length: 19}, (_, i) => `double_${i}`),
            ...Array.from({length: 38}, (_, i) => `single_${i}`),
            "other_weights"
        ],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Double Blocks Only": { enabled: [...Array.from({length: 19}, (_, i) => `double_${i}`), "other_weights"], strength: 1.0 },
            "Single Blocks Only": { enabled: [...Array.from({length: 38}, (_, i) => `single_${i}`), "other_weights"], strength: 1.0 },
            "High Impact Double": { enabled: Array.from({length: 13}, (_, i) => `double_${i + 6}`), strength: 1.0 },
            "Core Double": { enabled: Array.from({length: 10}, (_, i) => `double_${i + 8}`), strength: 1.0 },
            "Face Focus": { enabled: ["double_7", "double_12", "double_16", "single_7", "single_12", "single_16", "single_20"], strength: 1.0 },
            "Face Aggressive": { enabled: ["double_4", "double_7", "double_8", "double_12", "double_15", "double_16", "single_4", "single_7", "single_8", "single_12", "single_15", "single_16", "single_19", "single_20"], strength: 1.0 },
            "Style Only (No Face)": {
                enabled: [
                    ...Array.from({length: 19}, (_, i) => `double_${i}`),
                    ...Array.from({length: 38}, (_, i) => `single_${i}`)
                ].filter(b => !["double_7", "double_12", "double_16", "single_7", "single_12", "single_16", "single_20"].includes(b)),
                strength: 1.0
            },
            "Evens Only": { enabled: [...Array.from({length: 10}, (_, i) => `double_${i * 2}`), ...Array.from({length: 19}, (_, i) => `single_${i * 2}`)], strength: 1.0 },
            "Odds Only": { enabled: [...Array.from({length: 9}, (_, i) => `double_${i * 2 + 1}`), ...Array.from({length: 19}, (_, i) => `single_${i * 2 + 1}`)], strength: 1.0 },
        }
    },
    "WanAnalyzerSelectiveLoaderV2": {
        blocks: [...Array.from({length: 40}, (_, i) => `block_${i}`), "other_weights"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (30-39)": { enabled: [...Array.from({length: 10}, (_, i) => `block_${i + 30}`), "other_weights"], strength: 1.0 },
            "Mid-Late (20-39)": { enabled: [...Array.from({length: 20}, (_, i) => `block_${i + 20}`), "other_weights"], strength: 1.0 },
            "Skip Early (10-39)": { enabled: [...Array.from({length: 30}, (_, i) => `block_${i + 10}`), "other_weights"], strength: 1.0 },
            "Mid Only (15-25)": { enabled: Array.from({length: 11}, (_, i) => `block_${i + 15}`), strength: 1.0 },
            "Early Only (0-19)": { enabled: Array.from({length: 20}, (_, i) => `block_${i}`), strength: 1.0 },
            "Evens Only": { enabled: Array.from({length: 20}, (_, i) => `block_${i * 2}`), strength: 1.0 },
            "Odds Only": { enabled: Array.from({length: 20}, (_, i) => `block_${i * 2 + 1}`), strength: 1.0 },
        }
    },
    "QwenAnalyzerSelectiveLoaderV2": {
        blocks: [...Array.from({length: 60}, (_, i) => `block_${i}`), "other_weights"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (45-59)": { enabled: [...Array.from({length: 15}, (_, i) => `block_${i + 45}`), "other_weights"], strength: 1.0 },
            "Mid-Late (30-59)": { enabled: [...Array.from({length: 30}, (_, i) => `block_${i + 30}`), "other_weights"], strength: 1.0 },
            "Skip Early (15-59)": { enabled: [...Array.from({length: 45}, (_, i) => `block_${i + 15}`), "other_weights"], strength: 1.0 },
            "Mid Only (20-40)": { enabled: Array.from({length: 21}, (_, i) => `block_${i + 20}`), strength: 1.0 },
            "Early Only (0-29)": { enabled: Array.from({length: 30}, (_, i) => `block_${i}`), strength: 1.0 },
            "Evens Only": { enabled: Array.from({length: 30}, (_, i) => `block_${i * 2}`), strength: 1.0 },
            "Odds Only": { enabled: Array.from({length: 30}, (_, i) => `block_${i * 2 + 1}`), strength: 1.0 },
        }
    },
    // Model Layer Editor nodes (base model per-block control)
    "SDXLModelLayerEditor": {
        blocks: [...Array.from({length: 12}, (_, i) => `input_${i}`), "mid", ...Array.from({length: 12}, (_, i) => `output_${i}`), "other"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Outputs Only": { enabled: [...Array.from({length: 12}, (_, i) => `output_${i}`), "mid", "other"], strength: 1.0 },
            "Inputs Only": { enabled: [...Array.from({length: 12}, (_, i) => `input_${i}`), "mid", "other"], strength: 1.0 },
            "Custom": { enabled: "ALL", strength: 1.0 },
        }
    },
    "SD15ModelLayerEditor": {
        blocks: [...Array.from({length: 12}, (_, i) => `input_${i}`), "mid", ...Array.from({length: 12}, (_, i) => `output_${i}`), "other"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Outputs Only": { enabled: [...Array.from({length: 12}, (_, i) => `output_${i}`), "mid", "other"], strength: 1.0 },
            "Inputs Only": { enabled: [...Array.from({length: 12}, (_, i) => `input_${i}`), "mid", "other"], strength: 1.0 },
            "Custom": { enabled: "ALL", strength: 1.0 },
        }
    },
    "FLUXModelLayerEditor": {
        blocks: [...Array.from({length: 19}, (_, i) => `double_${i}`), ...Array.from({length: 38}, (_, i) => `single_${i}`), "other"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Double Only": { enabled: [...Array.from({length: 19}, (_, i) => `double_${i}`), "other"], strength: 1.0 },
            "Single Only": { enabled: [...Array.from({length: 38}, (_, i) => `single_${i}`), "other"], strength: 1.0 },
            "Custom": { enabled: "ALL", strength: 1.0 },
        }
    },
    "ZImageModelLayerEditor": {
        blocks: [...Array.from({length: 30}, (_, i) => `layer_${i}`), "other"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (20-29)": { enabled: [...Array.from({length: 10}, (_, i) => `layer_${i + 20}`), "other"], strength: 1.0 },
            "Mid-Late (15-29)": { enabled: [...Array.from({length: 15}, (_, i) => `layer_${i + 15}`), "other"], strength: 1.0 },
            "Custom": { enabled: "ALL", strength: 1.0 },
        }
    },
    "WanModelLayerEditor": {
        blocks: [...Array.from({length: 40}, (_, i) => `block_${i}`), "other"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (30-39)": { enabled: [...Array.from({length: 10}, (_, i) => `block_${i + 30}`), "other"], strength: 1.0 },
            "Custom": { enabled: "ALL", strength: 1.0 },
        }
    },
    "QwenModelLayerEditor": {
        blocks: [...Array.from({length: 60}, (_, i) => `block_${i}`), "other"],
        presets: {
            "Default": { enabled: "ALL", strength: 1.0 },
            "All Off": { enabled: [], strength: 0.0 },
            "Half Strength": { enabled: "ALL", strength: 0.5 },
            "Late Only (45-59)": { enabled: [...Array.from({length: 15}, (_, i) => `block_${i + 45}`), "other"], strength: 1.0 },
            "Custom": { enabled: "ALL", strength: 1.0 },
        }
    }
};

// Extension for combining block toggle + strength widgets (selective loaders)
app.registerExtension({
    name: "SelectiveLoRA.CombinedWidgets",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to selective loader nodes (including V2 combined nodes)
        const selectiveLoaders = [
            "SDXLSelectiveLoRALoader",
            "ZImageSelectiveLoRALoader",
            "FLUXSelectiveLoRALoader",
            "WanSelectiveLoRALoader",
            "QwenSelectiveLoRALoader",
            // V2 Combined Analyzer + Selective Loaders
            "ZImageAnalyzerSelectiveLoaderV2",
            "SDXLAnalyzerSelectiveLoaderV2",
            "FLUXAnalyzerSelectiveLoaderV2",
            "WanAnalyzerSelectiveLoaderV2",
            "QwenAnalyzerSelectiveLoaderV2",
            // Model Layer Editor nodes (base model per-block control)
            "SDXLModelLayerEditor",
            "SD15ModelLayerEditor",
            "FLUXModelLayerEditor",
            "ZImageModelLayerEditor",
            "WanModelLayerEditor",
            "QwenModelLayerEditor"
        ];

        if (!selectiveLoaders.includes(nodeData.name)) {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }

            // Capture 'this' in closure to prevent race conditions with multiple nodes
            const node = this;

            // After node is created, combine toggle + strength widget pairs, add preset dropdown, set width
            setTimeout(() => {
                // Guard against running multiple times (e.g., when loading old workflows)
                if (node._selectiveLoraInitialized) return;
                node._selectiveLoraInitialized = true;

                node.combineBlockWidgets();
                node.addPresetWidget(nodeData.name);
                node.setupSchedulePresetWidget();

                // Double the default width for better slider usability
                const minWidth = 500;
                if (node.size[0] < minWidth) {
                    node.size[0] = minWidth;
                    node.setDirtyCanvas(true);
                }
            }, 50);
        };

        // Handle workflow restoration - fixes corrupt state when loading saved workflows
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (origOnConfigure) {
                origOnConfigure.apply(this, arguments);
            }

            const node = this;

            // Sanitize strength widget values during restoration (only if corrupt)
            // ComfyUI deserializes as strings, but parseFloat handles this correctly
            setTimeout(() => {
                const config = SELECTIVE_LOADER_PRESETS[nodeData.name];
                if (!config) return;

                for (const blockName of config.blocks) {
                    const strWidget = node.widgets.find(w => w.name === blockName + "_str");
                    if (strWidget) {
                        // Only fix truly corrupt values (non-numeric strings)
                        // Preserve valid numbers including 0.0, negatives, etc.
                        let val = parseFloat(strWidget.value);
                        if (isNaN(val)) {
                            // Value is corrupt (e.g., "Custom" from old workflows)
                            // Only NOW do we default to 1.0
                            val = 1.0;
                            strWidget.value = val;
                        }
                        // If val is a valid number (including 0.0), keep the original value
                        // Don't overwrite it
                    }
                }

                // Note: We no longer need to force preset to "Custom" since we're using
                // the existing Python preset widget directly. ComfyUI will restore its
                // saved value correctly.

                // Merge workflow-embedded presets into localStorage (for Model Layer Editor)
                // This ensures old workflow presets appear in dropdown immediately on load
                if (nodeData.name.includes("ModelLayerEditor")) {
                    const browserPresetsWidget = node.widgets.find(w => w.name === "browser_presets_json");
                    if (browserPresetsWidget && browserPresetsWidget.value && browserPresetsWidget.value !== "{}") {
                        try {
                            const workflowPresets = JSON.parse(browserPresetsWidget.value);
                            if (workflowPresets && typeof workflowPresets === 'object') {
                                const localPresets = node.loadUserPresets ? node.loadUserPresets(nodeData.name) : {};
                                let merged = false;
                                for (const [name, data] of Object.entries(workflowPresets)) {
                                    if (name && !localPresets[name] && name.toLowerCase() !== 'true' && name.toLowerCase() !== 'false') {
                                        localPresets[name] = data;
                                        merged = true;
                                    }
                                }
                                if (merged && node.saveUserPreset) {
                                    // Save merged presets to localStorage
                                    const storageKey = `comfyui_model_layer_presets_${nodeData.name}`;
                                    localStorage.setItem(storageKey, JSON.stringify(localPresets));
                                    console.log(`[Model Layer Editor] Merged workflow presets into browser storage`);
                                    // Refresh dropdown to include new presets
                                    const presetWidget = node.widgets.find(w => w.name === "preset");
                                    if (presetWidget && presetWidget.options) {
                                        for (const name of Object.keys(localPresets)) {
                                            if (!presetWidget.options.values.includes(name)) {
                                                const customIdx = presetWidget.options.values.indexOf("Custom");
                                                if (customIdx >= 0) {
                                                    presetWidget.options.values.splice(customIdx, 0, name);
                                                } else {
                                                    presetWidget.options.values.push(name);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } catch (e) {
                            // Silent fail - workflow preset merge is optional
                        }
                    }
                }

                node.setDirtyCanvas(true);
            }, 150); // Longer delay to ensure ComfyUI finishes deserializing first
        };

        // Hook onExecuted to store analysis data and sync user presets
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(output) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }
            // Store analysis_json from our UI output
            if (output && output.analysis_json && output.analysis_json[0]) {
                try {
                    const jsonStr = output.analysis_json[0];
                    if (jsonStr && jsonStr.length > 2) {  // Not empty "{}"
                        const analysisData = JSON.parse(jsonStr);
                        this._analysisData = analysisData;
                        this.setDirtyCanvas(true);

                        // Sync ALL user presets from Python config file to localStorage
                        // This handles new computer / cleared storage scenarios
                        if (analysisData.user_presets) {
                            const { node_id, presets } = analysisData.user_presets;
                            const storageKey = `comfyui_model_layer_presets_${node_id}`;
                            try {
                                localStorage.setItem(storageKey, JSON.stringify(presets));
                                const count = Object.keys(presets).length;
                                if (count > 0) {
                                    console.log(`[Model Layer Editor] Synced ${count} user preset(s) to browser storage`);
                                }
                            } catch (e) {
                                console.warn("[Model Layer Editor] Failed to sync presets:", e);
                            }
                        }

                        // Handle deleted preset - remove from localStorage
                        if (analysisData.deleted_preset) {
                            const { node_id, preset_name } = analysisData.deleted_preset;
                            const storageKey = `comfyui_model_layer_presets_${node_id}`;
                            try {
                                const storedPresets = JSON.parse(localStorage.getItem(storageKey) || "{}");
                                if (storedPresets[preset_name]) {
                                    delete storedPresets[preset_name];
                                    localStorage.setItem(storageKey, JSON.stringify(storedPresets));
                                    console.log(`[Model Layer Editor] Deleted preset '${preset_name}' from browser storage`);
                                }
                            } catch (e) {
                                // Ignore errors
                            }
                        }

                        // Log confirmation for just-saved preset
                        if (analysisData.saved_preset) {
                            const { preset_name } = analysisData.saved_preset;
                            console.log(`[Model Layer Editor] Saved preset '${preset_name}' - available after restart`);
                        }

                        // CRITICAL: Clear the hidden save/delete fields after processing
                        // This prevents accidental re-saves on subsequent executions
                        const saveNameWidget = this.widgets?.find(w => w.name === "save_preset_name");
                        const deleteNameWidget = this.widgets?.find(w => w.name === "delete_preset_name");
                        if (saveNameWidget) saveNameWidget.value = "";
                        if (deleteNameWidget) deleteNameWidget.value = "";
                    }
                } catch (e) {
                    // Silent fail - analysis coloring is optional
                }
            }
        };

        nodeType.prototype.combineBlockWidgets = function() {
            // Find all widget pairs (toggle + _str)
            const widgetPairs = [];
            const strWidgetNames = new Set();

            for (const widget of this.widgets) {
                if (widget.name.endsWith('_str')) {
                    strWidgetNames.add(widget.name);
                }
            }

            for (const widget of this.widgets) {
                const strName = widget.name + '_str';
                if (strWidgetNames.has(strName)) {
                    const strWidget = this.widgets.find(w => w.name === strName);
                    if (strWidget) {
                        widgetPairs.push({
                            toggle: widget,
                            strength: strWidget,
                            name: widget.name
                        });
                    }
                }
            }

            // Create combined widgets
            for (const pair of widgetPairs) {
                this.createCombinedWidget(pair);
            }

            // Resize node to fit
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true);
        };

        nodeType.prototype.createCombinedWidget = function(pair) {
            const { toggle, strength, name } = pair;

            // Hide original widgets by replacing their draw methods
            const originalToggleDraw = toggle.draw?.bind(toggle);
            const originalStrengthDraw = strength.draw?.bind(strength);

            // Combined draw function for toggle widget (strength widget will be hidden)
            toggle.draw = function(ctx, node, widgetWidth, y, widgetHeight) {
                const margin = 10;
                const checkboxSize = 14;
                const labelWidth = 95; // Fixed label width
                const valueWidth = 38; // Value display
                const gap = 6;
                // Slider takes remaining space
                const sliderWidth = widgetWidth - margin - checkboxSize - gap - labelWidth - gap - valueWidth - margin - gap;

                // Calculate positions
                const checkboxX = margin;
                const labelX = checkboxX + checkboxSize + gap;
                const sliderX = labelX + labelWidth + gap;
                const valueX = sliderX + sliderWidth + gap;

                const enabled = Boolean(toggle.value);
                // Read strength value (ComfyUI may deserialize as string)
                let strengthVal = parseFloat(strength.value);
                if (isNaN(strengthVal)) {
                    // Fallback for corrupt data - should rarely happen after onConfigure fix
                    strengthVal = 1.0;
                }

                // Get impact score from analysis if available
                let impactScore = null;
                const analysis = getAnalysisFromInput(node);
                if (analysis && analysis.blocks) {
                    // Try to find this block's score
                    // Map widget name to analysis key (other_weights -> other)
                    const analysisKey = name === "other_weights" ? "other" : name;
                    const blockData = analysis.blocks[analysisKey];
                    if (blockData && typeof blockData.score === 'number') {
                        impactScore = blockData.score;
                    }
                }

                // Determine checkbox color based on impact score or default
                let checkboxColor = "#5599ff";  // Default blue
                if (impactScore !== null) {
                    checkboxColor = getImpactColor(impactScore);
                }

                // Background
                ctx.fillStyle = enabled ? "#2a2a2a" : "#1e1e1e";
                ctx.fillRect(0, y, widgetWidth, widgetHeight);

                // Checkbox
                ctx.strokeStyle = enabled ? checkboxColor : "#555";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(checkboxX, y + (widgetHeight - checkboxSize) / 2, checkboxSize, checkboxSize);

                // Always show impact color, but dimmed when disabled
                ctx.globalAlpha = enabled ? 1.0 : 0.35;
                ctx.fillStyle = checkboxColor;
                ctx.fillRect(checkboxX + 2, y + (widgetHeight - checkboxSize) / 2 + 2, checkboxSize - 4, checkboxSize - 4);
                ctx.globalAlpha = 1.0;

                // Label
                ctx.fillStyle = enabled ? "#ddd" : "#666";
                ctx.font = "11px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                // Truncate label if too long
                let displayLabel = name;
                const maxLabelWidth = labelWidth - 4;
                if (ctx.measureText(displayLabel).width > maxLabelWidth) {
                    while (ctx.measureText(displayLabel + "…").width > maxLabelWidth && displayLabel.length > 3) {
                        displayLabel = displayLabel.slice(0, -1);
                    }
                    displayLabel += "…";
                }
                ctx.fillText(displayLabel, labelX, y + widgetHeight / 2);

                // Slider track
                const trackY = y + widgetHeight / 2;
                const trackHeight = 4;
                const min = strength.options?.min ?? -2.0;
                const max = strength.options?.max ?? 2.0;
                const range = max - min;
                const normalizedStrength = (strengthVal - min) / range;
                const zeroPos = (0 - min) / range;

                ctx.fillStyle = "#333";
                ctx.beginPath();
                ctx.roundRect(sliderX, trackY - trackHeight / 2, sliderWidth, trackHeight, 2);
                ctx.fill();

                // Slider fill (from zero point)
                ctx.fillStyle = enabled ? (strengthVal >= 0 ? "#5599ff" : "#ff6655") : "#444";
                if (min < 0) {
                    const zeroX = sliderX + zeroPos * sliderWidth;
                    const strengthX = sliderX + normalizedStrength * sliderWidth;
                    const fillStart = Math.min(zeroX, strengthX);
                    const fillWidth = Math.abs(strengthX - zeroX);
                    ctx.beginPath();
                    ctx.roundRect(fillStart, trackY - trackHeight / 2, fillWidth, trackHeight, 2);
                    ctx.fill();
                } else {
                    ctx.beginPath();
                    ctx.roundRect(sliderX, trackY - trackHeight / 2, normalizedStrength * sliderWidth, trackHeight, 2);
                    ctx.fill();
                }

                // Slider handle
                const handleX = sliderX + normalizedStrength * sliderWidth;
                const handleRadius = 5;
                ctx.fillStyle = enabled ? "#fff" : "#666";
                ctx.beginPath();
                ctx.arc(handleX, trackY, handleRadius, 0, Math.PI * 2);
                ctx.fill();

                // Value text
                ctx.fillStyle = enabled ? "#ddd" : "#555";
                ctx.textAlign = "right";
                ctx.font = "10px Arial";
                ctx.fillText(strengthVal.toFixed(2), widgetWidth - margin, y + widgetHeight / 2);
            };

            // Store layout info for mouse handling
            toggle.sliderInfo = {
                margin: 10,
                checkboxSize: 14,
                labelWidth: 95,
                valueWidth: 38,
                gap: 6,
                min: -2.0,
                max: 2.0,
                step: 0.05,  // Hardcoded - ComfyUI widget options not reliably accessible
                getLayout: function(widgetWidth) {
                    const sliderWidth = widgetWidth - this.margin - this.checkboxSize - this.gap - this.labelWidth - this.gap - this.valueWidth - this.margin - this.gap;
                    const checkboxX = this.margin;
                    const labelX = checkboxX + this.checkboxSize + this.gap;
                    const sliderX = labelX + this.labelWidth + this.gap;
                    const valueX = sliderX + sliderWidth + this.gap;
                    return { sliderWidth, checkboxX, sliderX, valueX };
                }
            };

            // Mouse handling for slider - let default toggle behavior work for other clicks
            const originalMouse = toggle.mouse?.bind(toggle);
            toggle.mouse = function(event, pos, node) {
                const widgetWidth = node.size[0];
                const info = toggle.sliderInfo;
                const layout = info.getLayout(widgetWidth);
                const localX = pos[0];

                // Slider interaction - intercept drag on slider area
                if (localX >= layout.sliderX - 5 && localX <= layout.sliderX + layout.sliderWidth + 5) {
                    if (event.type === "pointerdown" || event.type === "pointermove") {
                        let normalized = (localX - layout.sliderX) / layout.sliderWidth;
                        normalized = Math.max(0, Math.min(1, normalized));
                        let newStrength = info.min + normalized * (info.max - info.min);
                        // Snap to step
                        newStrength = Math.round(newStrength / info.step) * info.step;
                        newStrength = Math.max(info.min, Math.min(info.max, newStrength));
                        strength.value = newStrength;
                        node.setDirtyCanvas(true);
                        return true;
                    }
                }

                // Let default behavior handle toggle clicks
                if (originalMouse) {
                    return originalMouse(event, pos, node);
                }
                return false;
            };

            // Hide the strength widget completely
            strength.draw = function() {};
            strength.computeSize = function() { return [0, -4]; }; // Negative height to collapse
        };

        // Add preset dropdown widget
        nodeType.prototype.addPresetWidget = function(nodeName) {
            const config = SELECTIVE_LOADER_PRESETS[nodeName];
            if (!config) return;

            // Start with hardcoded presets
            const presetNames = Object.keys(config.presets);

            // Add user presets from localStorage (for Model Layer Editor nodes)
            if (nodeName.includes("ModelLayerEditor")) {
                const userPresets = this.loadUserPresets(nodeName);
                for (const name of Object.keys(userPresets)) {
                    if (!presetNames.includes(name)) {
                        // Insert before "Custom" if present, otherwise at end
                        const customIdx = presetNames.indexOf("Custom");
                        if (customIdx >= 0) {
                            presetNames.splice(customIdx, 0, name);
                        } else {
                            presetNames.push(name);
                        }
                    }
                }
            }

            const node = this;

            // Find the Python preset widget and convert it to our JS preset widget
            // This avoids adding a new widget which would cause index mismatch
            const presetWidget = this.widgets.find(w => w.name === "preset");
            if (!presetWidget) return;

            // IMPORTANT: Start with Python's preset values (includes file-saved user presets)
            // Then merge with our JS presets + localStorage presets
            const pythonPresets = presetWidget.options?.values || [];
            const allPresetNames = [...presetNames];

            // Add any Python presets not already in our list (these are file-saved user presets)
            const addedFromPython = [];
            for (const name of pythonPresets) {
                if (!allPresetNames.includes(name)) {
                    addedFromPython.push(name);
                    // Insert before "Custom" if present
                    const customIdx = allPresetNames.indexOf("Custom");
                    if (customIdx >= 0) {
                        allPresetNames.splice(customIdx, 0, name);
                    } else {
                        allPresetNames.push(name);
                    }
                }
            }
            if (addedFromPython.length > 0) {
                console.log(`[Model Layer Editor] Added ${addedFromPython.length} preset(s) from Python config file: ${addedFromPython.join(", ")}`);
            }

            // Convert the existing combo widget to work with our JS preset system
            presetWidget.options = presetWidget.options || {};
            presetWidget.options.values = allPresetNames;

            // Set initial value to "Default" (or current value if it's valid)
            if (!allPresetNames.includes(presetWidget.value)) {
                presetWidget.value = "Default";
            }

            // Override callback to apply preset when changed
            const origCallback = presetWidget.callback;
            presetWidget.callback = function(value) {
                // When user changes preset via UI, apply it
                node.applyPreset(nodeName, value);

                // Call original callback if it exists
                if (origCallback) {
                    origCallback.call(this, value);
                }
            };

            // CRITICAL FIX: Override serializeValue to always return "Custom" for Python
            // This way Python always uses individual toggle values, while JavaScript
            // manages its own independent preset system in the UI
            const origSerializeValue = presetWidget.serializeValue;
            presetWidget.serializeValue = function() {
                return "Custom";
            };

            // Store reference for later
            this.presetWidget = presetWidget;

            // Add Save Preset button for Model Layer Editor nodes
            if (nodeName.includes("ModelLayerEditor")) {
                this.addSavePresetButton(nodeName);
            }

            // Resize node
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true);
        };

        // ============================================================================
        // USER PRESET SYSTEM
        // ============================================================================
        // Save/delete user presets with dual persistence (localStorage + file)
        // Preset system: localStorage for instant UI updates, Python config file for persistence

        // Add Save Preset button for Model Layer Editor nodes
        nodeType.prototype.addSavePresetButton = function(nodeName) {
            const node = this;
            const config = SELECTIVE_LOADER_PRESETS[nodeName];
            if (!config) return;

            // Find and HIDE the save_preset_name, delete_preset_name, and browser_presets_json widgets
            // They're still used internally for Python sync, but hidden from UI
            const saveNameWidget = this.widgets.find(w => w.name === "save_preset_name");
            const deleteNameWidget = this.widgets.find(w => w.name === "delete_preset_name");
            const browserPresetsWidget = this.widgets.find(w => w.name === "browser_presets_json");

            if (saveNameWidget) {
                saveNameWidget.draw = function() {};
                saveNameWidget.computeSize = function() { return [0, -4]; };
            }
            if (deleteNameWidget) {
                deleteNameWidget.draw = function() {};
                deleteNameWidget.computeSize = function() { return [0, -4]; };
            }
            if (browserPresetsWidget) {
                browserPresetsWidget.draw = function() {};
                browserPresetsWidget.computeSize = function() { return [0, -4]; };
                // Override serializeValue to send all localStorage presets to Python
                const nodeNameCopy = nodeName;  // Capture for closure
                browserPresetsWidget.serializeValue = function() {
                    try {
                        const storageKey = `comfyui_model_layer_presets_${nodeNameCopy}`;
                        const saved = localStorage.getItem(storageKey);
                        const presets = saved ? JSON.parse(saved) : {};
                        const json = JSON.stringify(presets);
                        console.log(`[Model Layer Editor] Serializing ${Object.keys(presets).length} preset(s) for sync`);
                        return json;
                    } catch (e) {
                        console.warn("[Model Layer Editor] Failed to serialize presets:", e);
                        return "{}";
                    }
                };
                // Also set the value directly so it's available immediately
                browserPresetsWidget.value = browserPresetsWidget.serializeValue();
            } else {
                console.warn("[Model Layer Editor] browser_presets_json widget not found!");
            }

            // Add the save button - uses prompt() for name input
            const saveButton = this.addWidget("button", "💾 Save Preset", null, () => {
                // Prompt for preset name
                const presetName = prompt("Enter preset name:")?.trim();

                if (!presetName) {
                    return;  // User cancelled or empty name
                }

                // Gather current values from widgets
                const enabledBlocks = [];
                const overrides = {};

                for (const blockName of config.blocks) {
                    const toggleWidget = node.widgets.find(w => w.name === blockName);
                    const strWidget = node.widgets.find(w => w.name === blockName + "_str");

                    if (toggleWidget && toggleWidget.value) {
                        enabledBlocks.push(blockName);
                    }
                    if (strWidget && strWidget.value !== 1.0) {
                        overrides[blockName] = strWidget.value;
                    }
                }

                // Build preset data
                const preset = {
                    enabled: enabledBlocks.length === config.blocks.length ? "ALL" : enabledBlocks,
                    strength: 1.0,
                    overrides: overrides
                };

                // Save to localStorage
                node.saveUserPreset(nodeName, presetName, preset);

                // Update browser_presets_json widget value for Python sync
                const browserPresetsWidgetSave = node.widgets.find(w => w.name === "browser_presets_json");
                if (browserPresetsWidgetSave) {
                    const allPresets = node.loadUserPresets(nodeName);
                    browserPresetsWidgetSave.value = JSON.stringify(allPresets);
                    console.log(`[Model Layer Editor] Updated sync widget: ${Object.keys(allPresets).length} presets`);
                }

                // Update preset dropdown to include the new preset
                const presetWidget = node.widgets.find(w => w.name === "preset");
                if (presetWidget && presetWidget.options && presetWidget.options.values) {
                    if (!presetWidget.options.values.includes(presetName)) {
                        // Insert before "Custom" if it exists, otherwise at end
                        const customIdx = presetWidget.options.values.indexOf("Custom");
                        if (customIdx >= 0) {
                            presetWidget.options.values.splice(customIdx, 0, presetName);
                        } else {
                            presetWidget.options.values.push(presetName);
                        }
                    }
                    // Set dropdown to the new preset
                    presetWidget.value = presetName;
                }

                node.setDirtyCanvas(true);
                console.log(`[Model Layer Editor] Saved preset '${presetName}' to browser storage`);
            });

            // Add delete button that deletes currently selected preset (if it's a user preset)
            const deleteButton = this.addWidget("button", "🗑️ Delete Selected Preset", null, () => {
                const presetWidget = node.widgets.find(w => w.name === "preset");
                if (!presetWidget) return;

                const presetName = presetWidget.value;

                // Check if it's a built-in preset (can't delete those)
                if (config.presets[presetName]) {
                    alert(`Cannot delete built-in preset '${presetName}'`);
                    return;
                }

                // Check if it's a user preset
                const userPresets = node.loadUserPresets(nodeName);
                if (!userPresets[presetName]) {
                    alert(`'${presetName}' is not a user preset`);
                    return;
                }

                // Confirm deletion
                if (!confirm(`Delete preset '${presetName}'?`)) {
                    return;
                }

                // Remove from localStorage
                delete userPresets[presetName];
                const storageKey = `comfyui_model_layer_presets_${nodeName}`;
                localStorage.setItem(storageKey, JSON.stringify(userPresets));
                console.log(`[Model Layer Editor] Deleted preset '${presetName}' from browser storage`);

                // Update browser_presets_json widget value for Python sync
                const browserPresetsWidgetDel = node.widgets.find(w => w.name === "browser_presets_json");
                if (browserPresetsWidgetDel) {
                    browserPresetsWidgetDel.value = JSON.stringify(userPresets);
                    console.log(`[Model Layer Editor] Updated sync widget: ${Object.keys(userPresets).length} presets`);
                }

                // Remove from dropdown
                if (presetWidget.options && presetWidget.options.values) {
                    const idx = presetWidget.options.values.indexOf(presetName);
                    if (idx >= 0) {
                        presetWidget.options.values.splice(idx, 1);
                    }
                }

                // Reset to Default
                presetWidget.value = "Default";
                node.applyPreset(nodeName, "Default");
                node.setDirtyCanvas(true);
            });

            // Leave buttons at the very end of the widget array where addWidget() put them
            // DO NOT reposition - inserting anywhere else shifts widget indices and breaks serialization
        };

        // Setup schedule preset widget to populate strength_schedule text field
        nodeType.prototype.setupSchedulePresetWidget = function() {
            const node = this;

            // Find the schedule_preset dropdown widget
            const schedulePresetWidget = this.widgets.find(w => w.name === "schedule_preset");
            if (!schedulePresetWidget) return;

            // Find the strength_schedule text widget
            const strengthScheduleWidget = this.widgets.find(w => w.name === "strength_schedule");
            if (!strengthScheduleWidget) return;

            // Override callback to fill strength_schedule when preset is selected
            const origCallback = schedulePresetWidget.callback;
            schedulePresetWidget.callback = function(value) {
                // Look up the preset value and fill the text field
                const scheduleValue = SCHEDULE_PRESETS[value];
                if (scheduleValue !== undefined) {
                    strengthScheduleWidget.value = scheduleValue;
                    node.setDirtyCanvas(true);
                }

                // Call original callback if it exists
                if (origCallback) {
                    origCallback.call(this, value);
                }
            };

            // Store reference for later
            this.schedulePresetWidget = schedulePresetWidget;
            this.strengthScheduleWidget = strengthScheduleWidget;
        };

        // Load user presets from localStorage (part of dual-persistence preset system)
        nodeType.prototype.loadUserPresets = function(nodeName) {
            const storageKey = `comfyui_model_layer_presets_${nodeName}`;
            try {
                const saved = localStorage.getItem(storageKey);
                const presets = saved ? JSON.parse(saved) : {};
                const count = Object.keys(presets).length;
                if (count > 0) {
                    console.log(`[Model Layer Editor] Loaded ${count} preset(s) from browser storage for ${nodeName}`);
                }
                return presets;
            } catch (e) {
                console.warn(`[Model Layer Editor] Failed to load presets from browser storage:`, e);
                return {};
            }
        };

        // Save user presets to localStorage (called from Python via message)
        nodeType.prototype.saveUserPreset = function(nodeName, presetName, presetData) {
            const storageKey = `comfyui_model_layer_presets_${nodeName}`;
            try {
                const presets = this.loadUserPresets(nodeName);
                presets[presetName] = presetData;
                localStorage.setItem(storageKey, JSON.stringify(presets));
                console.log(`[Model Layer Editor] Saved user preset '${presetName}' for ${nodeName}`);
            } catch (e) {
                console.warn("[Model Layer Editor] Failed to save user preset:", e);
            }
        };

        // Apply a preset to all block toggles and strengths
        nodeType.prototype.applyPreset = function(nodeName, presetName) {
            const config = SELECTIVE_LOADER_PRESETS[nodeName];
            if (!config) return;

            // First check hardcoded presets
            let preset = config.presets[presetName];

            // If not found, check user presets in localStorage
            if (!preset) {
                const userPresets = this.loadUserPresets(nodeName);
                preset = userPresets[presetName];
            }

            // If still not found, check if it's likely a file-saved preset that needs syncing
            if (!preset) {
                // Check if this preset name is in the dropdown (meaning Python knows about it from file)
                const presetWidget = this.widgets.find(w => w.name === "preset");
                const isInDropdown = presetWidget?.options?.values?.includes(presetName);

                if (isInDropdown && nodeName.includes("ModelLayerEditor")) {
                    // It's a file-saved preset that hasn't been synced to browser storage yet
                    console.log(`[Model Layer Editor] Preset '${presetName}' exists in file but not browser storage. Run workflow to sync.`);
                    // Don't show alert - just keep current values until sync happens
                } else {
                    console.log(`[Model Layer Editor] Preset '${presetName}' not found, keeping current values`);
                }
                return;
            }

            const allBlocks = config.blocks;
            const enabledBlocks = preset.enabled === "ALL" ? allBlocks : preset.enabled;
            const enabledSet = new Set(enabledBlocks);
            const baseStrength = preset.strength;
            const overrides = preset.overrides || {};  // Per-block strength overrides

            // Update all toggle and strength widgets
            for (const blockName of allBlocks) {
                const toggleWidget = this.widgets.find(w => w.name === blockName);
                const strWidget = this.widgets.find(w => w.name === blockName + "_str");

                if (toggleWidget) {
                    toggleWidget.value = enabledSet.has(blockName);
                }
                if (strWidget) {
                    // Use override if present, otherwise base strength
                    strWidget.value = overrides.hasOwnProperty(blockName) ? overrides[blockName] : baseStrength;
                }
            }

            this.setDirtyCanvas(true);
        };
    }
});

// Extension for ScheduledLoRALoader - just the schedule preset dropdown
app.registerExtension({
    name: "ScheduledLoRA.PresetDropdown",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ScheduledLoRALoader") {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }

            const node = this;

            // Setup schedule preset dropdown after node is created
            setTimeout(() => {
                // Find the schedule_preset dropdown widget
                const schedulePresetWidget = node.widgets.find(w => w.name === "schedule_preset");
                if (!schedulePresetWidget) return;

                // Find the strength_schedule text widget
                const strengthScheduleWidget = node.widgets.find(w => w.name === "strength_schedule");
                if (!strengthScheduleWidget) return;

                // Override callback to fill strength_schedule when preset is selected
                const origCallback = schedulePresetWidget.callback;
                schedulePresetWidget.callback = function(value) {
                    // Look up the preset value and fill the text field
                    const scheduleValue = SCHEDULE_PRESETS[value];
                    if (scheduleValue !== undefined) {
                        strengthScheduleWidget.value = scheduleValue;
                        node.setDirtyCanvas(true);
                    }

                    // Call original callback if it exists
                    if (origCallback) {
                        origCallback.call(this, value);
                    }
                };
            }, 50);
        };
    }
});

// Extension for Clippy Reloaded - display text below image
app.registerExtension({
    name: "ClippyReloaded.TextDisplay",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ClippyRebornImageLoader") {
            return;
        }

        // Override onExecuted to display the Clippy message
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }

            // Display Clippy's message below the image
            if (message && message.text && message.text.length > 0) {
                const clippyText = message.text[0];

                // Find or create the text display element
                if (!this.clippyTextWidget) {
                    // Add a text widget to display Clippy's message
                    this.clippyTextWidget = this.addWidget("text", "clippy_says", "", () => {}, {
                        serialize: false
                    });
                }
                this.clippyTextWidget.value = clippyText;
                this.setDirtyCanvas(true);
            }
        };
    }
});

// Extension for Image of the Day - persist API keys per source
app.registerExtension({
    name: "ImageOfDay.ApiKeyPersistence",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ImageOfDayLoader") {
            return;
        }

        // Local storage key for API keys
        const STORAGE_KEY = "comfyui_imageofday_apikeys";

        // Sources that need API keys (must match Python)
        const API_SOURCES = ["NASA APOD (API)", "Unsplash (API)", "Pexels (API)"];

        // Check if source needs an API key
        function needsApiKey(source) {
            return API_SOURCES.includes(source);
        }

        // Load saved API keys from localStorage
        function loadApiKeys() {
            try {
                const saved = localStorage.getItem(STORAGE_KEY);
                return saved ? JSON.parse(saved) : {};
            } catch (e) {
                return {};
            }
        }

        // Save API keys to localStorage
        function saveApiKeys(keys) {
            try {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(keys));
            } catch (e) {
                console.warn("[ImageOfDay] Failed to save API keys:", e);
            }
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }

            const node = this;

            // Setup after widgets are created
            setTimeout(() => {
                const sourceWidget = node.widgets.find(w => w.name === "source");
                const apiKeyWidget = node.widgets.find(w => w.name === "api_key");

                if (!sourceWidget || !apiKeyWidget) return;

                // Function to update API key field based on source
                function updateApiKeyField(source) {
                    if (needsApiKey(source)) {
                        // Load saved key for this API source
                        const keys = loadApiKeys();
                        apiKeyWidget.value = keys[source] || "";
                    } else {
                        // No key needed for this source
                        apiKeyWidget.value = "no key needed";
                    }
                }

                // Set initial API key for current source
                updateApiKeyField(sourceWidget.value);

                // Override source callback to update API key field
                const origCallback = sourceWidget.callback;
                sourceWidget.callback = function(value) {
                    updateApiKeyField(value);
                    node.setDirtyCanvas(true);

                    if (origCallback) {
                        origCallback.call(this, value);
                    }
                };

                // Save API key when it changes (only for API sources)
                const origApiKeyCallback = apiKeyWidget.callback;
                apiKeyWidget.callback = function(value) {
                    const source = sourceWidget.value;
                    if (needsApiKey(source) && value && value.trim() && value !== "no key needed") {
                        const keys = loadApiKeys();
                        keys[source] = value.trim();
                        saveApiKeys(keys);
                    }

                    if (origApiKeyCallback) {
                        origApiKeyCallback.call(this, value);
                    }
                };

                // Store save function for use before execution
                node._imageOfDaySaveKey = function() {
                    const source = sourceWidget.value;
                    const value = apiKeyWidget.value;
                    if (needsApiKey(source) && value && value.trim() && value !== "no key needed") {
                        const keys = loadApiKeys();
                        keys[source] = value.trim();
                        saveApiKeys(keys);
                    }
                };
            }, 50);
        };

        // Save API key before execution
        const origOnExecute = nodeType.prototype.onExecute;
        nodeType.prototype.onExecute = function() {
            if (this._imageOfDaySaveKey) {
                this._imageOfDaySaveKey();
            }
            if (origOnExecute) {
                origOnExecute.apply(this, arguments);
            }
        };
    }
});
