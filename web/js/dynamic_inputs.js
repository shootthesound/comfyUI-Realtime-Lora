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
        if (nodeData.name !== "LoRALoaderWithAnalysis") return;

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
            "Peak Impact (18-25)": { enabled: Array.from({length: 8}, (_, i) => `layer_${i + 18}`), strength: 1.0 },
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
        }
    }
};

// Extension for combining block toggle + strength widgets (selective loaders)
app.registerExtension({
    name: "SelectiveLoRA.CombinedWidgets",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to selective loader nodes
        const selectiveLoaders = [
            "SDXLSelectiveLoRALoader",
            "ZImageSelectiveLoRALoader",
            "FLUXSelectiveLoRALoader",
            "WanSelectiveLoRALoader",
            "QwenSelectiveLoRALoader"
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
                node.setupBidirectionalSync(nodeData.name);

                // Populate string on first creation
                setTimeout(() => {
                    if (node.updateStringFromUI) {
                        node.updateStringFromUI();
                    }
                }, 150);

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

            // Restore widget values from saved workflow
            setTimeout(() => {
                const config = SELECTIVE_LOADER_PRESETS[nodeData.name];
                if (!config) return;

                // Restore all toggle and strength values from info.widgets_values if available
                if (info && info.widgets_values) {
                    for (let i = 0; i < node.widgets.length && i < info.widgets_values.length; i++) {
                        const widget = node.widgets[i];
                        const savedValue = info.widgets_values[i];

                        // Restore the value
                        if (widget.name.endsWith('_str')) {
                            // Strength widget - ensure numeric
                            let val = parseFloat(savedValue);
                            if (isNaN(val)) {
                                val = 1.0; // Fallback for corrupt data
                            }
                            widget.value = val;
                        } else if (widget.type === 'toggle' || typeof widget.value === 'boolean') {
                            // Boolean toggle widget
                            widget.value = Boolean(savedValue);
                        } else {
                            // Other widgets (preset, etc.)
                            widget.value = savedValue;
                        }
                    }
                }

                // Sanitize strength widget values during restoration (only if corrupt)
                for (const blockName of config.blocks) {
                    const strWidget = node.widgets.find(w => w.name === blockName + "_str");
                    if (strWidget) {
                        let val = parseFloat(strWidget.value);
                        if (isNaN(val)) {
                            val = 1.0;
                            strWidget.value = val;
                        }
                    }
                }

                // Note: We no longer need to force preset to "Custom" since we're using
                // the existing Python preset widget directly. ComfyUI will restore its
                // saved value correctly.

                // Update string from restored UI values (for copy/paste or workflow load)
                if (node.updateStringFromUI) {
                    node.updateStringFromUI();
                }

                node.setDirtyCanvas(true);
            }, 150); // Longer delay to ensure ComfyUI finishes deserializing first
        };

        // Hook onExecuted to store analysis data when we receive it
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
                        this._analysisData = JSON.parse(jsonStr);
                        this.setDirtyCanvas(true);
                    }
                } catch (e) {
                    // Silent fail - analysis coloring is optional
                }
            }
        };

        // Hook onMouseUp to trigger sync after slider interaction completes
        const origOnMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function(e, localPos, graphCanvas) {
            const result = origOnMouseUp ? origOnMouseUp.apply(this, arguments) : false;

            // Trigger UI→Text sync after any mouse interaction that might have changed sliders
            if (this.updateStringFromUI) {
                this.updateStringFromUI();
            }

            return result;
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

                if (enabled) {
                    ctx.fillStyle = checkboxColor;
                    ctx.fillRect(checkboxX + 2, y + (widgetHeight - checkboxSize) / 2 + 2, checkboxSize - 4, checkboxSize - 4);
                }

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

                        // Force Python preset to "Custom" when manually changing strength
                        if (node._pythonPresetWidget) {
                            node._pythonPresetWidget.value = "Custom";
                        }

                        node.setDirtyCanvas(true);
                        return true;
                    }
                }

                // Let default behavior handle toggle clicks
                if (originalMouse) {
                    const result = originalMouse(event, pos, node);
                    // Force Python preset to "Custom" when manually toggling blocks
                    if (event.type === "pointerdown" && node._pythonPresetWidget) {
                        node._pythonPresetWidget.value = "Custom";
                    }
                    return result;
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

            const presetNames = Object.keys(config.presets);
            const node = this;

            // Find the Python preset widget and convert it to our JS preset widget
            // This avoids adding a new widget which would cause index mismatch
            const presetWidget = this.widgets.find(w => w.name === "preset");
            if (!presetWidget) return;

            // Convert the existing combo widget to work with our JS preset system
            presetWidget.options = presetWidget.options || {};
            presetWidget.options.values = presetNames;

            // Set initial value to "Default" (or current value if it's valid)
            if (!presetNames.includes(presetWidget.value)) {
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

            // Resize node
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true);
        };

        // Apply a preset to all block toggles and strengths
        nodeType.prototype.applyPreset = function(nodeName, presetName) {
            const config = SELECTIVE_LOADER_PRESETS[nodeName];
            if (!config) return;

            const preset = config.presets[presetName];
            if (!preset) return;

            const allBlocks = config.blocks;
            const enabledBlocks = preset.enabled === "ALL" ? allBlocks : preset.enabled;
            const enabledSet = new Set(enabledBlocks);
            const strength = preset.strength;

            // Update all toggle and strength widgets
            for (const blockName of allBlocks) {
                const toggleWidget = this.widgets.find(w => w.name === blockName);
                const strWidget = this.widgets.find(w => w.name === blockName + "_str");

                if (toggleWidget) {
                    toggleWidget.value = enabledSet.has(blockName);
                }
                if (strWidget) {
                    strWidget.value = strength;
                }
            }

            // Force Python preset to "Custom" so it reads individual toggles
            if (this._pythonPresetWidget) {
                this._pythonPresetWidget.value = "Custom";
            }

            // Trigger bidirectional sync to update string
            if (this.updateStringFromUI) {
                this.updateStringFromUI();
            }

            this.setDirtyCanvas(true);
        };

        // Bidirectional sync between block_weights_string and UI sliders
        nodeType.prototype.setupBidirectionalSync = function(nodeName) {
            const node = this;
            const config = SELECTIVE_LOADER_PRESETS[nodeName];
            if (!config) return;

            const stringWidget = node.widgets.find(w => w.name === 'block_weights_string');
            if (!stringWidget) return;

            // Track last known string value
            node._lastStringValue = stringWidget.value || "";

            // Text → UI sync: Parse string and update sliders when text changes
            node.parseStringToUI = function() {
                const stringWidget = node.widgets.find(w => w.name === 'block_weights_string');
                if (!stringWidget || !stringWidget.value.trim() || stringWidget._updating) return;

                const input = stringWidget.value.trim();

                // Support both formats:
                // 1. Named format: "%default=1.0, te1=0.5, in7-8=1.2"
                // 2. Positional format: "1.0, 0.0, 1.5, ..." (comma-separated numbers)

                if (input.startsWith('%')) {
                    // Named format - not implemented yet for UI sync (Python handles it)
                    return;
                }

                // Positional format: comma-separated numbers
                const values = input.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                const blocks = config.blocks;

                if (values.length !== blocks.length) {
                    console.warn(`Block weights string has ${values.length} values but expected ${blocks.length}`);
                    return;
                }

                // Update UI sliders from string values
                for (let i = 0; i < blocks.length && i < values.length; i++) {
                    const blockName = blocks[i];
                    const value = values[i];

                    const toggleWidget = node.widgets.find(w => w.name === blockName);
                    const strWidget = node.widgets.find(w => w.name === blockName + "_str");

                    if (toggleWidget && strWidget) {
                        toggleWidget.value = value !== 0;
                        strWidget.value = value !== 0 ? value : 1.0;
                    }
                }

                node.setDirtyCanvas(true);
            };

            // UI → Text sync: Update string when sliders change
            node.updateStringFromUI = function() {
                const stringWidget = node.widgets.find(w => w.name === 'block_weights_string');
                if (!stringWidget || stringWidget._updating) return;

                const blocks = config.blocks;
                const values = [];

                for (const blockName of blocks) {
                    const toggleWidget = node.widgets.find(w => w.name === blockName);
                    const strWidget = node.widgets.find(w => w.name === blockName + "_str");

                    if (toggleWidget && strWidget) {
                        // If enabled, use strength value; if disabled, use 0
                        values.push(toggleWidget.value ? strWidget.value.toFixed(2) : "0.00");
                    } else {
                        values.push("1.00"); // Default fallback
                    }
                }

                // Set flag to prevent infinite loop
                stringWidget._updating = true;
                stringWidget.value = values.join(", ");
                stringWidget._updating = false;
            };

            // Add method to check for text changes (called on various interactions)
            node.checkAndSyncTextChanges = function() {
                const stringWidget = node.widgets.find(w => w.name === 'block_weights_string');
                if (!stringWidget) return;

                // Check if string widget text changed
                if (stringWidget.value !== node._lastStringValue && !stringWidget._updating) {
                    node._lastStringValue = stringWidget.value;
                    node.parseStringToUI();
                }
            };

            // Check on mouse interactions with node (text → UI sync)
            const origOnMouseDown = node.onMouseDown;
            node.onMouseDown = function(e, pos, canvas) {
                // Check if text changed and sync to UI
                setTimeout(() => node.checkAndSyncTextChanges(), 10);
                if (origOnMouseDown) {
                    return origOnMouseDown.apply(this, arguments);
                }
            };

            // Whenever a slider changes, update the string
            // Use setTimeout to ensure widgets are fully initialized
            setTimeout(() => {
                for (const blockName of config.blocks) {
                    const toggleWidget = node.widgets.find(w => w.name === blockName);
                    const strWidget = node.widgets.find(w => w.name === blockName + "_str");

                    if (toggleWidget && strWidget) {
                        // Store reference to node for closure
                        const nodeRef = node;

                        // Wrap original callbacks
                        const origToggleCallback = toggleWidget.callback;
                        toggleWidget.callback = function(value) {
                            if (origToggleCallback) origToggleCallback.call(this, value);
                            // Update string after toggle changes
                            if (nodeRef.updateStringFromUI) {
                                nodeRef.updateStringFromUI();
                            }
                        };

                        const origStrCallback = strWidget.callback;
                        strWidget.callback = function(value) {
                            if (origStrCallback) origStrCallback.call(this, value);
                            // Update string after strength changes
                            if (nodeRef.updateStringFromUI) {
                                nodeRef.updateStringFromUI();
                            }
                        };
                    }
                }
            }, 200);
        };
    }
});
