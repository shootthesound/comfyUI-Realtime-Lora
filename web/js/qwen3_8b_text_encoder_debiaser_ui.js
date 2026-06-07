import { app } from "../../../scripts/app.js";

// ============================================================================
// QWEN3-8B TEXT ENCODER DEEP DEBIASER — Sub-Component UI Extension
// For Flux 2 Klein 9B
// 182 combined checkbox+slider widgets with functional-group presets.
// ============================================================================

function getImpactColor(score) {
    if (score < 10) return "#0066ff";
    if (score < 20) return "#0088ff";
    if (score < 30) return "#00aaff";
    if (score < 40) return "#00cccc";
    if (score < 50) return "#00cc66";
    if (score < 60) return "#88cc00";
    if (score < 70) return "#cccc00";
    if (score < 80) return "#ff9900";
    if (score < 90) return "#ff6600";
    return "#ff3300";
}

const SUB_COLORS = {
    "input_norm": "#88cc88",
    "attn":       "#4488ff",
    "attn_norm":  "#66aacc",
    "mlp":        "#ff8844",
    "post_norm":  "#cccc88",
};

function getSubColor(blockName) {
    for (const [sub, color] of Object.entries(SUB_COLORS)) {
        if (blockName.endsWith("_" + sub)) return color;
    }
    if (blockName === "embed_tokens") return "#cc88cc";
    if (blockName === "final_norm") return "#88cccc";
    return "#5599ff";
}

function getSubBg(blockName, enabled) {
    if (!enabled) return "#1e1e1e";
    if (blockName.endsWith("_attn")) return "#28282d";
    if (blockName.endsWith("_mlp")) return "#2d2828";
    if (blockName.endsWith("_input_norm") || blockName.endsWith("_post_norm")) return "#282d28";
    if (blockName.endsWith("_attn_norm")) return "#282d2d";
    return "#2a2a2a";
}

const NUM_LAYERS = 36;

const EMBED_BLOCKS = ["embed_tokens"];

const LAYER_SUBS = ["input_norm", "attn", "attn_norm", "mlp", "post_norm"];
const LAYER_BLOCKS = [];
for (let i = 0; i < NUM_LAYERS; i++) {
    for (const sub of LAYER_SUBS) LAYER_BLOCKS.push(`l${i}_${sub}`);
}

const FINAL_BLOCKS = ["final_norm"];

const ALL_BLOCKS = [...EMBED_BLOCKS, ...LAYER_BLOCKS, ...FINAL_BLOCKS];

function makePreset(enabled, strength, overrideFn) {
    const overrides = {};
    if (overrideFn) {
        for (const b of ALL_BLOCKS) {
            const val = overrideFn(b);
            if (val !== null) overrides[b] = val;
        }
    }
    return { enabled, strength, overrides };
}

const DEEP_PRESETS = {
    "Custom": null,
    "Default": makePreset("ALL", 1.0, null),
    "Weaken ALL attn 90%": makePreset("ALL", 1.0, b => b.endsWith("_attn") ? 0.90 : null),
    "Weaken ALL attn 85%": makePreset("ALL", 1.0, b => b.endsWith("_attn") ? 0.85 : null),
    "Weaken ALL attn 80%": makePreset("ALL", 1.0, b => b.endsWith("_attn") ? 0.80 : null),
    "Weaken ALL mlp 90%": makePreset("ALL", 1.0, b => b.endsWith("_mlp") ? 0.90 : null),
    "Weaken ALL mlp 85%": makePreset("ALL", 1.0, b => b.endsWith("_mlp") ? 0.85 : null),
    "Weaken ALL attn+mlp 90%": makePreset("ALL", 1.0, b => (b.endsWith("_attn") || b.endsWith("_mlp")) ? 0.90 : null),
    "Weaken ALL attn+mlp 85%": makePreset("ALL", 1.0, b => (b.endsWith("_attn") || b.endsWith("_mlp")) ? 0.85 : null),
    "Weaken ALL norms 90%": makePreset("ALL", 1.0, b => (b.endsWith("_input_norm") || b.endsWith("_post_norm") || b.endsWith("_attn_norm") || b === "final_norm") ? 0.90 : null),
    "Global 95%": makePreset("ALL", 0.95, null),
    "Global 90%": makePreset("ALL", 0.90, null),
    "Global 85%": makePreset("ALL", 0.85, null),
};

app.registerExtension({
    name: "Qwen3_8BTextEncoderDebiaser.SubComponentControl",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Qwen3_8BTextEncoderDebiaser") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                if (node._qwen3_8bDeepInit) return;
                node._qwen3_8bDeepInit = true;
                node.combineBlockWidgets();
                node.setupPresetWidget();
                if (node.size[0] < 520) {
                    node.size[0] = 520;
                    node.setDirtyCanvas(true);
                }
            }, 50);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                for (const blockName of ALL_BLOCKS) {
                    const strWidget = node.widgets.find(w => w.name === blockName + "_str");
                    if (strWidget) {
                        let val = parseFloat(strWidget.value);
                        if (isNaN(val)) strWidget.value = 1.0;
                    }
                }
                node.setDirtyCanvas(true);
            }, 150);
        };

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(output) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            if (output && output.analysis_json && output.analysis_json[0]) {
                try {
                    const jsonStr = output.analysis_json[0];
                    if (jsonStr && jsonStr.length > 2) {
                        this._analysisData = JSON.parse(jsonStr);
                        this.setDirtyCanvas(true);
                    }
                } catch (e) {}
            }
        };

        nodeType.prototype.combineBlockWidgets = function() {
            const widgetPairs = [];
            const strWidgetNames = new Set();
            for (const widget of this.widgets) {
                if (widget.name.endsWith('_str')) strWidgetNames.add(widget.name);
            }
            for (const widget of this.widgets) {
                const strName = widget.name + '_str';
                if (strWidgetNames.has(strName)) {
                    const strWidget = this.widgets.find(w => w.name === strName);
                    if (strWidget) {
                        widgetPairs.push({ toggle: widget, strength: strWidget, name: widget.name });
                    }
                }
            }
            for (const pair of widgetPairs) this.createCombinedWidget(pair);
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true);
        };

        nodeType.prototype.createCombinedWidget = function(pair) {
            const { toggle, strength, name } = pair;

            toggle.draw = function(ctx, node, widgetWidth, y, widgetHeight) {
                const margin = 10;
                const checkboxSize = 14;
                const labelWidth = 115;
                const valueWidth = 38;
                const gap = 6;
                const sliderWidth = widgetWidth - margin - checkboxSize - gap - labelWidth - gap - valueWidth - margin - gap;
                const checkboxX = margin;
                const labelX = checkboxX + checkboxSize + gap;
                const sliderX = labelX + labelWidth + gap;

                const enabled = Boolean(toggle.value);
                let strengthVal = parseFloat(strength.value);
                if (isNaN(strengthVal)) strengthVal = 1.0;

                let impactScore = null;
                if (node._analysisData && node._analysisData.blocks) {
                    const blockData = node._analysisData.blocks[name];
                    if (blockData && typeof blockData.score === 'number') impactScore = blockData.score;
                }

                let checkboxColor = getSubColor(name);
                if (impactScore !== null) checkboxColor = getImpactColor(impactScore);

                ctx.fillStyle = getSubBg(name, enabled);
                ctx.fillRect(0, y, widgetWidth, widgetHeight);

                ctx.strokeStyle = enabled ? checkboxColor : "#555";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(checkboxX, y + (widgetHeight - checkboxSize) / 2, checkboxSize, checkboxSize);
                ctx.globalAlpha = enabled ? 1.0 : 0.35;
                ctx.fillStyle = checkboxColor;
                ctx.fillRect(checkboxX + 2, y + (widgetHeight - checkboxSize) / 2 + 2, checkboxSize - 4, checkboxSize - 4);
                ctx.globalAlpha = 1.0;

                ctx.fillStyle = enabled ? "#ddd" : "#666";
                ctx.font = "11px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                let displayLabel = name;
                const maxLabelWidth = labelWidth - 4;
                if (ctx.measureText(displayLabel).width > maxLabelWidth) {
                    while (ctx.measureText(displayLabel + "…").width > maxLabelWidth && displayLabel.length > 3) {
                        displayLabel = displayLabel.slice(0, -1);
                    }
                    displayLabel += "…";
                }
                ctx.fillText(displayLabel, labelX, y + widgetHeight / 2);

                const trackY = y + widgetHeight / 2;
                const trackHeight = 4;
                const min = -2.0, max = 2.0, range = max - min;
                const normalizedStrength = (strengthVal - min) / range;
                const zeroPos = (0 - min) / range;

                ctx.fillStyle = "#333";
                ctx.beginPath();
                ctx.roundRect(sliderX, trackY - trackHeight / 2, sliderWidth, trackHeight, 2);
                ctx.fill();

                ctx.fillStyle = enabled ? (strengthVal >= 0 ? "#5599ff" : "#ff6655") : "#444";
                const zeroX = sliderX + zeroPos * sliderWidth;
                const strengthX = sliderX + normalizedStrength * sliderWidth;
                ctx.beginPath();
                ctx.roundRect(Math.min(zeroX, strengthX), trackY - trackHeight / 2, Math.abs(strengthX - zeroX), trackHeight, 2);
                ctx.fill();

                ctx.fillStyle = enabled ? "#fff" : "#666";
                ctx.beginPath();
                ctx.arc(sliderX + normalizedStrength * sliderWidth, trackY, 5, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = enabled ? "#ddd" : "#555";
                ctx.textAlign = "right";
                ctx.font = "10px Arial";
                ctx.fillText(strengthVal.toFixed(2), widgetWidth - margin, y + widgetHeight / 2);
            };

            toggle.sliderInfo = {
                margin: 10, checkboxSize: 14, labelWidth: 115, valueWidth: 38, gap: 6,
                min: -2.0, max: 2.0, step: 0.01,
                getLayout: function(widgetWidth) {
                    const sliderWidth = widgetWidth - this.margin - this.checkboxSize - this.gap - this.labelWidth - this.gap - this.valueWidth - this.margin - this.gap;
                    const checkboxX = this.margin;
                    const sliderX = checkboxX + this.checkboxSize + this.gap + this.labelWidth + this.gap;
                    return { sliderWidth, checkboxX, sliderX };
                }
            };

            const originalMouse = toggle.mouse?.bind(toggle);
            toggle.mouse = function(event, pos, node) {
                const widgetWidth = node.size[0];
                const layout = toggle.sliderInfo.getLayout(widgetWidth);
                const valueX = widgetWidth - toggle.sliderInfo.margin - toggle.sliderInfo.valueWidth;
                const localX = pos[0];

                // Click on the numeric value column => prompt for direct entry.
                // Without this, clicks fell through to the boolean toggle handler
                // and ended up disabling the row instead of editing the value.
                if (event.type === "pointerdown" && localX >= valueX - 4) {
                    const current = parseFloat(strength.value);
                    const seed = isNaN(current) ? "1.0" : current.toFixed(2);
                    const input = window.prompt(
                        `Set strength (${toggle.sliderInfo.min.toFixed(1)} to ${toggle.sliderInfo.max.toFixed(1)}):`,
                        seed
                    );
                    if (input !== null && input !== "") {
                        const parsed = parseFloat(input);
                        if (!isNaN(parsed)) {
                            let v = Math.max(toggle.sliderInfo.min, Math.min(toggle.sliderInfo.max, parsed));
                            v = Math.round(v / toggle.sliderInfo.step) * toggle.sliderInfo.step;
                            strength.value = v;
                            node.setDirtyCanvas(true);
                        }
                    }
                    return true;
                }

                // Slider drag — RELATIVE / anchored (matches the flux_klein debiaser, #42).
                // Grab at the current value instead of jumping to the cursor; adjust from
                // the grab point so small drags give fine control and you never have to
                // drag to the track edge to reach min/max.
                if (event.type === "pointerdown" && localX >= layout.sliderX - 5 && localX <= layout.sliderX + layout.sliderWidth + 5) {
                    toggle._strDrag = { x: localX, v: parseFloat(strength.value) || 0 };
                    return true;
                }
                if (event.type === "pointermove" && toggle._strDrag) {
                    if (event.buttons === 0) {
                        toggle._strDrag = null;
                    } else {
                        let newStr = toggle._strDrag.v + ((localX - toggle._strDrag.x) / layout.sliderWidth) * (toggle.sliderInfo.max - toggle.sliderInfo.min);
                        newStr = Math.round(newStr / toggle.sliderInfo.step) * toggle.sliderInfo.step;
                        newStr = Math.max(toggle.sliderInfo.min, Math.min(toggle.sliderInfo.max, newStr));
                        strength.value = newStr;
                        node.setDirtyCanvas(true);
                        return true;
                    }
                }
                if (event.type === "pointerup" && toggle._strDrag) {
                    toggle._strDrag = null;
                    return true;
                }
                if (originalMouse) return originalMouse(event, pos, node);
                return false;
            };

            strength.draw = function() {};
            strength.computeSize = function() { return [0, -4]; };
        };

        nodeType.prototype.setupPresetWidget = function() {
            const node = this;
            const presetWidget = this.widgets.find(w => w.name === "preset");
            if (!presetWidget) return;
            const presetNames = Object.keys(DEEP_PRESETS);
            presetWidget.options = presetWidget.options || {};
            presetWidget.options.values = presetNames;
            if (!presetNames.includes(presetWidget.value)) presetWidget.value = "Default";
            const origCallback = presetWidget.callback;
            presetWidget.callback = function(value) {
                node.applyPreset(value);
                if (origCallback) origCallback.call(this, value);
            };
            presetWidget.serializeValue = function() { return "Custom"; };
            this.presetWidget = presetWidget;
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true);
        };

        nodeType.prototype.applyPreset = function(presetName) {
            const preset = DEEP_PRESETS[presetName];
            if (!preset) return;
            const enabledBlocks = preset.enabled === "ALL" ? ALL_BLOCKS : preset.enabled;
            const enabledSet = new Set(enabledBlocks);
            const baseStrength = preset.strength;
            const overrides = preset.overrides || {};
            for (const blockName of ALL_BLOCKS) {
                const toggleWidget = this.widgets.find(w => w.name === blockName);
                const strWidget = this.widgets.find(w => w.name === blockName + "_str");
                if (toggleWidget) toggleWidget.value = enabledSet.has(blockName);
                if (strWidget) strWidget.value = overrides.hasOwnProperty(blockName) ? overrides[blockName] : baseStrength;
            }
            this.setDirtyCanvas(true);
        };
    }
});
